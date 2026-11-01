"""Train the point-cloud VLA (scripts/pcvla/model.py) on the recorded tea-harvest episodes.

    python scripts/pcvla/train.py --data output/adeept/datasets/tea_ue_pc --out output/adeept/pcvla/run1
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PCVLA  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def normalize_servo(x):
    """servo degrees -> about [-1, 1]: arm axes around 90, gripper over 40..150."""
    x = np.asarray(x, np.float32).copy()
    x[..., :4] = (x[..., :4] - 90.0) / 90.0
    x[..., 4] = (x[..., 4] - 95.0) / 55.0
    return x


def denormalize_servo(x):
    x = np.asarray(x, np.float32).copy()
    x[..., :4] = x[..., :4] * 90.0 + 90.0
    x[..., 4] = x[..., 4] * 55.0 + 95.0
    return x


CROP = None   # optional (3, 2) bounds in the platform frame; set from --crop


def crop_resample(pts, rng, n=2048):
    """Keep points inside CROP (if set) and resample to n points (with replacement if fewer)."""
    if CROP is None:
        return pts
    m = np.all((pts[:, :3] >= CROP[:, 0]) & (pts[:, :3] <= CROP[:, 1]), axis=1)
    p = pts[m] if m.sum() >= 32 else pts
    return p[rng.choice(len(p), n, replace=len(p) < n)]


class Episodes:
    def __init__(self, folder, lang, horizon, seed=0):
        self.eps = []
        for f in [f for d in str(folder).split(",") for f in sorted(Path(d).glob("episode_*.npz"))]:
            z = np.load(f)
            self.eps.append(dict(points=z["points"], state=normalize_servo(z["state"]), action=normalize_servo(z["action"]),
                                 lang=lang[str(z["task"])].numpy()))
        self.index = [(e, t) for e, ep in enumerate(self.eps) for t in range(len(ep["state"]))]
        self.h = horizon
        self.rng = np.random.default_rng(seed)

    def batch(self, n, augment=True):
        sel = self.rng.integers(len(self.index), size=n)
        P, L, S, A = [], [], [], []
        for i in sel:
            e, t = self.index[i]; ep = self.eps[e]
            pts = crop_resample(ep["points"][t].astype(np.float32), self.rng)
            if augment:
                pts[:, :3] += self.rng.normal(0, 0.002, (len(pts), 3))
                pts[:, 3:] = np.clip(pts[:, 3:] * self.rng.uniform(.85, 1.15) + self.rng.normal(0, .03, 3), 0, 1)
            T = len(ep["action"])
            idx = np.minimum(np.arange(t, t + self.h), T - 1)
            P.append(pts); L.append(ep["lang"]); S.append(ep["state"][t]); A.append(ep["action"][idx])
        f = lambda x: torch.from_numpy(np.stack(x)).float()
        return f(P), f(L), f(S), f(A)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "output" / "adeept" / "datasets" / "tea_ue_pc"))
    p.add_argument("--lang", default=str(ROOT / "output" / "adeept" / "pcvla" / "lang.pt"))
    p.add_argument("--out", default=str(ROOT / "output" / "adeept" / "pcvla" / "run1"))
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop", default="", help="x0,x1,y0,y1,z0,z1 (platform frame) to focus the cloud on the shoots")
    p.add_argument("--groups", type=int, default=64)
    p.add_argument("--no-state", action="store_true", help="zero the proprioceptive input (forces use of the point cloud)")
    a = p.parse_args()
    global CROP
    if a.crop:
        CROP = np.array([float(v) for v in a.crop.split(",")]).reshape(3, 2)
    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lang = torch.load(a.lang)
    data = Episodes(a.data, lang, a.horizon, a.seed)
    dev = torch.device("cuda")
    model = PCVLA(horizon=a.horizon, n_groups=a.groups).to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    cfg = dict(vars(a), n_episodes=len(data.eps), n_frames=len(data.index), params=sum(x.numel() for x in model.parameters()))
    (out / "config.json").write_text(json.dumps(cfg, indent=1)); print(cfg, flush=True)
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        P, L, S, A = (x.to(dev, non_blocking=True) for x in data.batch(a.batch))
        if a.no_state:
            S = torch.zeros_like(S)
        loss = model.loss(P, L, S, A)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        run = loss.item() if step == 1 else .98 * run + .02 * loss.item()
        if step % 200 == 0:
            print(f"step {step} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step % 5000 == 0 or step == a.steps:
            torch.save({"model": ema.state_dict(), "config": cfg}, out / f"ckpt_{step:06d}.pt")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
