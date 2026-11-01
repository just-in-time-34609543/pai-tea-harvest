"""Language-conditioned target-shoot detector on the fused point cloud (the perception half of a modular pipeline).

Given the cloud and the instruction ("... on the left / in the middle / on the right ...", or any shoot), regress the
3D grasp point of the instructed shoot in the platform frame. The arm motion is then planned from that point by the
Adeept IK (approach -> grasp -> pluck -> basket), i.e. AI detects the candidate, the controller executes.

    python scripts/pcvla/target_net.py train --data output/adeept/datasets/tea_mj3_pc --out output/adeept/pcvla/target_v1
    python scripts/pcvla/target_net.py eval  --ckpt output/adeept/pcvla/target_v1/ckpt.pt --renderer mujoco --episodes 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import PointTokenizer  # noqa: E402
import train as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CROP = "-0.02,0.24,-0.22,0.22,-0.03,0.26"


class TargetNet(nn.Module):
    def __init__(self, lang_dim=960, dim=256, n_groups=128):
        super().__init__()
        self.points = PointTokenizer(dim, n_groups=n_groups)
        self.lang = nn.Sequential(nn.LayerNorm(lang_dim), nn.Linear(lang_dim, dim))
        enc = nn.TransformerEncoderLayer(dim, 8, 4 * dim, 0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 4)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))

    def forward(self, pts, lang):
        p, _ = self.points(pts)
        x = torch.cat([self.query.expand(len(p), -1, -1), self.lang(lang)[:, None], p], 1)
        return self.head(self.enc(x)[:, 0]) * 0.1 + torch.tensor([0.10, 0.0, 0.075], device=pts.device)


def load_examples(folders, frames=20):
    """(cloud, instruction, target xyz, all shoot xyz) from the first frames of each episode (arm still away)."""
    import harvest_env as H
    from make_tea_demos import platform_origin
    env = H.HarvestEnv(renderer="none")
    out = []
    files = [f for folder in str(folders).split(",") for f in sorted(Path(folder).glob("episode_*.npz"))]
    for f in files:
        z = np.load(f)
        task = next(k for k, t in H.TASKS.items() if t.language == str(z["task"]))
        seed = int(z["seed"])
        env.origin = np.asarray(platform_origin(np.random.default_rng(seed)))
        env.reset(seed, task=task)
        assert env.target == int(z["target"]), (f, env.target, int(z["target"]))
        spots = np.stack(env.spots).astype(np.float32)
        for t in range(min(frames, len(z["points"]))):
            out.append((z["points"][t].astype(np.float32), str(z["task"]), spots[env.target], spots, H.TASKS[task].target == "any"))
    return out


def train(a):
    T.CROP = np.array([float(v) for v in CROP.split(",")]).reshape(3, 2)
    lang = torch.load(ROOT / "output" / "adeept" / "pcvla" / "lang.pt")
    ex = load_examples(a.data)
    print("examples", len(ex), flush=True)
    rng = np.random.default_rng(0)
    dev = torch.device("cuda")
    net = TargetNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0, run = time.time(), None
    for step in range(1, a.steps + 1):
        idx = rng.integers(len(ex), size=a.batch)
        P = []
        for i in idx:
            p = T.crop_resample(ex[i][0], rng)
            p[:, :3] += rng.normal(0, .0015, (len(p), 3))
            p[:, 3:] = np.clip(p[:, 3:] * rng.uniform(.85, 1.15) + rng.normal(0, .03, 3), 0, 1)
            P.append(p)
        P = torch.from_numpy(np.stack(P)).to(dev)
        L = torch.stack([lang[ex[i][1]] for i in idx]).float().to(dev)
        tgt = torch.from_numpy(np.stack([ex[i][2] for i in idx])).to(dev)
        allp = torch.from_numpy(np.stack([ex[i][3] for i in idx])).to(dev)
        anyt = torch.tensor([ex[i][4] for i in idx], device=dev)
        pred = net(P, L)
        d_t = (pred - tgt).abs().sum(-1)
        d_any = (pred[:, None] - allp).abs().sum(-1).min(1).values      # "any shoot": nearest one counts
        loss = torch.where(anyt, d_any, d_t).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = loss.item() if run is None else .98 * run + .02 * loss.item()
        if step % 200 == 0:
            print(f"step {step} L1(m) {run:.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save({"model": net.state_dict(), "crop": CROP}, out / "ckpt.pt")
    print("TRAIN_DONE", flush=True)


def evaluate(a):
    import harvest_env as H
    from make_tea_demos import platform_origin
    from ue_render_client import UERenderClient
    import imageio
    T.CROP = np.array([float(v) for v in CROP.split(",")]).reshape(3, 2)
    lang = torch.load(ROOT / "output" / "adeept" / "pcvla" / "lang.pt")
    ck = torch.load(a.ckpt, map_location="cpu")
    net = TargetNet().cuda().eval(); net.load_state_dict(ck["model"])
    env = H.HarvestEnv(renderer="none" if a.renderer == "ue" else "mujoco", size=(240, 320))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res, err = {}, {}
    for task in a.tasks.split(","):
        ok, errs = 0, []
        for ep in range(a.episodes):
            seed = a.seed0 + ep
            origin = platform_origin(np.random.default_rng(seed)); env.origin = np.asarray(origin)
            env.ue = UERenderClient(origin=origin, port=a.port, depth=True) if a.renderer == "ue" else None
            env.reset(seed, task=task)
            r = np.random.default_rng(seed)
            cloud = env.ue.cloud(env, r) if env.ue else env.cloud(r)
            with torch.no_grad():
                p = net(torch.from_numpy(T.crop_resample(cloud, r))[None].float().cuda(), lang[env.language][None].float().cuda())[0].cpu().numpy()
            # which shoot does the prediction point at? plan the expert motion to that shoot's predicted point
            d = [np.linalg.norm(p - s) for s in env.spots]
            chosen = int(np.argmin(d))
            if H.TASKS[task].target == "any":
                env.target = chosen
            errs.append(float(np.linalg.norm(p - env.spots[env.target])))
            env.spots = list(env.spots); env.spots[env.target] = p.astype(float)     # the controller only knows the prediction
            frames, done, success = [], False, False
            try:
                ex = H.Expert(env)
                while not done:
                    obs, success, done = env.step(ex.act())
                    if ep == 0 and "observation.images.camera1" in obs:
                        frames.append(np.hstack([obs["observation.images.camera1"], obs["observation.images.camera2"]]))
            except RuntimeError:
                success = False
            if env.ue:
                env.ue.close()
            ok += success
            if ep == 0 and frames:
                imageio.mimsave(out / f"{task}.mp4", frames, fps=env.fps)
            print(f"{task} ep {ep}: {'success' if success else 'fail'} err {errs[-1]*1000:.1f} mm target {env.target} picked {env.picked}", flush=True)
        res[task], err[task] = ok / a.episodes, float(np.median(errs))
    res["mean"] = float(np.mean([res[t] for t in a.tasks.split(",")]))
    (out / "results.json").write_text(json.dumps({"success": res, "median_err_m": err}, indent=2))
    for t, v in res.items():
        print(f"{t:16s} {v:6.1%}  median err {err.get(t, float('nan'))*1000:.1f} mm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--data", default=str(ROOT / "output" / "adeept" / "datasets" / "tea_mj3_pc"))
    p.add_argument("--out", default=str(ROOT / "output" / "adeept" / "pcvla" / "target_v1"))
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ckpt")
    p.add_argument("--renderer", default="mujoco")
    p.add_argument("--port", type=int, default=5600)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--tasks", default="harvest,harvest_left,harvest_middle,harvest_right")
    p.add_argument("--seed0", type=int, default=100000)
    a = p.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
