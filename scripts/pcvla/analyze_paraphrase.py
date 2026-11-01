"""Why paraphrases fail: cosine similarity of each paraphrase to the 4 training sentences, and which leaf position the
detector picks for them (from the same seeds as eval_paraphrase.py; prediction only, no rollout).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harvest_env as H  # noqa: E402
import train as T  # noqa: E402
from eval_paraphrase import PARA, embed  # noqa: E402
from make_tea_demos import platform_origin  # noqa: E402
from target_net import TargetNet, CROP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ckpt, out = sys.argv[1], Path(sys.argv[2])
T.CROP = np.array([float(v) for v in CROP.split(",")]).reshape(3, 2)
orig = {k: t.language for k, t in H.TASKS.items()}
emb = embed(list(orig.values()) + [s for v in PARA.values() for s in v])
net = TargetNet().eval(); net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
summ = json.load(open(out / "summary.json"))
env = H.HarvestEnv(renderer="mujoco", size=(240, 320))
names = {"harvest": "任意", "harvest_left": "左", "harvest_middle": "真ん中", "harvest_right": "右"}
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
rows = []
for task, sents in PARA.items():
    for k, s in enumerate(sents):
        sims = {names[t]: cos(emb[s], emb[orig[t]]) for t in orig}
        picks = {"left": 0, "middle": 0, "right": 0}
        for seed in range(100000, 100010):
            env.origin = np.asarray(platform_origin(np.random.default_rng(seed))); env.reset(seed, task=task)
            r = np.random.default_rng(seed); cloud = env.cloud(r)
            with torch.no_grad():
                pred = net(torch.from_numpy(T.crop_resample(cloud, r))[None].float(), emb[s][None].float())[0].numpy()
            j = int(np.argmin([np.linalg.norm(pred - p) for p in env.spots]))
            picks[next(n for n, v in env.layout.items() if v == j)] += 1
        sm = summ[f"{task}/para{k}"]
        rows.append(dict(instruction=names[task], paraphrase=s, cos=sims, nearest=max(sims, key=sims.get), success=sm["success"],
                         wrong=sm["wrong"], picked_position=picks))
(out / "paraphrase_analysis.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
for r in rows:
    print(r["instruction"], "|", r["paraphrase"], "|", " ".join(f"{k}{v:.3f}" for k, v in r["cos"].items()), "| nearest", r["nearest"],
          "| succ", r["success"], "wrong", r["wrong"], "| picks", r["picked_position"])
