"""Evaluate the leaf target detector with paraphrased instructions (prereg: docs/eval/paraphrase_prereg.md).

Embeddings are computed at evaluation time with the same VLM text tower and mean pooling as embed_language.py;
the 4 original sentences go through the same path and are checked against lang.pt.

    python scripts/pcvla/eval_paraphrase.py --ckpt output/adeept/pcvla/leaf_ue_mj_target/ckpt.pt --out output/adeept/eval/paraphrase_final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harvest_env as H  # noqa: E402
import train as T  # noqa: E402
from make_tea_demos import platform_origin  # noqa: E402
from target_net import TargetNet, CROP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARA = {
    "harvest": ["pick one green tea leaf and drop it into the basket",
                "take any green leaf off the tea plant and place it in the basket",
                "harvest a single green tea leaf into the basket"],
    "harvest_left": ["pick the green tea leaf on the left side and drop it into the basket",
                     "take the leftmost green tea leaf and place it in the basket",
                     "harvest the left green leaf into the basket"],
    "harvest_middle": ["pick the green tea leaf in the center and drop it into the basket",
                       "take the central green tea leaf and place it in the basket",
                       "harvest the middle green leaf into the basket"],
    "harvest_right": ["pick the green tea leaf on the right side and drop it into the basket",
                      "take the rightmost green tea leaf and place it in the basket",
                      "harvest the right green leaf into the basket"],
}


def embed(sentences):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    vlm = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    proc = AutoProcessor.from_pretrained(vlm)
    model = AutoModelForImageTextToText.from_pretrained(vlm, dtype=torch.float32).eval()
    out = {}
    with torch.no_grad():
        for s in sentences:
            out[s] = model.model.text_model(**proc.tokenizer(s, return_tensors="pt")).last_hidden_state[0].mean(0)
    return out


def run_episode(env, net, emb, task, seed):
    origin = platform_origin(np.random.default_rng(seed)); env.origin = np.asarray(origin)
    env.reset(seed, task=task)
    r = np.random.default_rng(seed)
    cloud = env.cloud(r)
    with torch.no_grad():
        pred = net(torch.from_numpy(T.crop_resample(cloud, r))[None].float().cuda(), emb[None].float().cuda())[0].cpu().numpy()
    true_target = env.target
    chosen = int(np.argmin([np.linalg.norm(pred - s) for s in env.spots]))
    if H.TASKS[task].target == "any":
        env.target = chosen
    err = float(np.linalg.norm(pred - env.spots[env.target]))
    env.spots = list(env.spots); env.spots[env.target] = pred.astype(float)
    try:
        ex = H.Expert(env); done, success = False, False
        while not done:
            _, success, done = env.step(ex.act())
    except RuntimeError:
        success = False
    wrong = H.TASKS[task].target != "any" and any(env.picked[j] for j in range(H.N_SHOOTS) if j != true_target)
    return bool(success), bool(wrong), err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed0", type=int, default=100000)
    p.add_argument("--episodes", type=int, default=10)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T.CROP = np.array([float(v) for v in CROP.split(",")]).reshape(3, 2)
    orig = {k: t.language for k, t in H.TASKS.items()}
    emb = embed(list(orig.values()) + [s for v in PARA.values() for s in v])
    table = torch.load(ROOT / "output" / "adeept" / "pcvla" / "lang.pt")
    check = {s: float((emb[s] - table[s]).abs().max()) for s in orig.values()}
    print("embedding check (max abs diff vs lang.pt):", check, flush=True)
    net = TargetNet().cuda().eval(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    env = H.HarvestEnv(renderer="mujoco", size=(240, 320))
    rows, rows_orig, rows_para = [], [], []
    for task in H.TASKS:
        for ep in range(a.episodes):
            seed = a.seed0 + ep
            s, w, e = run_episode(env, net, emb[orig[task]], task, seed)
            rows.append(dict(task=task, sentence=orig[task], kind="original", seed=seed, success=s, wrong=w, err=e))
            rows_orig.append((f"{task}_{seed}", float(s)))
            ps = []
            for k, sent in enumerate(PARA[task]):
                s2, w2, e2 = run_episode(env, net, emb[sent], task, seed)
                rows.append(dict(task=task, sentence=sent, kind=f"para{k}", seed=seed, success=s2, wrong=w2, err=e2))
                ps.append(float(s2))
            rows_para.append((f"{task}_{seed}", float(np.mean(ps))))
            print(task, seed, "orig", s, "para", ps, flush=True)
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    for name, rr in (("original.csv", rows_orig), ("paraphrase.csv", rows_para)):
        (out / name).write_text("episode,score\n" + "".join(f"{e},{v:.4f}\n" for e, v in rr))
    summ = {}
    for task in H.TASKS:
        for kind in ["original", "para0", "para1", "para2"]:
            r = [x for x in rows if x["task"] == task and x["kind"] == kind]
            summ[f"{task}/{kind}"] = dict(sentence=r[0]["sentence"], success=np.mean([x["success"] for x in r]),
                                          wrong=np.mean([x["wrong"] for x in r]), median_err_mm=1000 * float(np.median([x["err"] for x in r])))
    o = np.mean([x["success"] for x in rows if x["kind"] == "original"]); q = np.mean([x["success"] for x in rows if x["kind"] != "original"])
    summ["overall"] = dict(original=o, paraphrase=q, relative_drop=(o - q) / o if o else None, embedding_check=check)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    for k, v in summ.items():
        print(k, v)


if __name__ == "__main__":
    main()
