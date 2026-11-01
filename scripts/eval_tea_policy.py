"""Closed-loop evaluation of a SmolVLA checkpoint on the tea-harvest task, images rendered by Unreal.

    python scripts/eval_tea_policy.py --policy <ckpt>/pretrained_model --episodes 10
    python scripts/eval_tea_policy.py --expert --episodes 10

Held-out seeds 100000+ (demos use 0+); the platform position along the centre row is drawn from the seed as in
make_tea_demos.py. Prints the success rate per task, and also counts wrong-shoot picks (a non-target shoot plucked).
Saves one video per task (camera1 | camera2) and results.json in --out.
"""
import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_env as H  # noqa: E402
from make_tea_demos import platform_origin  # noqa: E402
from ue_render_client import UERenderClient  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy")
    p.add_argument("--pcvla", help="point-cloud VLA checkpoint (scripts/pcvla/train.py)")
    p.add_argument("--exec-steps", type=int, default=8, help="PC-VLA: actions executed per predicted chunk")
    p.add_argument("--expert", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--tasks", default=",".join(H.TASKS))
    p.add_argument("--n-action-steps", type=int)
    p.add_argument("--port", type=int, default=5600)
    p.add_argument("--size", default="240x320")
    p.add_argument("--renderer", default="ue", choices=["ue", "mujoco"])
    p.add_argument("--seed0", type=int, default=100000, help="first evaluation seed (demos used 0+)")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "output" / "adeept" / "eval"))
    a = p.parse_args()
    if not (a.policy or a.expert or a.pcvla):
        p.error("give --policy, --pcvla or --expert")
    if a.pcvla:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent / "pcvla"))
        from model import PCVLA
        import train as pctrain
        from train import normalize_servo, denormalize_servo
        ck = torch.load(a.pcvla, map_location="cpu")
        if ck["config"].get("crop"):
            pctrain.CROP = np.array([float(v) for v in ck["config"]["crop"].split(",")]).reshape(3, 2)
        pc_model = PCVLA(horizon=ck["config"]["horizon"], n_groups=ck["config"].get("groups", 64)).cuda().eval()
        pc_model.load_state_dict(ck["model"])
        lang = torch.load(ck["config"]["lang"])
    if a.policy:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.control_utils import predict_action
        device = torch.device("cuda")
        policy = SmolVLAPolicy.from_pretrained(a.policy).to(device).eval()
        if a.n_action_steps:
            policy.config.n_action_steps = a.n_action_steps
        pre, post = make_pre_post_processors(policy.config, pretrained_path=a.policy,
                                             preprocessor_overrides={"device_processor": {"device": device.type}})
    h, w = map(int, a.size.split("x"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = H.HarvestEnv(renderer="none" if a.renderer == "ue" else "mujoco", size=(h, w))
    results, wrong = {}, {}
    for task in a.tasks.split(","):
        ok = bad = 0
        for ep in range(a.episodes):
            seed = a.seed0 + ep
            origin = platform_origin(np.random.default_rng(seed))
            env.ue = UERenderClient(origin=origin, port=a.port, depth=bool(a.pcvla)) if a.renderer == "ue" else None
            env.origin = np.asarray(origin)
            queue = []
            obs = env.reset(seed, task=task)
            if a.policy:
                policy.reset()
            else:
                expert = H.Expert(env)
            frames, done, success = [], False, False
            while not done:
                if a.pcvla:
                    if not queue:
                        r = np.random.default_rng(env.ticks)
                        raw = env.ue.cloud(env, r) if env.ue else env.cloud(r)
                        pts = torch.from_numpy(pctrain.crop_resample(raw, r)).float()[None].cuda()
                        st = torch.from_numpy(normalize_servo(env.state)).float()[None].cuda()
                        if ck["config"].get("no_state"):
                            st = torch.zeros_like(st)
                        chunk = pc_model.sample(pts, lang[env.language][None].float().cuda(), st)[0].cpu().numpy()
                        queue = list(denormalize_servo(chunk)[:a.exec_steps])
                    act = queue.pop(0)
                elif a.policy:
                    act = predict_action(obs, policy, device, pre, post, use_amp=False, task=env.language, robot_type="adeept_ada031_tea_ue")
                    act = act.squeeze(0).cpu().numpy()[:5]
                else:
                    act = expert.act()
                obs, success, done = env.step(act)
                if ep == 0:
                    frames.append(np.hstack([obs["observation.images.camera1"], obs["observation.images.camera2"]]))
            if env.ue:
                env.ue.close()
            ok += success
            bad += (sum(env.picked) > 1) if env.task.target == "any" else any(env.picked[j] for j in range(H.N_SHOOTS) if j != env.target)
            if ep == 0:
                imageio.mimsave(out / f"{task}.mp4", frames, fps=env.fps)
            print(f"{task} ep {ep}: {'success' if success else 'fail'} ticks {env.ticks} picked {env.picked} target {env.target}", flush=True)
        results[task], wrong[task] = ok / a.episodes, bad / a.episodes
    results["mean"] = float(np.mean([results[t] for t in a.tasks.split(",")]))
    (out / "results.json").write_text(json.dumps({"success": results, "wrong_shoot": wrong, "episodes": a.episodes}, indent=2))
    for t, r in results.items():
        print(f"{t:16s} {r:6.1%}  wrong {wrong.get(t, 0):5.1%}")


if __name__ == "__main__":
    main()
