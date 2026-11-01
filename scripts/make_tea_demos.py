"""Record scripted-expert tea-harvest demonstrations rendered by Unreal as a LeRobot dataset (SmolVLA input).

    python scripts/make_tea_demos.py --episodes 40 --root output/adeept/datasets/tea_ue

Physics: MuJoCo (scripts/harvest_env.py). Images: the Unreal render server (scripts/ue/render_server.py) must be
running on --port. Each episode places the arm platform at a random point along the centre row of the greenhouse,
so backgrounds vary. Only successful expert episodes are kept. Seeds start at 0; evaluation uses 100000+.
Frame layout follows the Adeept recorder: observation before the tick, the commanded servo targets as action.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_env as H  # noqa: E402
from ue_render_client import UERenderClient  # noqa: E402

JOINTS = ["base", "shoulder", "elbow", "wrist", "gripper"]


def features(h, w):
    img = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return {"observation.state": {"dtype": "float32", "shape": (5,), "names": JOINTS},
            "action": {"dtype": "float32", "shape": (5,), "names": JOINTS},
            "observation.images.camera1": img, "observation.images.camera2": dict(img)}


def platform_origin(rng):
    """Platform frame in the greenhouse: under the ceiling rail on the centre row, between two arches (the arch cameras
    of those two arches look at it)."""
    xs = H.arch_xs()
    k = int(rng.integers(1, len(xs) - 2))
    x = 0.5 * (xs[k] + xs[k + 1]) + rng.uniform(-0.12, 0.12)
    return (float(x), 0.0, float(1.85 + rng.uniform(-0.03, 0.03)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40, help="successful episodes per task")
    p.add_argument("--tasks", default=",".join(H.TASKS))
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "output" / "adeept" / "datasets" / "tea_ue"))
    p.add_argument("--repo-id", default="local/tea_ue")
    p.add_argument("--port", type=int, default=5600)
    p.add_argument("--size", default="240x320", help="HxW of rendered images (must match the render server)")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--pointcloud", action="store_true", help="also save fused RGB-D point clouds (episode npz files)")
    p.add_argument("--renderer", default="ue", choices=["ue", "mujoco"], help="image source; mujoco = fallback without Unreal")
    a = p.parse_args()
    h, w = map(int, a.size.split("x"))
    if a.overwrite:
        shutil.rmtree(a.root, ignore_errors=True)
    env = H.HarvestEnv(renderer="none" if a.renderer == "ue" else "mujoco", size=(h, w))
    ds = LeRobotDataset.create(a.repo_id, env.fps, features(h, w), root=a.root, robot_type="adeept_ada031_tea_ue")
    seed, t0 = a.seed0, time.time()
    try:
        for task in a.tasks.split(","):
            kept = 0
            while kept < a.episodes:
                rng = np.random.default_rng(seed)
                origin = platform_origin(rng)
                env.ue = UERenderClient(origin=origin, port=a.port, depth=a.pointcloud) if a.renderer == "ue" else None
                env.origin = np.asarray(origin)
                clouds = []
                try:
                    obs = env.reset(seed, task=task)
                    expert, frames, success = H.Expert(env, noise=a.noise), [], False
                    # record until the expert has finished its whole plan (gripper fully open over the basket),
                    # not just until the success test first passes, so the release is in the demonstrations
                    while env.ticks < env.max_ticks and not (expert.index >= len(expert.plan) and success):
                        action = expert.act()
                        if a.pointcloud:
                            clouds.append((env.ue.cloud(env, rng) if env.ue else env.cloud(rng)).astype(np.float16))
                        nxt, s, _ = env.step(action)
                        success = success or s
                        frames.append({**obs, "action": env.state.astype("float32"), "task": env.language})
                        obs = nxt
                    success = success and env.success()
                except RuntimeError as e:
                    print(f"{task} seed {seed}: {e}", flush=True); success = False
                finally:
                    if env.ue:
                        env.ue.close()
                seed += 1
                if not success:
                    print(f"{task} seed {seed - 1}: expert failed, skipped", flush=True)
                    continue
                for fr in frames:
                    ds.add_frame(fr)
                ds.save_episode()
                if a.pointcloud:
                    pc_dir = Path(a.root).parent / (Path(a.root).name + "_pc"); pc_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(pc_dir / f"episode_{ds.meta.total_episodes - 1:06d}.npz", points=np.stack(clouds),
                                        state=np.stack([f["observation.state"] for f in frames]),
                                        action=np.stack([f["action"] for f in frames]), task=env.language, seed=seed - 1,
                                        target=env.target, layout=json.dumps(env.layout))
                kept += 1
                print(f"{task}: {kept}/{a.episodes} ({len(frames)} frames) seed {seed - 1} elapsed {time.time() - t0:.0f}s", flush=True)
    finally:
        ds.finalize()


if __name__ == "__main__":
    main()
