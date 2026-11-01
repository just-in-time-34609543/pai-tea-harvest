"""Merge the two parallel demo collections (tea_ue_a, tea_ue_b) into tea_ue (LeRobot, for SmolVLA) and tea_ue_pc
(point-cloud episodes, renumbered, for PC-VLA)."""
import shutil
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

D = Path(__file__).resolve().parents[1] / "output" / "adeept" / "datasets"
parts = sys.argv[1:] or ["tea_ue_a", "tea_ue_b"]
shutil.rmtree(D / "tea_ue", ignore_errors=True)
aggregate_datasets([f"local/{p}" for p in parts], "local/tea_ue", roots=[D / p for p in parts], aggr_root=D / "tea_ue")
out = D / "tea_ue_pc"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
n = 0
for p in parts:
    for f in sorted((D / f"{p}_pc").glob("episode_*.npz")):
        shutil.copy(f, out / f"episode_{n:06d}.npz"); n += 1
print("merged", parts, "pc episodes", n)
