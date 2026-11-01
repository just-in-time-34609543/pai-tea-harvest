"""Pluckable green mature leaf (visual mesh) for the harvest env, in the leaf body frame:
length along +z (tip up), width along +x (radial), blade normal along +y; blade centre at the origin.
Writes output/adeept/plants/leaf_green.obj and leaf_petiole.obj (petiole from the blade base down to the twig).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tea_plant_gen import Mesh, leaf_surface, sweep  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / "adeept" / "plants"
LENGTH, WIDTH = 0.050, 0.019


def main():
    v, uv, f = leaf_surface(LENGTH, WIDTH, fold=0.35, droop=0.04, nu=24, nv=8)
    # leaf frame (x along, y across, z normal) -> body frame (z up = along, x = across, y = normal); centre the blade
    v = np.stack([v[:, 1], v[:, 2], v[:, 0] - LENGTH / 2], 1)
    m = Mesh(); m.add(v, uv, f); m.write_obj(OUT / "leaf_green.obj")
    p = Mesh(); p.add(*sweep([[0, 0, -LENGTH / 2 + 0.004], [0, 0.002, -LENGTH / 2 - 0.006], [0, 0.003, -LENGTH / 2 - 0.014]], [.0016, .0015, .0016], n=10))
    p.write_obj(OUT / "leaf_petiole.obj")
    print("leaf meshes", OUT / "leaf_green.obj")


if __name__ == "__main__":
    main()
