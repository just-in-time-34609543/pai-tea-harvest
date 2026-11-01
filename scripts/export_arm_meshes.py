"""Export every visible body of the harvest MuJoCo model (arm, platform, basket, shoots) as OBJ meshes for Unreal.

One OBJ per (body, colour) in the body frame, metres; shoots use the procedural tea shoot meshes instead.
Writes output/adeept/arm_meshes/{bodies.json, *.obj}. Unreal places each body every frame from MuJoCo poses.
"""
from pathlib import Path
import json
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_env  # noqa: E402
from tea_plant_gen import Mesh, sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def prim(m, gtype, size):
    """Triangle mesh of a MuJoCo primitive geom in its own frame."""
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = size
        v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
        return v, f
    if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, h = size[0], size[1]
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            pts = [[0, 0, -h - r * c] for c in (1.0, .7, .3)][::-1]
            pts = [[0, 0, -h - r]] + [[0, 0, -h - r * (1 - k / 4)] for k in range(1, 4)] + [[0, 0, h + r * k / 4] for k in range(0, 4)] + [[0, 0, h + r]]
            rad = [1e-4] + [r * np.sqrt(1 - (1 - k / 4) ** 2) for k in range(1, 4)] + [r * np.sqrt(1 - (k / 4) ** 2) for k in range(0, 4)] + [1e-4]
        else:
            pts, rad = [[0, 0, -h], [0, 0, -h], [0, 0, h], [0, 0, h]], [1e-4, r, r, 1e-4]
        v, uv, f = sweep(pts, rad, n=16)
        return np.array(v), f
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = size[0]
        pts = [[0, 0, -r + 2 * r * k / 8] for k in range(9)]
        rad = [max(1e-4, np.sqrt(max(0, r * r - p[2] ** 2))) for p in pts]
        v, uv, f = sweep(pts, rad, n=16)
        return np.array(v), f
    return None, None


def quat_mat(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(q, float)); return m.reshape(3, 3)


def main(out=None):
    out = Path(out or ROOT / "output" / "adeept" / "arm_meshes"); out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(harvest_env.build_xml())
    bodies = {}
    for b in range(model.nbody):
        name = model.body(b).name if b else "platform"   # world-frame geoms = base plate, platform, shelf, hangers
        if name.startswith("fixture"):
            continue
        groups = {}
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b or model.geom_group[g] >= 3 or model.geom_rgba[g][3] == 0 and model.geom_matid[g] < 0:
                continue
            rgba = model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else model.geom_rgba[g]
            if rgba[3] == 0:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                v = model.mesh_vert[va:va + vn].copy(); f = model.mesh_face[fa:fa + fn].tolist()
            else:
                v, f = prim(None, model.geom_type[g], model.geom_size[g])
                if v is None:
                    continue
            v = v @ quat_mat(model.geom_quat[g]).T + model.geom_pos[g]
            key = tuple(np.round(rgba[:3], 2))
            groups.setdefault(key, Mesh()).add(v, np.zeros((len(v), 2)), f)
        if name.startswith("shoot"):
            groups = {}     # Unreal uses the procedural shoot meshes for these bodies
        parts = []
        for k, (col, m) in enumerate(groups.items()):
            fn = f"{name}_{k}.obj"; m.write_obj(out / fn); parts.append(dict(obj=fn, rgb=list(map(float, col))))
        bodies[name] = parts
    # long grey-green stem under each simulated shoot (bud centre at the body origin), down into the canopy
    stem = Mesh()
    stem.add(*sweep([[0, 0, -harvest_env.SHOOT_STEM - 0.10], [0.004, 0, -0.15], [0.001, 0, -0.06], [0, 0, -0.03]], [.0034, .003, .0026, .0023], n=10))
    stem.write_obj(out / "shoot_long_stem.obj")
    (out / "bodies.json").write_text(json.dumps(bodies, indent=1))
    print({k: len(v) for k, v in bodies.items()})


if __name__ == "__main__":
    main()
