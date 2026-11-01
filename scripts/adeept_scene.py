"""Scaled-up tea greenhouse scene for the Adeept arm (MJCF, SI units).

Builds output/adeept/scene.xml from config/adeept_scene.json. If the Adeept arm MJCF is given
(--arm path/to/adeept_arm.xml), it is attached at arm_base_mm; otherwise the scene has no arm.
Leaves are free bodies held by weld constraints (released when grasped), as in the patrol model.
"""
from pathlib import Path
import argparse
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "adeept"


def nums(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def frame_segments(g):
    """Arch frame centre-lines in metres: 5 arches, eave and ridge purlins."""
    r = g["tube_radius_mm"] * 1e-3
    lo = g["x_min_mm"] * 1e-3 + r
    hi = (g["x_min_mm"] + g["length_mm"]) * 1e-3 - r
    half = g["width_mm"] * 5e-4 - r
    eave = g["eave_mm"] * 1e-3
    rise = g["height_mm"] * 1e-3 - r - eave
    segs = []
    for x in np.linspace(lo, hi, 5):
        pts = [[x, half * math.cos(t), eave + rise * math.sin(t)] for t in np.linspace(0, math.pi, 25)]
        pts = [[x, half, r]] + pts + [[x, -half, r]]
        segs += [(a, b, r) for a, b in zip(pts, pts[1:])]
    for y, z in [(half, eave), (-half, eave), (0, g["height_mm"] * 1e-3 - r)]:
        segs.append(([lo, y, z], [hi, y, z], r))
    return segs


def add_plant(world, i, c, s):
    """Trunk, two branch tips with a target leaf each, two old leaves and a bud (all fixed)."""
    x, y = (v * 1e-3 for v in c)
    z0 = 0.04  # bench top
    b = ET.SubElement(world, "body", name=f"plant{i}", pos=nums([x, y, z0]))
    bark, green, old = ".36 .24 .12 1", ".30 .50 .16 1", ".16 .34 .11 1"
    def g(name, **a):
        ET.SubElement(b, "geom", name=f"plant{i}_{name}", contype="4", conaffinity="2", **a)
    g("pot", type="cylinder", pos=nums([0, 0, .03 * s / 3]), size=nums([.035, .03 * s / 3]), rgba=".45 .30 .22 1")
    g("trunk", type="capsule", fromto=nums([0, 0, .05, 0, 0, .05 * s]), size=".004", rgba=bark)
    tips = []
    for j, dx in enumerate((-20, 20)):
        dx *= 1e-3 * s / 3 * 1.5
        top = [dx, 0, .069 * s]
        g(f"branch{j}", type="capsule", fromto=nums([0, 0, .05 * s, *top]), size=".0022", rgba=bark)
        tips.append(top)
    g("old_a", type="ellipsoid", pos=nums([.025, 0, .039 * s]), size=".03 .011 .002", euler="0 -.25 0", rgba=old)
    g("old_b", type="ellipsoid", pos=nums([-.028, 0, .043 * s]), size=".031 .011 .002", euler="0 .2 0", rgba=old)
    g("bud", type="ellipsoid", pos=nums([0, .03, .06 * s]), size=".004 .004 .012", rgba=".48 .65 .12 1")
    return [[x + t[0], y + t[1], z0 + t[2]] for t in tips]


def build(cfg, arm_xml=None):
    m = ET.Element("mujoco", model="adeept_tea_greenhouse")
    ET.SubElement(m, "compiler", angle="radian", autolimits="true")
    ET.SubElement(m, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", noslip_iterations="3")
    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1=".55 .52 .46", rgb2=".50 .47 .41", width="256", height="256")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8")
    world = ET.SubElement(m, "worldbody")
    ET.SubElement(world, "light", name="sun", pos="0.3 -0.4 1.2", dir="-0.2 0.3 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .01", material="floor_mat", contype="1", conaffinity="1")
    for k, (a, b, r) in enumerate(frame_segments(cfg["greenhouse"])):
        ET.SubElement(world, "geom", name=f"frame{k}", type="capsule", fromto=nums([*a, *b]), size=f"{r:.4g}",
                      rgba=".80 .82 .84 1", contype="0", conaffinity="0", group="1")
    bx, by, bz = (v * 1e-3 for v in cfg["bench"]["size_mm"])
    px, py, _ = (v * 1e-3 for v in cfg["bench"]["pos_mm"])
    ET.SubElement(world, "geom", name="bench", type="box", pos=nums([px, py, bz / 2]), size=nums([bx / 2, by / 2, bz / 2]),
                  rgba=".42 .33 .24 1", contype="1", conaffinity="1")
    tips = []
    for i, c in enumerate(cfg["plants"]["centers_mm"]):
        tips += add_plant(world, i, c, cfg["plants"]["scale"])
    # collection bin
    cx, cy = (v * 1e-3 for v in cfg["bin"]["center_mm"])
    ix, iy, iz = (v * 1e-3 for v in cfg["bin"]["inner_mm"])
    w = cfg["bin"]["wall_mm"] * 1e-3
    z0 = bz
    binb = ET.SubElement(world, "body", name="bin", pos=nums([cx, cy, z0]))
    for name, pos, size in [("floor", [0, 0, w / 2], [ix / 2 + w, iy / 2 + w, w / 2]),
                            ("wx0", [-(ix + w) / 2, 0, iz / 2], [w / 2, iy / 2 + w, iz / 2]),
                            ("wx1", [(ix + w) / 2, 0, iz / 2], [w / 2, iy / 2 + w, iz / 2]),
                            ("wy0", [0, -(iy + w) / 2, iz / 2], [ix / 2, w / 2, iz / 2]),
                            ("wy1", [0, (iy + w) / 2, iz / 2], [ix / 2, w / 2, iz / 2])]:
        ET.SubElement(binb, "geom", name=f"bin_{name}", type="box", pos=nums(pos), size=nums(size), rgba=".85 .78 .55 1")
    # target leaves: free bodies welded to fixtures at branch tips
    eq = ET.SubElement(m, "equality")
    lx, ly, lz = (v * 5e-4 for v in cfg["leaf"]["blade_mm"])
    for k, t in enumerate(tips):
        pos = [t[0], t[1], t[2] + lx * .2]
        leaf = ET.SubElement(world, "body", name=f"leaf{k}", pos=nums(pos))
        ET.SubElement(leaf, "freejoint", name=f"leaf{k}_free")
        ET.SubElement(leaf, "geom", name=f"leaf{k}_blade", type="ellipsoid", size=nums([lx, ly, max(lz, .0012)]),
                      euler="0 -1.2 0", pos=nums([0, 0, lx * .8]), mass=str(cfg["leaf"]["mass_g"] * 1e-3),
                      rgba=".42 .70 .20 1", friction="1.2 .02 .002", condim="4")
        ET.SubElement(world, "body", name=f"fixture{k}", pos=nums(pos))
        ET.SubElement(eq, "weld", name=f"attach{k}", body1=f"leaf{k}", body2=f"fixture{k}", solref=".004 1")
    ET.SubElement(world, "camera", name="overview", pos="0.55 -0.55 0.55", xyaxes="0.707 0.707 0 -0.38 0.38 0.84", fovy="50")
    if arm_xml:
        arm = ET.parse(arm_xml).getroot()
        # merge assets/defaults/actuators/sensors; mount the arm's top-level bodies on a base frame
        for tag in ("asset", "default", "actuator", "sensor", "contact", "tendon"):
            src = arm.find(tag)
            if src is not None:
                dst = m.find(tag) or ET.SubElement(m, tag)
                dst.extend(list(src))
        mount = ET.SubElement(world, "body", name="arm_mount", pos=nums(np.array(cfg["arm_base_mm"]) * 1e-3))
        for child in list(arm.find("worldbody")):
            if child.tag in ("body", "geom", "site", "camera"):
                mount.append(child)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", help="Adeept arm MJCF (self-contained)")
    a = ap.parse_args()
    cfg = json.load(open(ROOT / "config" / "adeept_scene.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    m = build(cfg, a.arm)
    ET.indent(m)
    out = OUT / ("scene.xml" if a.arm else "scene_noarm.xml")
    ET.ElementTree(m).write(out, encoding="unicode")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out))
    print(out, "nbody", model.nbody, "ngeom", model.ngeom, "neq", model.neq)


if __name__ == "__main__":
    main()
