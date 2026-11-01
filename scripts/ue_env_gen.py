"""Visual-only environment for the Unreal scene, real scale (metres), from config/adeept_scene.json["world"].

Look-dev from the field photos: grey-green steel arch pipes (no film), a blue shade net on one side, a weedy ground
with ferns and leaf litter, dense tall tea bushes around, forested hills and blue sky. Also the ceiling rail that
carries the arm. No external assets; everything is generated from the config seed.
Frame: MuJoCo-style (x along the house, y across, z up); the UE script converts to cm with y negated.
"""
from pathlib import Path
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tea_plant_gen import Mesh, sweep, leaf_surface, frame_from, rot_axis, place  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def quad(m, p0, p1, p2, p3, uvscale=(1, 1)):
    su, sv = uvscale
    m.add([p0, p1, p2, p3], [[0, 0], [su, 0], [su, sv], [0, sv]], [[0, 1, 2], [0, 2, 3]])


def greenhouse(w):
    g, r = w["greenhouse"], w["greenhouse"]["pipe_r"]
    m = Mesh()
    lo, hi = g["x_min"], g["x_min"] + g["length"]
    half, eave, top = g["width"] / 2, g["eave"], g["height"]
    n_arch = int(round(g["length"] / g["arch_spacing"])) + 1
    for x in np.linspace(lo, hi, n_arch):
        # straight legs + elliptical roof, one continuous sweep
        pts = [[x, half, 0.0], [x, half, eave * .5]]
        pts += [[x, half * math.cos(t), eave + (top - eave) * math.sin(t)] for t in np.linspace(0, math.pi, 41)]
        pts += [[x, -half, eave * .5], [x, -half, 0.0]]
        m.add(*sweep(pts, [r] * len(pts), n=12, vscale=2))
    for y, z in [(half, eave), (-half, eave), (0, top), (half * .72, eave + (top - eave) * .70), (-half * .72, eave + (top - eave) * .70),
                 (half, eave * .45), (-half, eave * .45)]:
        m.add(*sweep([[lo, y, z], [hi, y, z]], [r * .85] * 2, n=12, vscale=2))
    return m


def rail(w):
    g, rl = w["greenhouse"], w["rail"]
    m = Mesh()
    lo, hi = g["x_min"] + .1, g["x_min"] + g["length"] - .1
    half, eave, top = g["width"] / 2, g["eave"], g["height"]
    for dy in (-rl["gauge"] / 2, rl["gauge"] / 2):
        m.add(*sweep([[lo, rl["y"] + dy, rl["z"]], [hi, rl["y"] + dy, rl["z"]]], [rl["r"]] * 2, n=10, vscale=2))
    n_arch = int(round(g["length"] / g["arch_spacing"])) + 1
    for x in np.linspace(g["x_min"], g["x_min"] + g["length"], n_arch)[1:-1]:
        for dy in (-rl["gauge"] / 2, rl["gauge"] / 2):
            y = rl["y"] + dy
            roof = eave + (top - eave) * math.sqrt(max(0, 1 - (y / half) ** 2))
            m.add(*sweep([[x, y, rl["z"]], [x, y, roof]], [.006] * 2, n=6))
    return m


def side_net(w):
    g, n = w["greenhouse"], w["net"]
    m = Mesh()
    lo, hi = g["x_min"], g["x_min"] + g["length"]
    y = n["side"] * (g["width"] / 2 + .02)
    quad(m, [lo, y, 0], [hi, y, 0], [hi, y, n["height"]], [lo, y, n["height"]], (g["length"] / .5, n["height"] / .5))
    return m


def ground(size=200.0):
    m = Mesh(); s = size / 2
    quad(m, [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], (size / 1.5, size / 1.5))
    return m


def hills(rng, radius, depth, n=240, rows=8):
    m = Mesh()
    h = lambda a: 14 + 9 * math.sin(2 * a + 1) + 5 * math.sin(5 * a + .3) + 2.5 * math.sin(11 * a)
    verts, uvs, faces = [], [], []
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        for j in range(rows + 1):
            s = j / rows
            z = h(a) * math.sin(math.pi * min(1, s * 1.5)) ** .7 + (rng.normal(0, .4) if 0 < j < rows else 0)
            r = radius + depth * s
            verts.append([r * math.cos(a), r * math.sin(a), z - .3]); uvs.append([i / n * 40, s * 5])
    W = rows + 1
    for i in range(n):
        for j in range(rows):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            faces += [[a, c, b], [b, c, d]]
    m.add(verts, uvs, faces)
    return m


# ---------------------------------------------------------------- ground cover (merged, scattered)
def scatter_xy(rng, n, half_x, half_y, avoid=()):
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y)
        if all((x - ax) ** 2 + (y - ay) ** 2 > r * r for ax, ay, r in avoid):
            pts.append((x, y))
    return pts


def grass_clump(rng, m, x, y):
    for _ in range(int(rng.integers(8, 18))):
        h = rng.uniform(.08, .32); wdt = rng.uniform(.004, .008)
        az = rng.uniform(0, 2 * math.pi); lean = rng.uniform(.1, .6)
        base = np.array([x + rng.normal(0, .02), y + rng.normal(0, .02), 0])
        d = np.array([math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)])
        side = np.array([-math.sin(az), math.cos(az), 0]) * wdt
        pts = [base + d * h * s + np.array([0, 0, -.25 * h * s * s * lean]) for s in (0, .5, 1)]
        verts = [pts[0] - side, pts[0] + side, pts[1] - side * .7, pts[1] + side * .7, pts[2], pts[2]]
        m.add(verts, [[0, 0], [1, 0], [0, .5], [1, .5], [.5, 1], [.5, 1]], [[0, 1, 2], [1, 3, 2], [2, 3, 4]])


def weed(rng, m, x, y):
    for k in range(int(rng.integers(4, 9))):
        L = rng.uniform(.03, .07); lv, luv, lf = leaf_surface(L, L * .55, fold=.3, droop=.3, nu=5, nv=2)
        az = rng.uniform(0, 2 * math.pi); el = math.radians(rng.uniform(10, 45))
        d = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        m.add(place(lv, frame_from(d), [x, y, .005 + k * .006]), luv, lf)


def fern(rng, m, x, y):
    for _ in range(int(rng.integers(3, 7))):
        L = rng.uniform(.25, .5); az = rng.uniform(0, 2 * math.pi); el = math.radians(rng.uniform(35, 65))
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        side = np.array([-math.sin(az), math.cos(az), 0])
        n = 14
        for i in range(1, n):
            s = i / n
            c = np.array([x, y, 0]) + d * L * s + np.array([0, 0, -.35 * L * s * s])
            pl = .06 * math.sin(math.pi * s) + .01
            for sgn in (-1, 1):
                tip = c + sgn * side * pl + d * pl * .4 + np.array([0, 0, -.01])
                m.add([c - d * .008, c + d * .008, tip], [[0, 0], [0, 1], [1, .5]], [[0, 1, 2]])


def litter(rng, m, x, y):
    L = rng.uniform(.05, .10); lv, luv, lf = leaf_surface(L, L * .3, fold=.2, droop=-.05, nu=4, nv=2)
    R = rot_axis([0, 0, 1], rng.uniform(0, 2 * math.pi)) @ rot_axis([1, 0, 0], rng.uniform(-.3, .3))
    m.add(place(lv, R, [x, y, .003]), luv, lf)


# ---------------------------------------------------------------- textures
def tex_metal(rng, n=512):
    base = np.ones((n, n, 3)) * np.array([.36, .41, .38]) + rng.normal(0, .02, (n, n, 3))
    img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    for _ in range(60):  # rust streaks and spots
        x = rng.uniform(0, n); y = rng.uniform(0, n); L = rng.uniform(6, 60)
        d.line([x, y, x + rng.uniform(-3, 3), y + L], fill=(118, 70, 38), width=int(rng.integers(1, 4)))
    return img.filter(ImageFilter.GaussianBlur(1.0))


def tex_ground(rng, n=1024):
    img = np.ones((n, n, 3)) * np.array([.16, .12, .08])
    moss = rng.random((n // 8, n // 8)) < .5
    moss = np.kron(moss, np.ones((8, 8))).astype(bool)
    img[moss] = np.array([.12, .17, .06])
    pil = Image.fromarray((np.clip(img + rng.normal(0, .03, img.shape), 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for _ in range(1500):
        x, y = rng.uniform(0, n, 2); L = rng.uniform(4, 14); a = rng.uniform(0, math.pi)
        col = tuple(int(c) for c in rng.choice([[92, 70, 40], [70, 50, 28], [104, 88, 50], [48, 66, 26]]))
        d.line([x, y, x + L * math.cos(a), y + L * math.sin(a)], fill=col, width=int(rng.integers(2, 4)))
    return pil.filter(ImageFilter.GaussianBlur(.8))


def tex_simple(rng, rgb, var=.05, n=256, streak=False):
    img = np.ones((n, n, 3)) * np.array(rgb) + rng.normal(0, var, (n, n, 3))
    if streak:
        img *= (0.85 + 0.3 * np.linspace(0, 1, n))[:, None, None]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))


def main(out=None):
    cfg = json.load(open(ROOT / "config" / "adeept_scene.json")); w = cfg["world"]
    out = Path(out or ROOT / "output" / "adeept" / "env"); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    greenhouse(w).write_obj(out / "arches.obj"); rail(w).write_obj(out / "rail.obj"); side_net(w).write_obj(out / "net.obj")
    ground().write_obj(out / "ground.obj"); hills(rng, w["hills"]["radius"], w["hills"]["depth"]).write_obj(out / "hills.obj")
    gc = w["ground_cover"]
    # keep ground cover off the bush bases inside the house
    bases = [(x, y, .25) for y in w["rows_y"] for x in np.arange(w["greenhouse"]["x_min"] + .5, w["greenhouse"]["x_min"] + w["greenhouse"]["length"], w["bush_spacing"])]
    for name, fn, n in (("grass", grass_clump, gc["grass"]), ("weeds", weed, gc["weeds"]), ("ferns", fern, gc["ferns"]), ("litter", litter, gc["litter"])):
        m = Mesh()
        for x, y in scatter_xy(rng, n, gc["half_x"], gc["half_y"], bases):
            fn(rng, m, x, y)
        m.write_obj(out / f"{name}.obj")
    tex_metal(rng).save(out / "tex_metal.png"); tex_ground(rng).save(out / "tex_ground.png")
    tex_simple(rng, [.10, .20, .06], streak=True).save(out / "tex_grass.png")
    tex_simple(rng, [.08, .22, .07], streak=True).save(out / "tex_fern.png")
    tex_simple(rng, [.12, .24, .08]).save(out / "tex_weed.png")
    tex_simple(rng, [.36, .26, .15], var=.08).save(out / "tex_litter.png")
    tex_simple(rng, [.12, .20, .09], var=.04).save(out / "tex_hills.png")
    # shade net weave: blue threads with small square holes (alpha), tiled every 0.5 m
    n = 512; img = Image.new("RGBA", (n, n), (20, 75, 150, 255)); d = ImageDraw.Draw(img)
    for i in range(0, n, 8):
        for j in range(0, n, 8):
            d.rectangle([i + 2, j + 2, i + 5, j + 5], fill=(20, 75, 150, 0))
    img.save(out / "tex_net.png")
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
