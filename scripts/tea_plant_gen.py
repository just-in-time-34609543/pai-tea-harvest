"""Procedural tea bush (visual meshes + procedural textures), modelled on reference photos of tea bushes (not included in this repository).

Real scale (metres). Look-dev targets taken from the photos:
- bush: several slender stems from the ground, 1.4-2.1 m tall, branching in the upper part, dense foliage;
- mature leaf: narrow elliptic (length ~3.5x width), acute tip, finely serrated margin, glossy dark green,
  pale yellow midrib; some leaves with yellowing margins, some with brown / purple spots;
- young shoot (harvest target) at a stem tip: a closed, slender, pointed terminal bud (pale tan, downy) and
  two half-folded young leaves rising from the nodes just below it at 30-45 deg, with mature leaves right under
  them (no bare stalk);
- stems: old wood red-brown, the upper / thin wood grey-green.

Everything is generated from a seed: no external assets. Per bush variant `v` (bush-local frame, base at origin):
  bush{v}_wood.obj, bush{v}_twig.obj, bush{v}_leaf_{a,b,c}.obj, bush{v}_bud.obj   static visual meshes
  shoot_stem.obj, shoot_bud.obj, shoot_young.obj                                  one harvest shoot (local frame)
and shoot poses (tips) per bush in bushes.json. Shared textures are PNGs in the same folder.
"""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEX = 512


# ---------------------------------------------------------------- textures
def leaf_texture(rng, kind):
    """Leaf albedo in UV space: u along the leaf (0 base .. 1 tip), v across (0 .. 1, midrib at .5)."""
    w, h = TEX * 2, TEX // 2
    u = np.linspace(0, 1, w)[None, :]
    v = np.linspace(0, 1, h)[:, None]
    d = np.abs(v - 0.5)
    if kind == "bud":
        base = np.array([0.62, 0.52, 0.36]) + np.array([0.08, 0.06, 0.02]) * u[..., None]
    elif kind == "young":  # pale yellow-green, tan toward the tip and margin
        g = np.array([0.50, 0.50, 0.30]); tan = np.array([0.64, 0.50, 0.38])
        k = np.clip(0.45 + 0.35 * u + 0.8 * np.clip(d - 0.3, 0, 1), 0, 1)
        base = g * (1 - k)[..., None] + tan * k[..., None]
    else:
        dark = np.array([0.10, 0.22, 0.06]); light = np.array([0.20, 0.36, 0.10])
        t = 0.45 + 0.15 * np.sin(2.3 * u + rng.uniform(0, 6)) + 0.25 * d
        base = dark * (1 - t)[..., None] + light * t[..., None]
    img = base * np.ones((h, w, 1))
    rib = np.exp(-(d / 0.010) ** 2)
    rib_col = {"bud": [.78, .70, .55], "young": [.80, .78, .55]}.get(kind, [.62, .66, .30])
    img = img * (1 - 0.75 * rib[..., None]) + np.array(rib_col) * 0.75 * rib[..., None]
    phase = (u * 11 - d * 4.5) % 1.0  # lateral veins curving toward the tip
    lat = np.exp(-((phase - 0.5) / 0.035) ** 2) * (d < 0.44) * (0.10 if kind in ("bud", "young") else 0.16)
    img = img * (1 - lat[..., None]) + np.array([0.40, 0.48, 0.20]) * lat[..., None]
    if kind == "b":  # yellowing margins
        edge = np.clip((d - 0.24) / 0.22, 0, 1) * (0.5 + 0.5 * u)
        img = img * (1 - edge[..., None]) + np.array([0.62, 0.58, 0.12]) * edge[..., None]
    img = np.clip(img + rng.normal(0, 0.018, img.shape), 0, 1)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    if kind in ("c", "b"):  # brown / purple spots with darker rims
        dr = ImageDraw.Draw(pil)
        for _ in range(int(rng.integers(3, 9) if kind == "c" else rng.integers(0, 3))):
            x, y = rng.uniform(0.15, 0.9) * w, rng.uniform(0.18, 0.82) * h
            r = rng.uniform(2, 8)
            col = (100, 55, 35) if rng.random() < .6 else (90, 40, 65)
            dr.ellipse([x - r * 1.6, y - r * 1.6, x + r * 1.6, y + r * 1.6], fill=(110, 85, 30))
            dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
    if kind in ("bud", "young"):  # down: fine light speckle, slightly soft
        sp = (rng.random((h, w)) < 0.05) * 35
        arr = np.array(pil).astype(int) + sp[..., None]
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    return pil


def wood_texture(rng, kind):
    w, h = TEX, TEX
    x = np.linspace(0, 1, w)[None, :]
    streak = 0.5 + 0.5 * np.sin(x * 50 + rng.normal(0, 1, (1, w)).cumsum() * 0.05)
    col = np.array([0.36, 0.20, 0.13]) if kind == "wood" else np.array([0.42, 0.44, 0.32])
    base = col * (0.85 + 0.15 * streak)[..., None] * np.ones((h, 1, 1)) + rng.normal(0, 0.025, (h, w, 3))
    pil = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    for _ in range(90 if kind == "wood" else 30):  # lenticels
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        dr.ellipse([cx - 3, cy - 1.2, cx + 3, cy + 1.2], fill=(140, 115, 90) if kind == "wood" else (150, 150, 120))
    if kind == "twig":
        pil = pil.filter(ImageFilter.GaussianBlur(0.8))
    return pil


# ---------------------------------------------------------------- meshes
class Mesh:
    def __init__(self):
        self.v, self.t, self.f = [], [], []

    def add(self, verts, uvs, faces):
        o = len(self.v)
        self.v += [list(map(float, p)) for p in verts]
        self.t += [list(map(float, q)) for q in uvs]
        self.f += [[a + o, b + o, c + o] for a, b, c in faces]

    def write_obj(self, path):
        with open(path, "w") as fh:
            for p in self.v:
                fh.write("v %.5f %.5f %.5f\n" % tuple(p))
            for q in self.t:
                fh.write("vt %.4f %.4f\n" % tuple(q))
            for a, b, c in self.f:
                fh.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")


def leaf_surface(length, width, fold, droop, nu=10, nv=4, closed=False, curl=0.0):
    """Leaf blade in local frame: base at origin, along +x, upper side ~ +z (one winding; two-sided material).

    fold: V-fold along the midrib (0 flat .. ~1.4 nearly closed); droop: tip drop as a fraction of length.
    """
    verts, uvs = [], []
    for i in range(nu + 1):
        s = i / nu
        if closed:  # spindle bud: widest at ~40 %, sharp tip
            prof = math.sin(math.pi * min(1.0, s * 0.95)) ** 1.1 * (1 - 0.45 * s)
        else:       # narrow elliptic leaf with acute tip and cuneate base
            prof = math.sin(math.pi * s) ** 0.8 * (1 - 0.12 * s)
        half = 0.5 * width * prof
        for j in range(nv + 1):
            c = -1 + 2 * j / nv
            y = c * half * math.cos(min(fold, 1.45) * abs(c) * .5)
            z = math.sin(min(fold, 1.45) * .5) * abs(c) * half * 1.6 - droop * length * s * s
            z += curl * (abs(c) ** 2) * half * (0.6 + 0.4 * s)  # margins roll upward (young leaves)
            verts.append([s * length, y, z])
            uvs.append([s, 0.5 + 0.5 * c])
    faces = []
    W = nv + 1
    for i in range(nu):
        for j in range(nv):
            a, b, c2, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            faces += [[a, c2, b], [b, c2, d]]
    return np.array(verts), np.array(uvs), faces


def sweep(points, radii, n=7, vscale=8.0):
    """Smooth tube along a polyline with parallel-transported frames (no twist, no seams)."""
    P = [np.asarray(p, float) for p in points]
    T = [P[1] - P[0]] + [P[k + 1] - P[k - 1] for k in range(1, len(P) - 1)] + [P[-1] - P[-2]]
    T = [t / np.linalg.norm(t) for t in T]
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < .9 else np.array([1.0, 0, 0])
    N = np.cross(T[0], ref); N /= np.linalg.norm(N)
    verts, uvs, faces, dist = [], [], [], 0.0
    for k, (p, t, r) in enumerate(zip(P, T, radii)):
        if k:
            N = N - np.dot(N, t) * t; N /= np.linalg.norm(N)
            dist += np.linalg.norm(P[k] - P[k - 1])
        B = np.cross(t, N)
        for j in range(n + 1):
            a = 2 * math.pi * j / n
            verts.append(p + r * (math.cos(a) * N + math.sin(a) * B)); uvs.append([j / n, dist * vscale])
    for k in range(len(P) - 1):
        for j in range(n):
            a, b = k * (n + 1) + j, k * (n + 1) + j + 1
            c, d = a + n + 1, b + n + 1
            faces += [[a, b, c], [b, d, c]]
    return verts, uvs, faces


def frame_from(direction, up_hint=(0, 0, 1)):
    x = np.asarray(direction, float); x /= np.linalg.norm(x)
    up = np.asarray(up_hint, float)
    if abs(np.dot(up, x)) > .97:
        up = np.array([1.0, 0, 0])
    y = np.cross(up, x); y /= np.linalg.norm(y); z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def rot_axis(axis, ang):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def place(verts, R, t):
    return np.asarray(verts) @ R.T + t


# ---------------------------------------------------------------- bush
class Bush:
    def __init__(self, rng):
        self.rng = rng
        self.wood, self.twig, self.bud = Mesh(), Mesh(), Mesh()
        self.leaves = {"a": Mesh(), "b": Mesh(), "c": Mesh()}
        self.tips = []      # harvest-shoot poses: (position, axis direction)
        self.proxies = []   # collision capsules near the canopy top (for MuJoCo)

    def leaf(self, node, stem_dir, phi, scale=1.0, young_frac=0.0):
        rng = self.rng
        L = rng.uniform(.075, .115) * scale; W = L * rng.uniform(.27, .33)
        lv, luv, lf = leaf_surface(L, W, fold=rng.uniform(.2, .5), droop=rng.uniform(.08, .25))
        # direction: out from the stem at azimuth phi, rising 25-55 deg above the stem-normal plane
        F = frame_from(stem_dir)
        out = math.cos(phi) * F[:, 1] + math.sin(phi) * F[:, 2]
        elev = math.radians(rng.uniform(25, 55))
        d = math.cos(elev) * out + math.sin(elev) * np.asarray(stem_dir) / np.linalg.norm(stem_dir)
        R = frame_from(d, up_hint=stem_dir) @ rot_axis([1, 0, 0], rng.uniform(-.5, .5))
        petiole = node + d * .006
        kind = "a" if rng.random() < .72 else ("b" if rng.random() < .55 else "c")
        self.leaves[kind].add(place(lv, R, petiole), luv, lf)

    def small_bud(self, tip, d):
        bv, buv, bf = leaf_surface(.018, .005, fold=1.4, droop=0, nu=6, nv=2, closed=True)
        self.bud.add(place(bv, frame_from(d), tip), buv, bf)

    def branch(self, start, direction, length, r0, depth, leafy_from=0.0, phyllo=0.0, shoot_tip=False):
        rng = self.rng
        nseg = max(3, int(length / .06))
        pts, d = [np.asarray(start, float)], np.asarray(direction, float) / np.linalg.norm(direction)
        for k in range(nseg):
            d = d + rng.normal(0, .06, 3) + np.array([0, 0, .05 if depth == 0 else .02]); d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * length / nseg)
        rad = [r0 * (1 - 0.65 * k / nseg) for k in range(nseg + 1)]
        # split: old wood below, grey-green twig in the last 20 cm / thin parts
        tl = np.cumsum([0] + [np.linalg.norm(pts[k + 1] - pts[k]) for k in range(nseg)])
        cut = next((k for k in range(nseg + 1) if tl[-1] - tl[k] < .20 or rad[k] < .0035), nseg)
        if cut > 0:
            self.wood.add(*sweep(pts[:cut + 1], rad[:cut + 1], n=7 if depth == 0 else 5))
        if cut < nseg:
            self.twig.add(*sweep(pts[cut:], rad[cut:], n=5))
        if pts[-1][2] > 1.0:
            for a, b in zip(pts[:-1], pts[1:]):
                if b[2] > 1.0:
                    self.proxies.append(dict(type="capsule", fromto=[*a, *b], size=float(max(rad[0] * .7, .003))))
        # leaves: alternate, ~2/5 phyllotaxis, every 3-4.5 cm along the leafy part
        s, phi = leafy_from * tl[-1] + .02, phyllo
        stop = tl[-1] - (.06 if shoot_tip else .015)
        while s < stop:
            k = min(int(np.searchsorted(tl, s)), nseg)
            a = pts[max(k - 1, 0)]; b = pts[k]
            node = a + (b - a) * ((s - tl[max(k - 1, 0)]) / max(tl[k] - tl[max(k - 1, 0)], 1e-6))
            self.leaf(node, b - a if np.linalg.norm(b - a) > 0 else d, phi, scale=1.0 - .25 * (s / tl[-1]) ** 2)
            phi += 2.513 + rng.normal(0, .25)
            s += rng.uniform(.022, .032)
        # side branches
        if depth < 3:
            s = max(leafy_from, .35) * tl[-1]
            while s < tl[-1] - .12:
                if rng.random() < (.7, .5, .3)[depth]:
                    k = min(int(np.searchsorted(tl, s)), nseg)
                    base = pts[k]
                    ax = pts[min(k + 1, nseg)] - pts[max(k - 1, 0)]
                    F = frame_from(ax)
                    az = rng.uniform(0, 2 * math.pi)
                    out = math.cos(az) * F[:, 1] + math.sin(az) * F[:, 2]
                    ang = math.radians(rng.uniform(28, 48))
                    bd = math.cos(ang) * ax / np.linalg.norm(ax) + math.sin(ang) * out
                    blen = (tl[-1] - s) * rng.uniform(.35, .65) * (1.0 if depth == 0 else .7)
                    self.branch(base, bd, max(blen, .08), rad[k] * .6, depth + 1, leafy_from=0.1, phyllo=az,
                                shoot_tip=(depth == 0 and blen > .25 and pts[-1][2] > 1.2))
                s += rng.uniform(.06, .12)
        tip_dir = pts[-1] - pts[-2]
        if shoot_tip:
            self.tips.append((pts[-1], tip_dir / np.linalg.norm(tip_dir)))
        else:
            self.small_bud(pts[-1], tip_dir)


def make_bush(rng, height):
    b = Bush(rng)
    n = int(rng.integers(8, 13))
    for k in range(n):
        az = 2 * math.pi * (k + rng.uniform(-.35, .35)) / n
        r = rng.uniform(.01, .07)
        base = np.array([r * math.cos(az), r * math.sin(az), 0])
        lean = math.radians(rng.uniform(3, 14))
        d = np.array([math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)])
        b.branch(base, d, height * rng.uniform(.72, 1.0), rng.uniform(.008, .013), 0,
                 leafy_from=rng.uniform(.30, .45), phyllo=rng.uniform(0, 6), shoot_tip=True)
    return b


def make_shoot(rng):
    """Harvest target in local frame (node at origin, stem continuing along +z):
    grey-green stem, two half-folded young leaves from the two top nodes, closed spindle terminal bud."""
    stem, bud, young = Mesh(), Mesh(), Mesh()
    stem.add(*sweep([[0, 0, 0], [0, 0, .02], [.001, 0, .04], [.0015, 0, .05]], [.0024, .0021, .0018, .0016], n=12))
    bv, buv, bf = leaf_surface(.034, .009, fold=1.45, droop=-.02, nu=24, nv=8, closed=True)
    bud.add(place(bv, frame_from([.03, 0, 1.0]), [.0015, 0, .048]), buv, bf)
    for k, (z, az) in enumerate(((.018, 0.3), (.034, 0.3 + math.pi))):
        L = rng.uniform(.048, .058) - .008 * k; W = L * .28
        lv, luv, lf = leaf_surface(L, W, fold=.75, droop=.06, nu=28, nv=10, curl=.35)
        ang = math.radians(rng.uniform(32, 42))
        d = np.array([math.sin(ang) * math.cos(az), math.sin(ang) * math.sin(az), math.cos(ang)])
        young.add(place(lv, frame_from(d, up_hint=[0, 0, 1]), [0, 0, z]), luv, lf)
    return stem, bud, young


def generate(out_dir, n_variants=4, seed=42):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for kind in ("a", "b", "c", "young", "bud"):
        leaf_texture(rng, kind).save(out / f"tex_leaf_{kind}.png")
    wood_texture(rng, "wood").save(out / "tex_wood.png"); wood_texture(rng, "twig").save(out / "tex_twig.png")
    shoot = make_shoot(rng)
    for name, m in zip(("shoot_stem", "shoot_bud", "shoot_young"), shoot):
        m.write_obj(out / f"{name}.obj")
    meta = {"seed": seed, "bushes": []}
    for v in range(n_variants):
        height = float(rng.uniform(1.45, 2.05))
        b = make_bush(rng, height)
        b.wood.write_obj(out / f"bush{v}_wood.obj"); b.twig.write_obj(out / f"bush{v}_twig.obj")
        # bake a shoot onto every shoot tip (visual); simulated harvest shoots are placed separately
        baked = [Mesh(), Mesh(), Mesh()]
        for p, d in b.tips:
            R = frame_from(d) @ np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # local +z -> tip direction
            R = R @ rot_axis([0, 0, 1], rng.uniform(0, 2 * math.pi))
            for src, dst in zip(shoot, baked):
                dst.add(place(src.v, R, p), src.t, [[a, b2, c] for a, b2, c in src.f])
        for name, m in zip(("stem", "bud", "young"), baked):
            m.write_obj(out / f"bush{v}_shoot_{name}.obj")
        b.bud.write_obj(out / f"bush{v}_bud.obj")
        for k, m in b.leaves.items():
            m.write_obj(out / f"bush{v}_leaf_{k}.obj")
        meta["bushes"].append(dict(height=height, tips=[dict(pos=list(map(float, p)), dir=list(map(float, d))) for p, d in b.tips],
                                   proxies=b.proxies, n_leaves={k: len(m.f) // 80 for k, m in b.leaves.items()},
                                   n_tris=sum(len(m.f) for m in (b.wood, b.twig, b.bud, *b.leaves.values()))))
    (out / "bushes.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    import sys
    m = generate(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "output" / "adeept" / "plants")
    for i, b in enumerate(m["bushes"]):
        print(i, round(b["height"], 2), "tips", len(b["tips"]), "leaves", b["n_leaves"], "tris", b["n_tris"])
