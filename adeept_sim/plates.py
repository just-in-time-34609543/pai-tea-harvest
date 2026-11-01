"""3 mm acrylic plate meshes built from the outlines in parts.json (see extract_parts.py)."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

PARTS = json.loads(Path(__file__).with_name("parts.json").read_text())
THICKNESS = 3.0   # mm
GRID = 0.25       # mm per raster cell for the flat faces


def hole(name, index):
    return np.array(PARTS[name]["holes"][index]["centre"])


def round_holes(name):
    return [np.array(h["centre"]) for h in PARTS[name]["holes"] if h["round"]]


def mesh(name, origin, axis_to, flip=False):
    """MJCF vertex/face strings in metres for plate `name`.

    Plate coordinates (mm, image axes) are moved so that `origin` is at (0, 0) and the
    direction origin -> axis_to becomes +x; the plate spans z in [-1.5, 1.5] mm.
    flip mirrors the plate across its x axis.
    """
    part = PARTS[name]
    origin, axis_to = np.asarray(origin, float), np.asarray(axis_to, float)
    angle = math.atan2(*(axis_to - origin)[::-1])
    c, s = math.cos(angle), math.sin(angle)

    def to_plate(points):
        d = np.asarray(points, float) - origin
        u = d[:, 0] * c + d[:, 1] * s
        v = -d[:, 0] * s + d[:, 1] * c
        return np.stack([u, -v if not flip else v], axis=1)   # image y points down; plate y up

    # flat faces: rasterise the plate and emit one quad per horizontal run
    size = np.ceil(np.array(part["size"]) / GRID).astype(int) + 4
    raster = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(raster, [np.round(np.array(part["outline"]) / GRID).astype(np.int32)], 1)
    for h in part["holes"]:
        cv2.fillPoly(raster, [np.round(np.array(h["outline"]) / GRID).astype(np.int32)], 0)
    verts, faces = [], []

    def quad(a, b, cc, d):
        n = len(verts)
        verts.extend([a, b, cc, d])
        faces.extend([(n, n + 1, n + 2), (n, n + 2, n + 3)])

    half = THICKNESS / 2
    for row in range(raster.shape[0]):
        line = np.concatenate([[0], raster[row], [0]])
        starts, ends = np.flatnonzero(np.diff(line) == 1), np.flatnonzero(np.diff(line) == -1)
        for x0, x1 in zip(starts, ends):
            corners = to_plate([[x0 * GRID, row * GRID], [x1 * GRID, row * GRID],
                                [x1 * GRID, (row + 1) * GRID], [x0 * GRID, (row + 1) * GRID]])
            top = [(*p, half) for p in corners]
            bottom = [(*p, -half) for p in corners]
            quad(*top)
            quad(*bottom[::-1])
    # side walls along the outline and every hole
    for loop in [part["outline"]] + [h["outline"] for h in part["holes"]]:
        pts = to_plate(loop)
        for p, q in zip(pts, np.roll(pts, -1, axis=0)):
            quad((*p, -half), (*q, -half), (*q, half), (*p, half))
    v = " ".join(f"{x / 1000:.5f}" for p in verts for x in p)
    f = " ".join(str(i) for t in faces for i in t)
    return v, f


def tip(name, origin):
    """Plate point furthest from origin (a finger tip), in image mm."""
    pts = np.array(PARTS[name]["outline"])
    return pts[np.argmax(np.linalg.norm(pts - origin, axis=1))]
