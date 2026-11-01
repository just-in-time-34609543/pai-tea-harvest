"""Fuse the Unreal RGB-D cameras into one coloured point cloud in the platform frame.

Depth comes from SceneCapture SCS_SCENE_DEPTH (cm along the view axis, red channel of an RGBA16F EXR); intrinsics
from the horizontal field of view sent to Unreal; extrinsics from the MuJoCo camera poses (platform frame).
The cloud is cropped to the workspace above the canopy, voxel-downsampled and sampled to a fixed size.
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

WORKSPACE = np.array([[-0.12, 0.34], [-0.30, 0.30], [-0.06, 0.30]])   # x, y, z bounds (m), platform frame
N_POINTS = 2048


def read_depth_m(path):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return d[..., 2].astype(np.float32) / 100.0      # BGRA -> R channel, cm -> m


def backproject(depth, rgb, cam_pos, cam_mat, fovx_deg, max_depth=2.5):
    """depth (H,W) metres along the view axis; cam_mat = MuJoCo cam_xmat (columns: right, up, back)."""
    H, W = depth.shape
    fx = (W / 2) / np.tan(np.radians(fovx_deg) / 2)
    u, v = np.meshgrid(np.arange(W) + .5 - W / 2, np.arange(H) + .5 - H / 2)
    ok = (depth > 0.02) & (depth < max_depth)
    z = depth[ok]
    xr, yu = u[ok] / fx * z, -v[ok] / fx * z
    R = cam_mat.reshape(3, 3)
    pts = cam_pos + np.outer(xr, R[:, 0]) + np.outer(yu, R[:, 1]) + np.outer(z, -R[:, 2])
    return pts, rgb[ok].astype(np.float32) / 255.0


def fuse(views, rng=None, n=N_POINTS, voxel=0.004):
    """views: list of (depth_m, rgb, cam_pos, cam_mat, fovx). Returns (n, 6) xyz+rgb float32."""
    P, C = [], []
    for v in views:
        p, c = backproject(*v)
        P.append(p); C.append(c)
    P, C = np.concatenate(P), np.concatenate(C)
    m = np.all((P >= WORKSPACE[:, 0]) & (P <= WORKSPACE[:, 1]), axis=1)
    P, C = P[m], C[m]
    if len(P) == 0:
        return np.zeros((n, 6), np.float32)
    k = np.floor((P - WORKSPACE[:, 0]) / voxel).astype(np.int64)
    key = (k[:, 0] * 1024 + k[:, 1]) * 1024 + k[:, 2]          # one int per voxel (workspace < 4 m per axis)
    _, idx = np.unique(key, return_index=True)
    P, C = P[idx], C[idx]
    rng = rng or np.random.default_rng(0)
    sel = rng.choice(len(P), n, replace=len(P) < n)
    return np.concatenate([P[sel], C[sel]], 1).astype(np.float32)
