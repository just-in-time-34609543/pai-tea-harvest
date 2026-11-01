"""Client for scripts/ue/render_server.py: sends MuJoCo body / camera poses, returns Unreal-rendered RGB images.

Usage: env.ue = UERenderClient(origin=(x, y, z)) — origin is where the platform frame sits in the greenhouse world.
Both cameras are rendered in one request per simulation tick and cached.
"""
import json
import os
import socket
import tempfile

import imageio.v2 as imageio
import mujoco
import numpy as np


class UERenderClient:
    def __init__(self, origin, port=5600, host="127.0.0.1", workdir=None, timeout=120, depth=False):
        self.origin = np.asarray(origin, float)
        self.depth = depth
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile("rb")
        self.dir = workdir or tempfile.mkdtemp(prefix="tea_ue_", dir="/dev/shm")
        self.cache_key, self.cache = None, {}
        self.n = 0

    def _request(self, req):
        self.sock.sendall((json.dumps(req) + "\n").encode())
        reply = json.loads(self.f.readline())
        if not reply.get("ok"):
            raise RuntimeError(f"render server: {reply}")
        return reply

    def poses(self, env):
        m, d = env.model, env.data
        bodies = {}
        for b in range(m.nbody):
            name = m.body(b).name if b else "platform"
            bodies[name] = [*(d.xpos[b] + self.origin), *d.xquat[b]] if b else [*self.origin, 1, 0, 0, 0]
        cams = {}
        for c in range(m.ncam):
            R = d.cam_xmat[c].reshape(3, 3)
            fwd, up = -R[:, 2], R[:, 1]
            cams[m.camera(c).name] = [*(d.cam_xpos[c] + self.origin), *fwd, *up, float(m.cam_fovy[c])]
        return bodies, cams

    def render(self, env, camera):
        key = (id(env), env.data.time)
        if key != self.cache_key:
            bodies, cams = self.poses(env)
            # MuJoCo fovy is vertical; SceneCapture fov_angle is horizontal
            h, w = env.size
            for c in cams.values():
                c[9] = float(np.degrees(2 * np.arctan(np.tan(np.radians(c[9]) / 2) * w / h)))
            prefix = os.path.join(self.dir, f"f{self.n:07d}")
            files = self._request({"bodies": bodies, "cams": cams, "out": prefix, "depth": self.depth})["files"]
            self.cache = {}
            for k, v in files.items():
                if k.endswith("_depth"):
                    from pointcloud import read_depth_m
                    self.cache[k] = read_depth_m(v)
                else:
                    self.cache[k] = imageio.imread(v)[..., :3]
            self.fovx = {k: c[9] for k, c in cams.items()}
            for v in files.values():
                os.remove(v)
            self.cache_key = key; self.n += 1
        return self.cache[camera].copy()

    def cloud(self, env, rng=None, n=None):
        """Fused point cloud (platform frame) for the current simulation state; renders if needed."""
        from pointcloud import fuse
        m, d = env.model, env.data
        self.render(env, "camera1")
        views = [(self.cache[m.camera(c).name + "_depth"], self.cache[m.camera(c).name], d.cam_xpos[c].copy(), d.cam_xmat[c].copy(),
                  self.fovx[m.camera(c).name]) for c in range(m.ncam)]
        return fuse(views, rng=rng) if n is None else fuse(views, rng=rng, n=n, voxel=0.002)

    def close(self, quit_server=False):
        if quit_server:
            try:
                self._request({"cmd": "quit"})
            except Exception:
                pass
        self.sock.close()
