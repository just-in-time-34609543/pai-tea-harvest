"""Tea-shoot harvest task for the Adeept ADA031 arm (MuJoCo physics; images from MuJoCo or Unreal).

The arm stands on a small platform hung from the ceiling-rail carriage (rail-suspended, not floor
standing). Around the platform, young tea shoots stick up from the canopy below. A shoot is a free body held to
the bush by a weld; the weld releases only once the gripper has closed on that shoot and lifted it (simulated
plucking: no cutting, no plant damage modelled). The harvested shoot goes into the basket on the platform.

The arm model, servo mapping, IK and expert motion come from the Adeept ADA031 simulator in
adeept_sim/ (included in this repository). Set ADEEPT_SIM to use another copy.

Frame: platform frame, arm base at the origin, z up, metres (same as the Adeept simulator).
"""
from dataclasses import dataclass
import math
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, os.environ.get("ADEEPT_SIM", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adeept_sim")))
import env as adeept  # noqa: E402

N_SHOOTS = 3
ARCH_CAMS = ("camera1", "camera3", "camera4", "camera5")
ARCH_CAM_FOVY = 22
PLANTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "adeept", "plants")
CANOPY_VARIANT, CANOPY_TOP = 2, 0.02     # bush variant under the platform; its top sits this high in the platform frame
WORKSPACE_CENTRE = np.array([0.10, 0.0, 0.07])     # platform frame


def _world():
    import json
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "adeept_scene.json")))
    return cfg["world"]


def arch_xs():
    g = _world()["greenhouse"]
    n = int(round(g["length"] / g["arch_spacing"])) + 1
    return np.linspace(g["x_min"], g["x_min"] + g["length"], n)


def arch_camera_mounts(x_platform, y_cam=0.9, drop=0.04):
    """World positions of the 4 arch cameras around a platform at x_platform: nearest arch on each side, y = +-y_cam."""
    g = _world()["greenhouse"]
    xs = arch_xs()
    left = xs[xs <= x_platform].max(); right = xs[xs > x_platform].min()
    half, eave, top = g["width"] / 2, g["eave"], g["height"]
    z = eave + (top - eave) * math.sqrt(1 - (y_cam / half) ** 2) - drop
    return [np.array([x, y, z]) for x in (left, right) for y in (-y_cam, y_cam)]


def lookat_quat(pos, target):
    """MuJoCo camera quaternion looking from pos to target with world z up (camera -z forward, +y up)."""
    f = np.asarray(target, float) - pos; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([r, u, -f], 1).flatten())
    return q
GRASP_Z = 0.075            # grasp point (centre of the bud cluster) above the base plate, like the cork
SHOOT_STEM = 0.20          # visible stem below the grasp point, down into the canopy


@dataclass
class HarvestTask:
    language: str
    target: str             # "left" | "middle" | "right" | "any"


TASKS = {
    "harvest": HarvestTask("harvest the young tea shoot and put it in the basket", "any"),
    "harvest_left": HarvestTask("harvest the young tea shoot on the left and put it in the basket", "left"),
    "harvest_middle": HarvestTask("harvest the young tea shoot in the middle and put it in the basket", "middle"),
    "harvest_right": HarvestTask("harvest the young tea shoot on the right and put it in the basket", "right"),
}


def build_xml():
    root = ET.fromstring(adeept.build_xml())
    root.set("model", "adeept_tea_harvest")
    wb = root.find("worldbody")
    for el in list(wb):
        if (el.tag == "body" and el.get("name", "").startswith("shuttle")) or (el.tag == "camera" and el.get("name") == "camera1") \
                or (el.tag == "geom" and el.get("name") == "table"):
            wb.remove(el)
    # hanging platform under the arm base: a disc with a shelf for the basket, hung by four rods (visual)
    ET.SubElement(wb, "geom", name="platform", type="cylinder", size="0.075 0.004", pos="0 0 -0.004", rgba=".55 .55 .52 1",
                  contype="1", conaffinity="1")
    ET.SubElement(wb, "geom", name="shelf", type="box", size="0.085 0.085 0.004", pos="0.051 0.141 -0.004", rgba=".55 .55 .52 1",
                  contype="1", conaffinity="1")
    for k, (x, y) in enumerate(((.05, .05), (-.05, .05), (.05, -.05), (-.05, -.05))):
        ET.SubElement(wb, "geom", name=f"hanger{k}", type="capsule", fromto=f"{x} {y} 0 {x} {y} 0.45", size="0.003",
                      rgba=".45 .47 .45 1", contype="0", conaffinity="0", group="1")
    # fixed scene camera on a hanger, looking down at the workspace (like the Adeept camera1, here looking from above-behind)
    # four RGB-D cameras fixed on the greenhouse arch pipes (two arches either side of the platform, at y = +-0.9 m on
    # the roof); their poses relative to the platform are set per episode (set_origin). camera2 stays on the wrist.
    for k, name in enumerate(ARCH_CAMS):
        ET.SubElement(wb, "camera", name=name, pos=f"{k} 0 1", fovy=str(ARCH_CAM_FOVY))
        b = ET.SubElement(wb, "body", name=f"cambox{k}", pos=f"{k} 0 1", mocap="true")
        ET.SubElement(b, "geom", type="box", size="0.03 0.022 0.022", rgba=".12 .12 .13 1", contype="0", conaffinity="0", group="1")
        ET.SubElement(b, "geom", type="cylinder", size="0.011 0.006", pos="0.03 0 0", euler="0 1.5708 0", rgba=".05 .05 .06 1", contype="0", conaffinity="0", group="1")
    asset = root.find("asset")
    ET.SubElement(asset, "material", name="shoot_bud", rgba=".72 .60 .44 1")
    ET.SubElement(asset, "material", name="shoot_stem", rgba=".45 .47 .33 1")
    ET.SubElement(asset, "material", name="shoot_young", rgba=".66 .58 .40 1", specular=".2")
    ET.SubElement(asset, "material", name="leaf_a", rgba=".16 .30 .09 1", specular=".5", shininess=".6")
    ET.SubElement(asset, "material", name="leaf_b", rgba=".30 .36 .10 1", specular=".4", shininess=".5")
    ET.SubElement(asset, "material", name="leaf_c", rgba=".20 .28 .10 1", specular=".4", shininess=".5")
    ET.SubElement(asset, "material", name="wood", rgba=".36 .22 .14 1")
    ET.SubElement(asset, "material", name="twig", rgba=".42 .44 .32 1")
    vis = dict(contype="0", conaffinity="0", group="2", mass="0")
    if os.path.exists(os.path.join(PLANTS, "shoot_young.obj")):
        for part in ("young", "bud", "stem"):
            ET.SubElement(asset, "mesh", name=f"m_shoot_{part}", file=os.path.abspath(os.path.join(PLANTS, f"shoot_{part}.obj")), inertia="shell")
        # visual canopy of one procedural tea bush under the platform (its shoots removed near the platform)
        import json as _json
        meta = _json.load(open(os.path.join(PLANTS, "bushes.json")))["bushes"][CANOPY_VARIANT]
        dz = CANOPY_TOP - meta["height"]
        canopy = ET.SubElement(wb, "body", name="canopy", pos=f"0.11 0 {dz:.4f}")
        for part, mat in (("wood", "wood"), ("twig", "twig"), ("leaf_a", "leaf_a"), ("leaf_b", "leaf_b"), ("leaf_c", "leaf_c"), ("bud", "shoot_bud")):
            ET.SubElement(asset, "mesh", name=f"m_canopy_{part}", file=os.path.abspath(os.path.join(PLANTS, f"bush{CANOPY_VARIANT}_{part}.obj")), inertia="shell")
            ET.SubElement(canopy, "geom", type="mesh", mesh=f"m_canopy_{part}", material=mat, **vis)
    eq = root.find("equality") or ET.SubElement(root, "equality")
    for i in range(N_SHOOTS):
        b = ET.SubElement(wb, "body", name=f"shoot{i}", pos=f"{1 + i} 3 0")
        ET.SubElement(b, "freejoint", name=f"shoot{i}_free")
        # bud cluster (grasp target) centred on the body origin; stem below
        ET.SubElement(b, "geom", name=f"bud{i}", type="capsule", fromto="0 0 -0.012 0 0 0.018", size="0.0065", rgba="0 0 0 0",
                      mass="0.0015", condim="4", friction="1.5 0.02 0.001", contype="1", conaffinity="1")
        ET.SubElement(b, "geom", name=f"stem{i}", type="capsule", fromto=f"0 0 -{SHOOT_STEM} 0 0 -0.012", size="0.0022",
                      material="shoot_stem", mass="0.001", contype="0", conaffinity="0")
        if os.path.exists(os.path.join(PLANTS, "shoot_young.obj")):
            # the procedural shoot (two half-folded young leaves + closed bud) around the grasp capsule
            for part, mat in (("young", "shoot_young"), ("bud", "shoot_bud"), ("stem", "shoot_stem")):
                ET.SubElement(b, "geom", type="mesh", mesh=f"m_shoot_{part}", material=mat, pos="0 0 -0.03",
                              contype="0", conaffinity="0", group="2", mass="0")
        ET.SubElement(wb, "body", name=f"fixture{i}", pos=f"{1 + i} 3 0", mocap="true")
        ET.SubElement(eq, "weld", name=f"attach{i}", body1=f"shoot{i}", body2=f"fixture{i}", solref="0.004 1")
    return ET.tostring(root, encoding="unicode")


class HarvestEnv(adeept.AdeeptSimEnv):
    def __init__(self, task="harvest", fps=10, max_step=3.0, max_ticks=220, size=(480, 640), renderer="mujoco"):
        self.task_name, self.task = task, TASKS[task]
        self.fps, self.max_step, self.one_axis, self.max_ticks = fps, max_step, False, max_ticks
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.size = size
        self.renderer = mujoco.Renderer(self.model, *size) if renderer == "mujoco" else None
        self.ue = None              # set to a UE render client to take images from Unreal
        self.substeps = round(1 / fps / self.model.opt.timestep)
        self.rng = np.random.default_rng()
        self.tcp, self.axis_tip = self.model.site("tcp").id, self.model.site("axis_tip").id
        self.shoot_qpos = [self.model.jnt_qposadr[self.model.body(f"shoot{i}").jntadr[0]] for i in range(N_SHOOTS)]
        self.fixture_mocap = [self.model.body_mocapid[self.model.body(f"fixture{i}").id] for i in range(N_SHOOTS)]
        self.attach = [self.model.equality(f"attach{i}").id for i in range(N_SHOOTS)]
        self.bud_geom = [self.model.geom(f"bud{i}").id for i in range(N_SHOOTS)]
        self.finger_bodies = {self.model.body("finger_l").id, self.model.body("finger_r").id}
        self.box_xy = np.array([0.051, 0.141])    # basket on the shelf (heading +70 deg, r 0.15)
        self.origin = np.array([0.45, 0.0, 1.85])  # platform frame in the greenhouse (set_origin)
        self.target = 0

    @property
    def language(self):
        return self.task.language

    # -- scene --
    def _sample_shoots(self):
        spots = []
        for _ in range(500):
            spots = []
            for _ in range(N_SHOOTS):
                for _ in range(200):
                    heading = math.radians(self.rng.uniform(-60, 40))
                    r = self.rng.uniform(0.09, 0.145)
                    xy = np.array([r * math.cos(heading), r * math.sin(heading)])
                    z = GRASP_Z + self.rng.uniform(-0.012, 0.012)
                    if all(np.linalg.norm(xy - s[:2]) > 0.055 for s in spots) and \
                            self.grasp_pose((*xy, z + 0.05), adeept.OPEN) is not None and self.grasp_pose((*xy, z), adeept.OPEN) is not None:
                        spots.append(np.array([*xy, z])); break
            if len(spots) == N_SHOOTS:
                return spots
        raise RuntimeError("no reachable shoot layout")

    def set_origin(self, origin):
        """Place the platform in the greenhouse; moves the arch cameras (fixed in the world) into the platform frame."""
        self.origin = np.asarray(origin, float)
        for k, (name, mount) in enumerate(zip(ARCH_CAMS, arch_camera_mounts(self.origin[0]))):
            local = mount - self.origin
            c = self.model.camera(name).id
            self.model.cam_pos[c] = local
            self.model.cam_quat[c] = lookat_quat(local, WORKSPACE_CENTRE)
            b = self.model.body_mocapid[self.model.body(f"cambox{k}").id]
            self.data.mocap_pos[b] = local
            # box +x along the view direction
            f = WORKSPACE_CENTRE - local; f /= np.linalg.norm(f)
            r = np.cross([0, 0, 1.0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
            q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([f, r, u], 1).flatten()); self.data.mocap_quat[b] = q

    def reset(self, seed=None, task=None):
        if task is not None:
            self.task_name, self.task = task, TASKS[task]
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        spots = self._sample_shoots()
        # left / middle / right as seen from camera1 (behind the arm): order by heading (atan2), largest = left
        order = np.argsort([-math.atan2(s[1], s[0]) for s in spots])   # order[0] = left
        self.layout = {"left": int(order[0]), "middle": int(order[1]), "right": int(order[2])}
        self.target = int(self.rng.integers(N_SHOOTS)) if self.task.target == "any" else self.layout[self.task.target]
        for i, s in enumerate(spots):
            quat = [1, 0, 0, 0]
            adr = self.shoot_qpos[i]
            self.data.qpos[adr:adr + 7] = [*s, *quat]
            self.data.mocap_pos[self.fixture_mocap[i]] = s
            self.data.eq_active[self.attach[i]] = 1
        self.data.mocap_pos[self.model.body_mocapid[self.model.body("box").id]] = [*self.box_xy, 0]
        self.set_origin(self.origin)
        self.spots = spots
        self.picked = [False] * N_SHOOTS
        self.contact_z = [None] * N_SHOOTS
        self.state = adeept.HOME.copy()
        self.data.qpos[:6] = adeept.servo_to_joints(self.state)
        self.data.ctrl[:] = adeept.servo_to_joints(self.state)
        self.ticks = 0
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        return self._observation()

    def _gripping(self, i):
        for c in self.data.contact[: self.data.ncon]:
            g = (c.geom1, c.geom2)
            if self.bud_geom[i] in g:
                other = g[1] if g[0] == self.bud_geom[i] else g[0]
                if self.model.geom_bodyid[other] in self.finger_bodies:
                    return True
        return False

    def step(self, action):
        target = np.clip(np.asarray(action, dtype=float), adeept.LOW, adeept.HIGH)
        self.state = self.state + np.clip(target - self.state, -self.max_step, self.max_step)
        self.data.ctrl[:] = adeept.servo_to_joints(self.state)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        # plucking: a shoot's weld releases when the fingers hold its bud with the gripper nearly closed and the
        # arm pulls upward (commanded TCP more than 5 mm above the held position)
        tcp_z = self.data.site_xpos[self.tcp][2]
        pull = self.fk(self.state)[0][2] - tcp_z
        for i in range(N_SHOOTS):
            if self.data.eq_active[self.attach[i]] and self.state[4] > 135 and self._gripping(i) and pull > 0.005:
                self.data.eq_active[self.attach[i]] = 0
                self.picked[i] = True
        self.ticks += 1
        success = self.success()
        return self._observation(), success, success or self.ticks >= self.max_ticks

    def shoot_pos(self, i):
        adr = self.shoot_qpos[i]
        return self.data.qpos[adr:adr + 3].copy()

    def in_basket(self, i):
        p = self.shoot_pos(i)
        return bool(np.all(np.abs(p[:2] - self.box_xy) < 0.06) and p[2] < 0.05 and not self._gripping(i))

    def success(self):
        if self.task.target == "any":    # any single shoot harvested into the basket counts
            picked = [j for j in range(N_SHOOTS) if self.picked[j]]
            return bool(len(picked) == 1 and self.in_basket(picked[0]) and self.state[4] < 80)
        others_attached = all(not self.picked[j] for j in range(N_SHOOTS) if j != self.target)
        return bool(self.picked[self.target] and self.in_basket(self.target) and others_attached and self.state[4] < 80)

    # -- images --
    def _observation(self):
        obs = {"observation.state": self.state.astype(np.float32)}
        if self.renderer is not None or self.ue is not None:
            obs["observation.images.camera1"] = self.render("camera1")
            obs["observation.images.camera2"] = self.render("camera2")
        return obs

    def cloud(self, rng=None):
        """Fused point cloud from MuJoCo-rendered RGB-D of all cameras (fallback when Unreal is unavailable)."""
        from pointcloud import fuse
        h, w = self.size
        if not hasattr(self, "depth_renderer"):
            self.depth_renderer = mujoco.Renderer(self.model, h, w); self.depth_renderer.enable_depth_rendering()
        views = []
        for c in range(self.model.ncam):
            name = self.model.camera(c).name
            self.renderer.update_scene(self.data, name); rgb = self.renderer.render().copy()
            self.depth_renderer.update_scene(self.data, name); depth = self.depth_renderer.render().copy()
            fovx = float(np.degrees(2 * np.arctan(np.tan(np.radians(self.model.cam_fovy[c]) / 2) * w / h)))
            views.append((depth, rgb, self.data.cam_xpos[c].copy(), self.data.cam_xmat[c].copy(), fovx))
        return fuse(views, rng=rng)

    def render(self, camera):
        if self.ue is not None:
            return self.ue.render(self, camera)
        self.renderer.update_scene(self.data, camera)
        return self.renderer.render().copy()

    # -- scripted expert --
    def expert_waypoints(self):
        bud = self.spots[self.target].copy()
        above = bud + [0, 0, 0.05]
        pre = self.grasp_pose(above, adeept.OPEN)
        grasp = None if pre is None else self.grasp_pose(bud, adeept.OPEN, near=pre)
        if grasp is None:
            return [(None, 0)]
        closed = grasp.copy(); closed[4] = adeept.CLOSED
        lift = pre.copy(); lift[4] = adeept.CLOSED
        over = self.grasp_pose((*self.box_xy, 0.09), adeept.CLOSED, near=lift)
        if over is None:
            return [(None, 0)]
        release = over.copy(); release[4] = adeept.OPEN
        return [(pre, 0), (grasp, 2), (closed, 4), (lift, 3), (over, 2), (release, 5)]


class Expert(adeept.Expert):
    """Adeept expert motion (straight lines in servo space between waypoints), without the shuttle drop re-aim."""

    def act(self):
        state = self.env.state
        if self.index >= len(self.plan):
            return state.copy()
        goal, hold = self.plan[self.index]
        delta = goal - state
        if np.max(np.abs(delta)) < 0.5:
            self.hold = hold if self.hold is None else self.hold - 1
            if self.hold <= 0:
                self.index, self.hold = self.index + 1, None
            return goal.copy()
        step = delta / max(1.0, np.max(np.abs(delta)) / self.env.max_step)
        if self.noise:
            step = step + self.env.rng.normal(0, self.noise, 5) * (np.abs(delta) > 1)
        return state + step


if __name__ == "__main__":
    import time
    env = HarvestEnv(renderer="none")
    ok = 0
    for seed in range(20):
        task = ["harvest_left", "harvest_middle", "harvest_right", "harvest"][seed % 4]
        env.reset(seed=seed, task=task)
        ex = Expert(env)
        t0 = time.time()
        for _ in range(env.max_ticks):
            _, success, done = env.step(ex.act())
            if done:
                break
        ok += success
        print(seed, task, "target", env.target, "picked", env.picked, "success", success, "ticks", env.ticks, f"{time.time()-t0:.1f}s")
    print("expert success", ok, "/ 20")
