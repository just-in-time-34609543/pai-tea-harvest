"""LIBERO-style MuJoCo simulation of the Adeept ADA031 V4 arm for SmolVLA.

Same interface as the real arm: actions and states are the five servo targets
in degrees (base, shoulder, elbow, wrist, gripper),
and the cameras are camera1 (scene view) and camera2 (under the gripper).

Geometry follows "5. Assembly of the Robotic Arm.pdf" of the V4.0 kit:
  - p.114 dimension drawing: table to turntable top 60 mm, turntable top to
    shoulder axis 40 mm, shoulder-elbow 65 mm, elbow-wrist 70 mm,
    wrist to gripper tip 66.5 mm, gripper offset 13 mm from the arm axis.
  - pp.39-66: the elbow servo sits on the forearm (A10) with its horn on the
    upper arm (A07), so the elbow angle is relative to the upper arm.
  - pp.91-104: two meshing gear fingers (A16/A17) pivot on the gripper plate.
  - Plate sizes from "_2 Packing List.pdf" (all parts drawn at one scale,
    0.1251 mm/px on the page; x1.15 to real size, anchored on the servo frame A09).
Joint zeros follow the manual (servos set to 90 before fitting the horns, with
the arm straight up). Rotation directions come from the real arm
(README 2026-10-25/26), the gripper finger angles are estimates.
"""
import math
from dataclasses import dataclass

import mujoco
import numpy as np

import plates

# The real firmware envelope (arm.py LOW/HIGH) was tuned on jt's arm, whose horns sit
# about 50 degrees off the manual (shoulder 40 is upright there, 90 here). With the manual's
# zeros that envelope cannot reach the table, so the simulation uses the servo's own range.
LOW = np.array([0.0, 0.0, 0.0, 0.0, 40.0])
HIGH = np.array([180.0, 180.0, 180.0, 180.0, 150.0])

# p.114 dimensions (m)
TURNTABLE_TOP, SHOULDER_UP = 0.060, 0.040
L2, L3, GRIP_OFFSET = 0.070, 0.0665, 0.013
# p.114 labels shoulder-elbow 65 mm, but the A07/A08 links measure 56.9 mm hole to hole at the
# scale that reproduces the 40 mm shoulder height and the 70 mm forearm, so the links win (要実測).
L1 = float(np.linalg.norm(plates.hole("A08", 0) - plates.hole("A08", 1))) / 1000
_finger_len = float(np.linalg.norm(plates.tip("A16", plates.hole("A16", 0)) - plates.hole("A16", 0))) / 1000
FINGER_PIVOT_Z = L3 - _finger_len                 # straight fingers end at the 66.5 mm tip
FINGER_PIVOT_Y = 0.0085                           # half the gear spacing (meshing gears, ~17 mm apart)
FINGER_LEN = L3 - FINGER_PIVOT_Z
TCP_Z = L3 - 0.008                                # between the finger pads, just short of the tips
CORK_Z = 0.072                                    # cork centre of a shuttle standing cork-up
HOME = np.array([90.0, 90.0, 90.0, 90.0, 90.0])
OPEN, CLOSED = 45.0, 150.0
CORK_COLORS = {"natural": (0.86, 0.72, 0.52, 1), "red": (0.85, 0.12, 0.12, 1), "blue": (0.15, 0.3, 0.9, 1)}


@dataclass(frozen=True)
class Task:
    language: str
    shuttles: tuple   # cork colour per shuttle; shuttle 0 is the target
    box: bool = False


TASKS = {
    "pick": Task("pick up the shuttlecock", ("natural",)),
    "pick_red": Task("pick up the red shuttlecock", ("red", "blue")),
    "pick_blue": Task("pick up the blue shuttlecock", ("blue", "red")),
    "place_box": Task("put the shuttlecock in the box", ("natural",), box=True),
}


def finger_angle(gripper):
    """Outward finger angle (deg): about 30 open at 40, past closed at 150 so the pads squeeze."""
    return 30 - (gripper - 40) * 37 / 110


def servo_to_joints(p):
    """Servo degrees -> joint radians. 90/90/90/90 is the manual's straight-up assembly pose."""
    a = math.radians(finger_angle(p[4]))
    return [math.radians(p[0] - 90),     # base: 90 faces +x, larger turns counter-clockwise (unverified)
            math.radians(p[1] - 90),     # shoulder: larger tilts the arm forward
            math.radians(p[2] - 90),     # elbow (relative): larger bends the forearm forward
            math.radians(90 - p[3]),     # wrist: larger pitches the gripper back/up
            a, a]


# ---------- MJCF ----------

def _lookat(pos, target):
    f = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    x = np.cross(f, [0, 0, 1]); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return " ".join(f"{v:.4f}" for v in (*x, *y))


def _skirt_mesh():
    ring = lambda r, z: [(r * math.cos(a), r * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    return " ".join(f"{v:.4f}" for p in ring(0.0125, 0.06) + ring(0.033, 0.0) for v in p)


def _shuttle(i):
    solid = 'contype="1" conaffinity="1"'
    return f"""
    <body name="shuttle{i}" pos="{1 + i} 2 0">
      <freejoint/>
      <geom name="skirt{i}" type="mesh" mesh="skirt" rgba="0.96 0.96 0.96 0.85" mass="0.002" friction="0.8 0.01 0.001" {solid}/>
      <geom name="cork{i}" type="cylinder" size="0.013 0.009" pos="0 0 0.069" mass="0.0025" condim="4" friction="1.5 0.02 0.001" {solid}/>
      <geom name="corktop{i}" type="sphere" size="0.013" pos="0 0 0.078" mass="0.0005" condim="4" friction="1.5 0.02 0.001" {solid}/>
    </body>"""


def _servo(pos, euler="0 0 0"):
    """AD002 micro servo body 22.8 x 12.2 x 22.7 mm with mounting tabs (visual only)."""
    return (f'<geom type="box" size="0.0114 0.0061 0.01135" pos="{pos}" euler="{euler}" material="servo"/>'
            f'<geom type="box" size="0.016 0.0061 0.0012" pos="{pos}" euler="{euler}" material="servo"/>')


def _plate_assets():
    """Mesh assets for the acrylic plates, anchored on their pivot holes (parts.json)."""
    assets, info = [], {}

    def add(name, mesh_name, origin, axis_to, flip=False):
        v, f = plates.mesh(name, origin, axis_to, flip)
        assets.append(f'<mesh name="{mesh_name}" vertex="{v}" face="{f}" inertia="shell"/>')

    up = np.array([0.0, -1.0])
    add("A05", "A05", plates.hole("A05", 0), plates.hole("A05", 0) + up)          # left upright, pivot hole
    add("A06", "A06", plates.hole("A06", 1), plates.hole("A06", 1) + up)          # right upright, servo keyhole
    add("A08", "A08", plates.hole("A08", 1), plates.hole("A08", 0))               # upper-arm links, hole to hole
    add("A07", "A07", plates.hole("A07", 3), plates.hole("A07", 2))
    add("A11", "A11", plates.hole("A11", 2), plates.hole("A11", 2) + [1, 0])      # forearm, elbow hole
    add("A10", "A10", plates.hole("A11", 2), plates.hole("A11", 2) + [1, 0])      # same elbow position
    for name in ("A04", "A02", "A14", "A15"):
        centre = np.array(plates.PARTS[name]["size"]) / 2
        add(name, name, centre, centre + [1, 0])
    # A01: the arm stands on the four standoff holes that match A02's corners
    a02 = plates.round_holes("A02")
    diagonal = max(np.linalg.norm(p - q) for p in a02 for q in a02)
    a01 = plates.round_holes("A01")
    centre = next((p + q) / 2 for p in a01 for q in a01
                  if abs(np.linalg.norm(p - q) - diagonal) < 1.0 and p[0] < q[0] and p[1] < q[1])
    away = -1.0 if centre[0] < plates.PARTS["A01"]["size"][0] / 2 else 1.0   # the rest of A01 goes behind the arm
    add("A01", "A01", centre, centre + [away, 0])
    # gear fingers: pivot on the gear hole, +x toward the tip
    for name, side_sign in (("A17", 1), ("A16", -1)):
        origin = plates.hole(name, 0)
        tip = plates.tip(name, origin)
        add(name, name, origin, tip)
        pts = np.array(plates.PARTS[name]["outline"])
        d = (tip - origin) / np.linalg.norm(tip - origin)
        rel = pts - origin
        bulge = np.mean(-(-rel[:, 0] * d[1] + rel[:, 1] * d[0]))   # mean plate-y of the outline
        near = pts[np.linalg.norm(rel, axis=1) < 14]
        info[name] = dict(length=np.linalg.norm(tip - origin) / 1000,
                          gear_r=np.max(np.linalg.norm(near - origin, axis=1)) / 1000,
                          ysign=side_sign * (1 if bulge > 0 else -1))
    return "\n    ".join(assets), info


def _finger(side, info):
    s = 1 if side == "l" else -1
    name = "A17" if side == "l" else "A16"      # A17 carries the servo horn
    length = info[name]["length"]
    return f"""
            <body name="finger_{side}" pos="{GRIP_OFFSET} {s * FINGER_PIVOT_Y:.4f} {FINGER_PIVOT_Z:.4f}">
              <joint name="finger_{side}" axis="{-s} 0 0" range="-0.5 0.9" damping="0.002" armature="0.0001"/>
              <geom type="mesh" mesh="{name}" xyaxes="0 0 1 0 {info[name]['ysign']} 0" material="acrylic" mass="0.002"/>
              <geom type="box" size="0.003 0.002 0.004" pos="0 {-s * 0.003:.4f} {length - 0.005:.4f}" rgba="0 0 0 0"
                    condim="4" friction="1.5 0.02 0.001" contype="1" conaffinity="1" mass="0.0005"/>
            </body>"""


def build_xml():
    plate_assets, info = _plate_assets()
    walls = "".join(
        f'<geom type="box" size="{sx} {sy} 0.01" pos="{px} {py} 0.01" material="wood_box" contype="1" conaffinity="1"/>'
        for sx, sy, px, py in ((0.068, 0.004, 0, 0.064), (0.068, 0.004, 0, -0.064),
                               (0.004, 0.068, 0.064, 0), (0.004, 0.068, -0.064, 0)))
    side = 'xyaxes="0 0 1 1 0 0" material="acrylic"'   # plate +x along the link, plate y forward, thickness sideways
    flat = 'xyaxes="1 0 0 0 1 0" material="acrylic"'
    return f"""
<mujoco model="adeept_ada031_v4">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="3"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.85 0.88 0.92" rgb2="0.55 0.6 0.68" width="64" height="64"/>
    <texture name="wood" type="2d" builtin="checker" rgb1="0.78 0.64 0.47" rgb2="0.74 0.6 0.44" width="256" height="256"/>
    <material name="table" texture="wood" texrepeat="6 6"/>
    <material name="acrylic" rgba="0.03 0.03 0.035 1" specular="0.7" shininess="0.9"/>
    <material name="servo" rgba="0.2 0.19 0.18 0.92"/>
    <material name="metal" rgba="0.75 0.75 0.77 1" specular="0.8"/>
    <material name="pcb" rgba="0.08 0.08 0.1 1"/>
    <material name="wood_box" rgba="0.62 0.44 0.26 1"/>
    <mesh name="skirt" vertex="{_skirt_mesh()}"/>
    {plate_assets}
  </asset>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0.5" armature="0.01"/>
  </default>
  <worldbody>
    <light pos="0.2 -0.3 1.0" dir="-0.2 0.3 -1" diffuse="0.75 0.75 0.75" castshadow="true"/>
    <light pos="-0.4 0.3 0.8" dir="0.4 -0.3 -0.8" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" material="table" contype="1" conaffinity="1" friction="0.9 0.01 0.001"/>
    <camera name="camera1" pos="0.36 -0.34 0.30" xyaxes="{_lookat((0.36, -0.34, 0.30), (0.06, 0.0, 0.09))}" fovy="50"/>

    <!-- A01 base plate on four suction cups, drive board and 18650x2 holder behind the arm -->
    <geom type="mesh" mesh="A01" pos="0 0 0.0085" {flat} mass="0.05"/>
    <geom type="cylinder" size="0.018 0.0035" pos="0.02 0.05 0.0035" rgba="0.85 0.88 0.9 0.6"/>
    <geom type="cylinder" size="0.018 0.0035" pos="0.02 -0.05 0.0035" rgba="0.85 0.88 0.9 0.6"/>
    <geom type="cylinder" size="0.018 0.0035" pos="-0.11 0.05 0.0035" rgba="0.85 0.88 0.9 0.6"/>
    <geom type="cylinder" size="0.018 0.0035" pos="-0.11 -0.05 0.0035" rgba="0.85 0.88 0.9 0.6"/>
    <geom type="box" size="0.03 0.0225 0.006" pos="-0.085 0.03 0.016" material="pcb"/>
    <geom type="box" size="0.0385 0.02 0.01" pos="-0.085 -0.03 0.02" material="acrylic"/>
    <!-- base servo under A02 on standoffs, square bearing turntable -->
    {_servo("0 0 0.026", "0 0 1.5708")}
    <geom type="cylinder" size="0.0028 0.0153" pos="0.0249 0.0249 0.0253" material="acrylic"/>
    <geom type="cylinder" size="0.0028 0.0153" pos="-0.0249 0.0249 0.0253" material="acrylic"/>
    <geom type="cylinder" size="0.0028 0.0153" pos="0.0249 -0.0249 0.0253" material="acrylic"/>
    <geom type="cylinder" size="0.0028 0.0153" pos="-0.0249 -0.0249 0.0253" material="acrylic"/>
    <geom type="mesh" mesh="A02" pos="0 0 0.0421" {flat} mass="0.02"/>
    <geom type="box" size="0.0295 0.0295 0.0062" pos="0 0 0.0497" material="metal"/>

    <body name="turret" pos="0 0 {TURNTABLE_TOP}">
      <joint name="base" axis="0 0 1" range="-3.2 3.2"/>
      <geom type="mesh" mesh="A04" pos="0 0 -0.0015" {flat} mass="0.02"/>
      <!-- uprights A05/A06 with the shoulder servo between them -->
      <geom type="mesh" mesh="A05" pos="0 0.0155 {SHOULDER_UP}" {side} mass="0.01"/>
      <geom type="mesh" mesh="A06" pos="0 -0.0155 {SHOULDER_UP}" {side} mass="0.01"/>
      {_servo(f"0 -0.004 {SHOULDER_UP - 0.006:.4f}", "1.5708 0 0")}
      <body name="upper" pos="0 0 {SHOULDER_UP}">
        <joint name="shoulder" axis="0 1 0" range="-3.2 3.2"/>
        <!-- upper-arm links A07/A08 (hole to hole = L1) -->
        <geom type="mesh" mesh="A08" pos="0 0.0195 0" {side} mass="0.01"/>
        <geom type="mesh" mesh="A07" pos="0 -0.0195 0" {side} mass="0.01"/>
        <geom type="cylinder" size="0.0028 0.0195" pos="0 0 0" euler="1.5708 0 0" material="metal"/>
        <body name="forearm" pos="0 0 {L1}">
          <joint name="elbow" axis="0 1 0" range="-3.2 3.2"/>
          <!-- forearm plates A10/A11 carrying the elbow servo -->
          <geom type="mesh" mesh="A11" pos="0 0.0225 0" {side} mass="0.01"/>
          <geom type="mesh" mesh="A10" pos="0 -0.0225 0" {side} mass="0.01"/>
          {_servo("0 -0.008 0.004", "1.5708 0 0")}
          <geom type="cylinder" size="0.0028 0.0225" pos="0 0 0" euler="1.5708 0 0" material="metal"/>
          <body name="gripper" pos="0 0 {L2}">
            <joint name="wrist" axis="0 1 0" range="-3.2 3.2"/>
            <!-- wrist servo, brackets A14, gripper plate A15 with the gripper servo behind it -->
            {_servo("0 0 0.002", "1.5708 0 0")}
            <geom type="mesh" mesh="A14" pos="0.004 0.016 0.012" {side} mass="0.005"/>
            <geom type="mesh" mesh="A14" pos="0.004 -0.016 0.012" {side} mass="0.005"/>
            <geom type="mesh" mesh="A15" pos="{GRIP_OFFSET - 0.0035:.4f} 0 {FINGER_PIVOT_Z - 0.004:.4f}" xyaxes="0 1 0 0 0 1" material="acrylic" mass="0.005"/>
            <geom type="box" size="0.0015 0.016 0.014" pos="{GRIP_OFFSET - 0.0035:.4f} 0 {FINGER_PIVOT_Z - 0.004:.4f}" rgba="0 0 0 0" contype="1" conaffinity="1" mass="0.005"/>
            {_servo(f"{GRIP_OFFSET - 0.016:.4f} 0 {FINGER_PIVOT_Z - 0.006:.4f}", "0 1.5708 0")}
            <camera name="camera2" pos="{GRIP_OFFSET + 0.012:.4f} 0 0.012" xyaxes="0 1 0 1 0 0" fovy="70"/>
            <site name="tcp" pos="{GRIP_OFFSET} 0 {TCP_Z}" size="0.002" rgba="1 0 0 0"/>
            <site name="axis_tip" pos="{GRIP_OFFSET} 0 {TCP_Z + 0.05}" size="0.002" rgba="1 0 0 0"/>
            {_finger("l", info)}{_finger("r", info)}
          </body>
        </body>
      </body>
    </body>

    <body name="box" mocap="true" pos="1 -2 0">
      <geom type="box" size="0.068 0.068 0.002" pos="0 0 0.002" material="wood_box" contype="1" conaffinity="1"/>
      {walls}
    </body>
    {_shuttle(0)}{_shuttle(1)}
  </worldbody>
  <actuator>
    <position joint="base" kp="8" forcerange="-0.25 0.25"/>
    <position joint="shoulder" kp="8" forcerange="-0.25 0.25"/>
    <position joint="elbow" kp="8" forcerange="-0.25 0.25"/>
    <position joint="wrist" kp="4" forcerange="-0.2 0.2"/>
    <position joint="finger_l" kp="1.0" forcerange="-0.04 0.04"/>
    <position joint="finger_r" kp="1.0" forcerange="-0.04 0.04"/>
  </actuator>
</mujoco>"""


# ---------- environment ----------

class AdeeptSimEnv:
    """One tick = 1/fps s; each axis moves at most max_step degrees toward the action per tick.

    The real firmware moves one axis by 1 degree per 300 ms; fps=3, max_step=1,
    one_axis=True reproduces that timing, the defaults trade it for shorter episodes.
    Reuse one env across tasks with reset(task=...): a second Renderer rendered black on the MSI.
    """

    def __init__(self, task="pick", fps=10, max_step=3.0, one_axis=False, max_ticks=200, size=(480, 640)):
        self.task_name, self.task = task, TASKS[task]
        self.fps, self.max_step, self.one_axis, self.max_ticks = fps, max_step, one_axis, max_ticks
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)   # kinematics for the expert's IK
        self.renderer = mujoco.Renderer(self.model, *size)
        self.substeps = round(1 / fps / self.model.opt.timestep)
        self.rng = np.random.default_rng()
        self.shuttle_qpos = [self.model.jnt_qposadr[self.model.body(f"shuttle{i}").jntadr[0]] for i in range(2)]
        self.tcp, self.axis_tip = self.model.site("tcp").id, self.model.site("axis_tip").id

    @property
    def language(self):
        return self.task.language

    # -- kinematics --
    def fk(self, p):
        self.scratch.qpos[:6] = servo_to_joints(p)
        mujoco.mj_kinematics(self.model, self.scratch)
        tcp = self.scratch.site_xpos[self.tcp].copy()
        return tcp, (self.scratch.site_xpos[self.axis_tip] - tcp) / 0.05

    def grasp_pose(self, target, gripper, near=None):
        """Servo targets that put the TCP on target with the gripper as close to pointing down as the limits allow.

        near: a previous pose; solutions on the same arm configuration are preferred so that
        consecutive waypoints do not flip between elbow-up and elbow-down.
        """
        target = np.asarray(target, float)
        base = 90 + math.degrees(math.atan2(target[1], target[0]))
        if not LOW[0] <= base <= HIGH[0]:
            return None

        def residual(q3):
            tcp, axis = self.fk([base, *q3, gripper])
            return np.concatenate([(tcp - target) * 1000, [8 * (1 + axis[2])]])  # mm, then a soft "point down"

        best = None
        seeds = [(120, 150, 20), (130, 140, 30), (100, 170, 10), (170, 80, 20), (150, 110, 30)]
        if near is not None:
            seeds.insert(0, tuple(near[1:4]))
        for seed in seeds:
            q = np.array(seed, float)
            for _ in range(60):
                r = residual(q)
                jac = np.stack([(residual(q + d) - r) / 0.5 for d in np.eye(3) * 0.5], axis=1)
                step = np.linalg.solve(jac.T @ jac + 0.5 * np.eye(3), -jac.T @ r)
                q = np.clip(q + np.clip(step, -10, 10), LOW[1:4], HIGH[1:4])
            tcp, axis = self.fk([base, *q, gripper])
            cost = axis[2] + (0.0 if near is None else 0.01 * np.abs(q - near[1:4]).max())
            if np.linalg.norm(tcp - target) < 0.002 and (best is None or cost < best[1]):
                best = (np.array([base, *q, gripper]), cost)
        return None if best is None else best[0]

    # -- scene sampling --
    def _sample_spot(self, taken, min_gap=0.085, heights=(CORK_Z, CORK_Z + 0.05), radius=(0.08, 0.13), yaw=55):
        for _ in range(300):
            heading = math.radians(self.rng.uniform(-yaw, yaw))
            r = self.rng.uniform(*radius)
            xy = np.array([r * math.cos(heading), r * math.sin(heading)])
            if any(np.linalg.norm(xy - t) < min_gap for t in taken):
                continue
            if all(self.grasp_pose((*xy, z), OPEN) is not None for z in heights):
                return xy
        raise RuntimeError("no reachable spot")

    def reset(self, seed=None, task=None):
        if task is not None:
            self.task_name, self.task = task, TASKS[task]
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        taken = []
        for i in range(2):
            adr = self.shuttle_qpos[i]
            if i < len(self.task.shuttles):
                xy = self._sample_spot(taken)
                taken.append(xy)
                self.data.qpos[adr:adr + 7] = [*xy, 0.0005, 1, 0, 0, 0]
                color = CORK_COLORS[self.task.shuttles[i]]
                for name in (f"cork{i}", f"corktop{i}"):
                    self.model.geom(name).rgba = color
            else:
                self.data.qpos[adr:adr + 7] = [1 + i, 2, 0, 1, 0, 0, 0]
        self.box_xy = (self._sample_spot(taken, min_gap=0.11, heights=(CORK_Z + 0.05,), radius=(0.10, 0.17), yaw=75)
                       if self.task.box else np.array([1.0, -2.0]))
        self.data.mocap_pos[0] = [*self.box_xy, 0]
        self.state = HOME.copy()
        self.data.qpos[:6] = servo_to_joints(self.state)
        self.data.ctrl[:] = servo_to_joints(self.state)
        self.ticks = 0
        for _ in range(100):  # let the shuttles settle
            mujoco.mj_step(self.model, self.data)
        return self._observation()

    def step(self, action):
        target = np.clip(np.asarray(action, dtype=float), LOW, HIGH)
        delta = np.clip(target - self.state, -self.max_step, self.max_step)
        if self.one_axis:
            keep = np.argmax(np.abs(delta))
            delta = np.where(np.arange(5) == keep, delta, 0)
        self.state = self.state + delta
        self.data.ctrl[:] = servo_to_joints(self.state)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.ticks += 1
        success = self.success()
        return self._observation(), success, success or self.ticks >= self.max_ticks

    # -- observation & success --
    def render(self, camera):
        self.renderer.update_scene(self.data, camera)
        return self.renderer.render().copy()

    def _observation(self):
        return {"observation.state": self.state.astype(np.float32),
                "observation.images.camera1": self.render("camera1"),
                "observation.images.camera2": self.render("camera2")}

    def shuttle_pos(self, i):
        adr = self.shuttle_qpos[i]
        return self.data.qpos[adr:adr + 3].copy()

    def success(self):
        lifted = lambda i: self.shuttle_pos(i)[2] > 0.03
        if self.task.box:
            p = self.shuttle_pos(0)
            return bool(np.all(np.abs(p[:2] - self.box_xy) < 0.06) and p[2] < 0.03 and self.state[4] < 80)
        others_down = all(not lifted(i) for i in range(1, len(self.task.shuttles)))
        return bool(lifted(0) and others_down)

    # -- scripted expert --
    def expert_waypoints(self):
        """Servo waypoints (with hold ticks) that solve the task from the current scene."""
        cork = np.array([*self.shuttle_pos(0)[:2], CORK_Z])
        above = cork + [0, 0, 0.05]
        pre = self.grasp_pose(above, OPEN)
        grasp = None if pre is None else self.grasp_pose(cork, OPEN, near=pre)
        if grasp is None:
            return [(None, 0)]
        closed = grasp.copy(); closed[4] = CLOSED
        lift = pre.copy(); lift[4] = CLOSED
        plan = [(pre, 0), (grasp, 2), (closed, 4), (lift, 4)]
        if self.task.box:
            over_box = self.grasp_pose((*self.box_xy, CORK_Z + 0.05), CLOSED, near=lift)
            if over_box is None:
                return [(None, 0)]
            release = over_box.copy(); release[4] = OPEN
            plan += [(over_box, 2), (release, 6)]
        return plan


class Expert:
    """Moves all axes together along straight lines in servo space between waypoints."""

    def __init__(self, env, noise=0.0):
        self.plan = env.expert_waypoints()
        self.env, self.noise, self.index, self.hold = env, noise, 0, None
        if any(p is None for p, _ in self.plan):
            raise RuntimeError("unreachable waypoint")

    def act(self):
        state = self.env.state
        if self.index >= len(self.plan):
            return state.copy()
        if self.env.task.box and self.index == 4 and self.hold is None and not getattr(self, "aimed", False):
            # the shuttle hangs off-centre in the tilted gripper: aim the box drop by its actual offset
            # (offset kept in the arm's heading frame, since the base turns on the way to the box)
            tcp, _ = self.env.fk(state)
            turn = math.radians(90 + math.degrees(math.atan2(self.env.box_xy[1], self.env.box_xy[0])) - state[0])
            c, s = math.cos(turn), math.sin(turn)
            offset = self.env.shuttle_pos(0)[:2] - tcp[:2]
            offset = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])
            over = self.env.grasp_pose((*(self.env.box_xy - offset), CORK_Z + 0.05), CLOSED, near=state)
            if over is not None:
                release = over.copy(); release[4] = OPEN
                self.plan[4], self.plan[5] = (over, self.plan[4][1]), (release, self.plan[5][1])
            self.aimed = True
        goal, hold = self.plan[self.index]
        delta = goal - state
        if np.max(np.abs(delta)) < 0.5:
            self.hold = hold if self.hold is None else self.hold - 1
            if self.hold <= 0:
                self.index, self.hold = self.index + 1, None
            return goal.copy()
        steps = math.ceil(np.max(np.abs(delta)) / self.env.max_step)
        action = state + delta / steps
        if self.noise:
            action[:4] += self.env.rng.normal(0, self.noise, 4)
        return action
