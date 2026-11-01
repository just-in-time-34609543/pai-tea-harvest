# Adeept ADA031 V4 simulation: spec

MuJoCo 3.17.0, Python 3.10, lerobot 0.4.4, torch 2.10.0+cu126, transformers 4.57.6.

## Files

- `env.py`: `build_xml()` writes the whole scene and `AdeeptSimEnv` runs it.
  - The scene is the arm, two shuttles, a box and two cameras.
  - The env does reset/step/render/success, plus a scripted expert.
- `adeept_arm.xml`: the output of `build_xml()`.
  - It is self-contained: the meshes are inline and there are no external files.
  - Regenerate it after editing env.py:
    ```
    python -c "from env import build_xml; open('adeept_arm.xml','w').write(build_xml())"
    ```
- `parts.json`: plate outlines and holes in mm, traced from the kit manual's Packing List. `plates.py` turns them into 3 mm plate meshes.

## Geometry (manual p.114)

| Part | Value |
|---|---|
| Turntable top | 60 mm |
| Shoulder above turntable | 40 mm |
| L1 | 56.9 mm (the drawing says 65; 要実測) |
| L2 | 70 mm |
| L3 | 66.5 mm |
| Gripper offset | 13 mm |

Units are SI and z is up. The base sits at the origin, and the arm reaches along +x when base = 90.

## Joints and actuators

The interface uses servo degrees, not joint radians. `servo_to_joints()` maps them; servo 90 means straight up.

| # | Servo | MJCF joint | Mapping (rad) | Sim range (deg) | Actuator |
|---|---|---|---|---|---|
| 0 | base | `base` (z) | p−90 | 0–180 | position kp 8, force ±0.25 |
| 1 | shoulder | `shoulder` (y) | p−90 | 0–180 | position kp 8, force ±0.25 |
| 2 | elbow | `elbow` (y) | p−90 | 0–180 | position kp 8, force ±0.25 |
| 3 | wrist | `wrist` (y) | 90−p | 0–180 | position kp 4, force ±0.2 |
| 4 | gripper | `finger_l`, `finger_r` | 30−(g−40)·37/110 deg, both fingers | 40–150 | position kp 1.0, force ±0.04 each |

- Arm joints have damping 0.5 and armature 0.01. Finger joints have range −0.5 to 0.9 rad, damping 0.002 and armature 0.0001.
- Gripper: 45 is open and 150 is closed. The two fingers are mirrored and both are driven from the one servo value. The cork contacts use condim 4 and friction 1.5.
- The real firmware (CommandArm 1.7.0) limits are low {0,45,60,50,40} and high {180,165,115,130,150}. The sim uses 0–180 because the real horns sit about 50° off the manual. The base direction is not yet checked against the real arm.

## Control

- Physics: timestep 2 ms, implicitfast integrator, elliptic cone, noslip 3.
- One env tick lasts 1/fps s, with fps 10 by default, so 50 substeps.
- Each tick, every axis moves at most `max_step` = 3° toward the action, and the result is sent as position targets.
- `one_axis=True` with fps 3 and max_step 1 imitates the real firmware, which moves one axis by 1° per 300 ms.
- An episode ends on success or after `max_ticks` = 200.

## Cameras (480×640 RGB, the MuJoCo default renderer)

- `camera1` is fixed at (0.36, −0.34, 0.30), looks at (0.06, 0, 0.09), and has fovy 50.
- `camera2` is on the wrist, under the gripper, pointing along the gripper, with fovy 70.

## What LeRobot sees

| Key | Type | Contents |
|---|---|---|
| `observation.state` | float32[5] | current servo targets (base, shoulder, elbow, wrist, gripper), deg |
| `action` | float32[5] | goal servo degrees, same order |
| `observation.images.camera1` | video 480×640×3 | fixed camera |
| `observation.images.camera2` | video 480×640×3 | wrist camera |
| task | string | one of the instructions below |

- The dataset uses fps 10.
- SmolVLA pads state and action to 32. Its image keys camera1/2 match the smolvla_base config, and camera3 is absent.

## Tasks

| Name | Instruction | Scene | Success |
|---|---|---|---|
| pick | pick up the shuttlecock | 1 shuttle (natural cork) | shuttle z > 0.03 |
| pick_red | pick up the red shuttlecock | red target, blue distractor | target lifted, distractor down |
| pick_blue | pick up the blue shuttlecock | blue target, red distractor | target lifted, distractor down |
| place_box | put the shuttlecock in the box | 1 shuttle, box | shuttle within ±0.06 of the box centre, z < 0.03, gripper < 80 |

- Shuttles are placed at radius 0.08–0.13 m, heading ±55°, at least 85 mm apart, and only where the IK can reach them.
- Demo seeds start at 0; eval seeds are 100000+.
