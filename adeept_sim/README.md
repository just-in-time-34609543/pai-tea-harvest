# Adeept ADA031 V4 arm simulator (MuJoCo)

The arm model, servo mapping, inverse kinematics and scripted expert used by `scripts/harvest_env.py`.
These files were written for this project; they are included here so the repository runs on its own.

- `env.py`: `build_xml()` builds the arm scene and `AdeeptSimEnv` runs it; also the IK and the scripted expert.
- `plates.py` and `parts.json`: outlines and holes of the kit acrylic plates in mm, traced from the dimensions in the kit manual, turned into 3 mm plate meshes.
- `adeept_arm.xml`: the output of `build_xml()` (self-contained, meshes inline). Regenerate it after editing `env.py`:
  `python -c "from env import build_xml; open(\x27adeept_arm.xml\x27,\x27w\x27).write(build_xml())"`
- `SPEC.md`: geometry, joints and actuators.

`scripts/harvest_env.py` reads this folder by default; set `ADEEPT_SIM` to use another copy.
