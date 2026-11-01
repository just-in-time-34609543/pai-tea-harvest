"""Unreal Editor Python (full editor, -RenderOffScreen): render preview stills of /Game/Tea/TeaGreenhouse.

Waits for shader compilation and Lumen / sky-light convergence by ticking the editor, then takes one
high-resolution screenshot per camera view through the regular viewport renderer, and quits.
Screenshots land in <Project>/Saved/Screenshots/LinuxEditor/; copy them to output/adeept/ue_preview.
"""
import json
import os

import unreal

REPO = os.environ.get("TEA_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # リポジトリのルート
_vf = os.path.join(REPO, "output", "adeept", "ue_views.json")
VIEWS = json.loads(os.environ.get("TEA_VIEWS", "{}")) or (json.load(open(_vf)) if os.path.exists(_vf) else None) or {
    "overview": [[1.05, -1.05, 0.62], [0.0, 0.05, 0.18], 55],
    "plant_close": [[0.20, -0.30, 0.38], [0.0, 0.11, 0.26], 45],
    "inside_low": [[-0.38, -0.20, 0.16], [0.05, 0.10, 0.25], 70],
}
WARM = int(os.environ.get("TEA_WARM_FRAMES", "400"))
GAP = 90
RES = (int(os.environ.get("TEA_W", "1600")), int(os.environ.get("TEA_H", "1000")))
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level("/Game/Tea/TeaGreenhouse")
cams = []
for name, (eye, at, fov) in VIEWS.items():
    e = unreal.Vector(eye[0] * 100, -eye[1] * 100, eye[2] * 100)
    t = unreal.Vector(at[0] * 100, -at[1] * 100, at[2] * 100)
    cam = eas.spawn_actor_from_class(unreal.CameraActor, e, unreal.MathLibrary.find_look_at_rotation(e, t))
    cam.camera_component.set_editor_property("field_of_view", fov)
    cam.camera_component.set_editor_property("constrain_aspect_ratio", False)
    cam.set_actor_label("PreviewCam_" + name)
    cams.append((name, cam))

state = {"frame": 0, "next": WARM, "i": 0, "handle": None, "busy": False}


def tick(dt):
    # take_high_res_screenshot can tick Slate re-entrantly, so advance the state before calling it
    if state["busy"]:
        return
    state["frame"] += 1
    if state["frame"] < state["next"]:
        return
    if state["i"] < len(cams):
        name, cam = cams[state["i"]]
        state["i"] += 1
        state["next"] = state["frame"] + GAP
        state["busy"] = True
        try:
            unreal.AutomationLibrary.take_high_res_screenshot(RES[0], RES[1], f"tea_{name}.png", camera=cam)
        finally:
            state["busy"] = False
        unreal.log(f"TEA screenshot requested {name} at frame {state['frame']}")
        return
    unreal.log("TEA all screenshots requested; quitting")
    unreal.unregister_slate_post_tick_callback(state["handle"])
    for _, cam in cams:
        eas.destroy_actor(cam)
    unreal.SystemLibrary.quit_editor()


state["handle"] = unreal.register_slate_post_tick_callback(tick)
