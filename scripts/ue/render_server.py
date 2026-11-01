"""Unreal Editor Python (start-up script in the full editor, -RenderOffScreen): pose-driven render server.

MuJoCo computes the physics (scripts/harvest_env.py); this server only draws. Every request carries the world poses
of the arm / platform / basket / shoot bodies and of the two cameras; the server moves the actors, renders both
cameras with SceneCapture2D in the tea greenhouse level, writes PNGs and replies.

Protocol (TCP 127.0.0.1:TEA_RENDER_PORT, one JSON object per line, reply one JSON line):
  {"bodies": {name: [x, y, z, qw, qx, qy, qz], ...},              MuJoCo world frame (metres)
   "cams": {name: [px, py, pz, fx, fy, fz, ux, uy, uz, fovy_deg]},  position, forward, up (MuJoCo frame)
   "out": "/path/prefix"}                                           -> writes <prefix>_<cam>.png
  {"cmd": "quit"}
"""
import json
import os
import socket

import unreal

REPO = os.environ.get("TEA_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # リポジトリのルート
MESHES = os.path.join(REPO, "output", "adeept", "arm_meshes")
PLANTS = os.path.join(REPO, "output", "adeept", "plants")
PORT = int(os.environ.get("TEA_RENDER_PORT", "5600"))
W, H = int(os.environ.get("TEA_RENDER_W", "640")), int(os.environ.get("TEA_RENDER_H", "480"))
GAME = "/Game/Tea"
ML = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def log(m):
    unreal.log("TEA-RS " + str(m))


def import_obj(path):
    name = os.path.splitext(os.path.basename(path))[0]
    t = unreal.AssetImportTask()
    t.filename = path; t.destination_path = f"{GAME}/Robot"; t.destination_name = "R_" + name
    t.automated = True; t.replace_existing = True; t.save = True
    tools.import_asset_tasks([t])
    return unreal.load_asset(f"{GAME}/Robot/R_{name}")


def color_material():
    path = f"{GAME}/Materials/M_Color"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    m = tools.create_asset("M_Color", f"{GAME}/Materials", unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("two_sided", True)
    col = ML.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -500, 0)
    col.set_editor_property("parameter_name", "Color")
    ML.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
    for i, (n, v, prop) in enumerate((("Roughness", .45, unreal.MaterialProperty.MP_ROUGHNESS), ("Metallic", 0., unreal.MaterialProperty.MP_METALLIC))):
        e = ML.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -500, 200 + 100 * i)
        e.set_editor_property("parameter_name", n); e.set_editor_property("default_value", v)
        ML.connect_material_property(e, "", prop)
    ML.recompile_material(m); unreal.EditorAssetLibrary.save_loaded_asset(m)
    return m


def color_instance(parent, rgb, rough=.45, metal=0.):
    name = "MI_Col_%02x%02x%02x" % tuple(int(255 * c) for c in rgb)
    path = f"{GAME}/Materials/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    mi = tools.create_asset(name, f"{GAME}/Materials", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    ML.set_material_instance_parent(mi, parent)
    # MuJoCo colours are sRGB-ish; convert to linear for Unreal
    ML.set_material_instance_vector_parameter_value(mi, "Color", unreal.LinearColor(*[c ** 2.2 for c in rgb], 1.0))
    ML.set_material_instance_scalar_parameter_value(mi, "Roughness", rough)
    ML.set_material_instance_scalar_parameter_value(mi, "Metallic", metal)
    ML.update_material_instance(mi); unreal.EditorAssetLibrary.save_loaded_asset(mi)
    return mi


def spawn(mesh, mat, label):
    a = eas.spawn_actor_from_object(mesh, unreal.Vector(0, 0, -10000), unreal.Rotator(0, 0, 0))
    a.set_actor_scale3d(unreal.Vector(100, 100, 100))
    a.static_mesh_component.set_material(0, mat)
    a.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    a.set_actor_label(label)
    return a


def ue_loc(p):
    return unreal.Vector(p[0] * 100, -p[1] * 100, p[2] * 100)


def ue_quat(q):  # MuJoCo (w,x,y,z), right-handed -> UE with y mirrored: (w, -x, y, -z)
    w, x, y, z = q
    return unreal.Quat(-x, y, -z, w)


unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(f"{GAME}/TeaGreenhouse")
parent = color_material()
bodies = json.load(open(os.path.join(MESHES, "bodies.json")))
actors = {}
for name, parts in bodies.items():
    acts = []
    for p in parts:
        metal = .6 if max(p["rgb"]) - min(p["rgb"]) < .08 and sum(p["rgb"]) < 1.6 else 0.
        acts.append(spawn(import_obj(os.path.join(MESHES, p["obj"])), color_instance(parent, p["rgb"], metal=metal), f"RB_{name}_{len(acts)}"))
    actors[name] = acts
# plucked leaves: part 0 (green leaf blade) follows the leaf body; parts 1-2 (petiole, twig) follow fixture<i>,
# so they stay on the bush.
mi = {k: unreal.load_asset(f"{GAME}/Materials/{n}") for k, n in (("leaf", "MI_LeafHealthy"), ("twig", "MI_Twig"))}
shoot_parts = [(import_obj(os.path.join(PLANTS, "leaf_green.obj")), mi["leaf"], 0.0), (import_obj(os.path.join(PLANTS, "leaf_petiole.obj")), mi["twig"], 0.0),
               (import_obj(os.path.join(MESHES, "shoot_long_stem.obj")), mi["twig"], -0.012)]
shoot_offsets = {}
for name in [n for n in bodies if n.startswith("shoot")]:
    actors[name] = []
    for k, (mesh, mat, dz) in enumerate(shoot_parts):
        actors[name].append(spawn(mesh, mat, f"RB_{name}_{k}"))
        shoot_offsets[(name, k)] = dz
# cameras
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
ppv = [a for a in eas.get_all_level_actors() if isinstance(a, unreal.PostProcessVolume)]
cams, dcams = {}, {}
CAMS = os.environ.get("TEA_CAMS", "camera1,camera2,camera3,camera4,camera5").split(",")
CINE = [int(v) for v in os.environ.get("TEA_CINE_RES", "1280x720").split("x")]
for cname in CAMS:
    cw, ch = (CINE if cname == "cine" else (W, H))
    rt = unreal.RenderingLibrary.create_render_target2d(world, cw, ch, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 200), unreal.Rotator(0, 0, 0))
    cc = cap.capture_component2d
    cc.set_editor_property("texture_target", rt)
    cc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    cc.set_editor_property("capture_every_frame", False)
    cc.set_editor_property("always_persist_rendering_state", True)
    if ppv:
        cc.set_editor_property("post_process_settings", ppv[0].settings); cc.set_editor_property("post_process_blend_weight", 1.0)
    cams[cname] = (cap, cc, rt)
    # depth twin: linear scene depth (cm along the view axis) into a float target
    drt = unreal.RenderingLibrary.create_render_target2d(world, cw, ch, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    dcap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 200), unreal.Rotator(0, 0, 0))
    dcc = dcap.capture_component2d
    dcc.set_editor_property("texture_target", drt)
    dcc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_SCENE_DEPTH)
    dcc.set_editor_property("capture_every_frame", False)
    dcams[cname] = (dcap, dcc, drt)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", PORT)); srv.listen(1); srv.setblocking(False)
state = {"conn": None, "buf": b"", "handle": None, "n": 0}
log(f"listening on {PORT}")


def apply(req):
    bodies_req = req.get("bodies", {})
    for name, pose in bodies_req.items():
        for k, a in enumerate(actors.get(name, [])):
            if name.startswith("shoot") and k >= 1:
                pose = bodies_req.get("fixture" + name[len("shoot"):], pose)
            loc = list(pose[:3])
            if (name, k) in shoot_offsets:
                loc[2] += shoot_offsets[(name, k)]
            a.set_actor_location_and_rotation(ue_loc(loc), ue_quat(pose[3:7]).rotator(), False, True)
    for cname, c in req.get("cams", {}).items():
        cap, cc, rt = cams[cname]
        f = unreal.Vector(c[3], -c[4], c[5]); u = unreal.Vector(c[6], -c[7], c[8])
        rot = unreal.MathLibrary.make_rot_from_xz(f, u)
        cap.set_actor_location_and_rotation(ue_loc(c[:3]), rot, False, True)
        cc.set_editor_property("fov_angle", float(c[9]))
        dcap, dcc, drt = dcams[cname]
        dcap.set_actor_location_and_rotation(ue_loc(c[:3]), rot, False, True)
        dcc.set_editor_property("fov_angle", float(c[9]))
    out = req.get("out")
    files = {}
    if out:
        d, prefix = os.path.split(out)
        for cname in req.get("cams", {}):
            cap, cc, rt = cams[cname]
            cc.capture_scene()
            unreal.RenderingLibrary.export_render_target(world, rt, d, f"{prefix}_{cname}.png")
            files[cname] = os.path.join(d, f"{prefix}_{cname}.png")
            if req.get("depth"):
                dcap, dcc, drt = dcams[cname]
                dcc.capture_scene()
                unreal.RenderingLibrary.export_render_target(world, drt, d, f"{prefix}_{cname}_depth.exr")
                files[cname + "_depth"] = os.path.join(d, f"{prefix}_{cname}_depth.exr")
    return files


def tick(dt):
    if state["conn"] is None:
        try:
            state["conn"], _ = srv.accept(); state["conn"].setblocking(False); log("client connected")
        except BlockingIOError:
            return
    try:
        data = state["conn"].recv(1 << 20)
        if not data:
            state["conn"] = None; return
        state["buf"] += data
    except BlockingIOError:
        pass
    while b"\n" in state["buf"]:
        line, state["buf"] = state["buf"].split(b"\n", 1)
        req = json.loads(line)
        if req.get("cmd") == "quit":
            state["conn"].sendall(b'{"ok": true}\n')
            unreal.unregister_slate_post_tick_callback(state["handle"]); srv.close()
            unreal.SystemLibrary.quit_editor(); return
        try:
            files = apply(req); reply = {"ok": True, "files": files}
        except Exception as e:  # report and keep serving
            reply = {"ok": False, "error": repr(e)}
        state["conn"].setblocking(True); state["conn"].sendall((json.dumps(reply) + "\n").encode()); state["conn"].setblocking(False)
        state["n"] += 1


state["handle"] = unreal.register_slate_post_tick_callback(tick)
