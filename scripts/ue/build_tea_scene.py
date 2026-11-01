"""Unreal Editor Python (UnrealEditor-Cmd -ExecutePythonScript): build the real-scale tea greenhouse level.

Inputs come from scripts/tea_plant_gen.py (output/adeept/plants) and scripts/ue_env_gen.py (output/adeept/env).
Frames: MuJoCo-style metres (x along the house, y across, z up) -> UE cm with y negated (as the OBJ importer does).
Writes output/adeept/ue_views.json (camera views incl. close-ups of real shoot tips) for render_tea_views.py.
Look-dev targets are the field photos (see scripts/tea_plant_gen.py and scripts/ue_env_gen.py docstrings).
"""
import json
import math
import os
import random

import unreal

REPO = os.environ.get("TEA_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # リポジトリのルート
OUT = os.path.join(REPO, "output", "adeept")
GAME = "/Game/Tea"
ML = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
cfg = json.load(open(os.path.join(REPO, "config", "adeept_scene.json")))
W = cfg["world"]
rnd = random.Random(cfg["seed"])


def log(msg):
    unreal.log("TEA " + str(msg))


def import_file(path, dest):
    name = os.path.splitext(os.path.basename(path))[0]
    t = unreal.AssetImportTask()
    t.filename = path; t.destination_path = dest; t.destination_name = name
    t.automated = True; t.replace_existing = True; t.save = True
    tools.import_asset_tasks([t])
    a = unreal.load_asset(f"{dest}/{name}")
    if a is None:
        a = unreal.load_asset(list(t.imported_object_paths)[0])
    return a


def scalar(m, name, default, x, y):
    e = ML.create_material_expression(m, unreal.MaterialExpressionScalarParameter, x, y)
    e.set_editor_property("parameter_name", name); e.set_editor_property("default_value", default)
    return e


def get_material(name, kind):
    """kind: 'foliage' (two-sided foliage), 'opaque', or 'translucent' (constant colour + opacity)."""
    path = f"{GAME}/Materials/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    m = tools.create_asset(name, f"{GAME}/Materials", unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("two_sided", True)
    if kind == "masked":
        m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
        tex = ML.create_material_expression(m, unreal.MaterialExpressionTextureSampleParameter2D, -700, 0)
        tex.set_editor_property("parameter_name", "Tex")
        ML.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
        ML.connect_material_property(tex, "A", unreal.MaterialProperty.MP_OPACITY_MASK)
        ML.connect_material_property(scalar(m, "Roughness", .9, -700, 300), "", unreal.MaterialProperty.MP_ROUGHNESS)
        ML.connect_material_property(scalar(m, "Specular", .2, -700, 400), "", unreal.MaterialProperty.MP_SPECULAR)
    elif kind == "translucent":
        m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        col = ML.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -500, 0)
        col.set_editor_property("parameter_name", "Color")
        ML.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
        ML.connect_material_property(scalar(m, "Opacity", .5, -500, 200), "", unreal.MaterialProperty.MP_OPACITY)
        ML.connect_material_property(scalar(m, "Roughness", .8, -500, 300), "", unreal.MaterialProperty.MP_ROUGHNESS)
        ML.connect_material_property(scalar(m, "Specular", .0, -500, 400), "", unreal.MaterialProperty.MP_SPECULAR)
        m.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_SURFACE)
    else:
        if kind == "foliage":
            m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
        tex = ML.create_material_expression(m, unreal.MaterialExpressionTextureSampleParameter2D, -700, 0)
        tex.set_editor_property("parameter_name", "Tex")
        tint = ML.create_material_expression(m, unreal.MaterialExpressionMultiply, -350, 0)
        ML.connect_material_expressions(tex, "RGB", tint, "A")
        ML.connect_material_expressions(scalar(m, "Brightness", 1.0, -700, 250), "", tint, "B")
        ML.connect_material_property(tint, "", unreal.MaterialProperty.MP_BASE_COLOR)
        ML.connect_material_property(scalar(m, "Roughness", .5, -700, 350), "", unreal.MaterialProperty.MP_ROUGHNESS)
        ML.connect_material_property(scalar(m, "Metallic", 0., -700, 450), "", unreal.MaterialProperty.MP_METALLIC)
        ML.connect_material_property(scalar(m, "Specular", .5, -700, 550), "", unreal.MaterialProperty.MP_SPECULAR)
        if kind == "foliage":  # light through the blade: tinted, darker copy of the albedo
            sss = ML.create_material_expression(m, unreal.MaterialExpressionMultiply, -350, 200)
            ML.connect_material_expressions(tint, "", sss, "A")
            ML.connect_material_expressions(scalar(m, "Translucency", .5, -700, 650), "", sss, "B")
            ML.connect_material_property(sss, "", unreal.MaterialProperty.MP_SUBSURFACE_COLOR)
    ML.recompile_material(m)
    unreal.EditorAssetLibrary.save_loaded_asset(m)
    return m


def instance(parent, name, tex=None, color=None, **scalars):
    path = f"{GAME}/Materials/{name}"
    mi = unreal.load_asset(path) if unreal.EditorAssetLibrary.does_asset_exist(path) else tools.create_asset(
        name, f"{GAME}/Materials", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    ML.set_material_instance_parent(mi, parent)
    if tex is not None:
        ML.set_material_instance_texture_parameter_value(mi, "Tex", tex)
    if color is not None:
        ML.set_material_instance_vector_parameter_value(mi, "Color", unreal.LinearColor(*color, 1.0))
    for k, v in scalars.items():
        ML.set_material_instance_scalar_parameter_value(mi, k, v)
    ML.update_material_instance(mi)
    unreal.EditorAssetLibrary.save_loaded_asset(mi)
    return mi


def ue_vec(p):
    return unreal.Vector(p[0] * 100, -p[1] * 100, p[2] * 100)


def place(mesh, mat, loc=(0, 0, 0), scale=1.0, yaw=0.0, label=None, shadows=True):
    a = eas.spawn_actor_from_object(mesh, ue_vec(loc), unreal.Rotator(0, 0, -yaw))
    a.set_actor_scale3d(unreal.Vector(100 * scale, 100 * scale, 100 * scale))
    comp = a.static_mesh_component
    comp.set_material(0, mat)
    comp.set_editor_property("cast_shadow", shadows)
    if label:
        a.set_actor_label(label)
    return a


def bush_tf(p, loc, scale, yaw_deg):
    """bush-local metres -> world metres (yaw about z, uniform scale, translation)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x, y, z = (v * scale for v in p)
    return [loc[0] + c * x - s * y, loc[1] + s * x + c * y, loc[2] + z]


def main():
    pdir, edir = os.path.join(OUT, "plants"), os.path.join(OUT, "env")
    tex = lambda d, n: import_file(os.path.join(d, n), f"{GAME}/Textures")
    T = {k: tex(pdir, f"tex_leaf_{k}.png") for k in ("a", "b", "c", "young", "bud")}
    T.update(wood=tex(pdir, "tex_wood.png"), twig=tex(pdir, "tex_twig.png"))
    for k in ("metal", "ground", "grass", "fern", "weed", "litter", "hills", "net"):
        T[k] = tex(edir, f"tex_{k}.png")
    F, O, TR = get_material("M_Leaf2", "foliage"), get_material("M_Opaque2", "opaque"), get_material("M_Masked2", "masked")
    MI = {
        "leaf_a": instance(F, "MI_LeafHealthy", T["a"], Roughness=.28, Specular=.65, Translucency=.45),
        "leaf_b": instance(F, "MI_LeafYellowing", T["b"], Roughness=.34, Specular=.6, Translucency=.5),
        "leaf_c": instance(F, "MI_LeafSpotted", T["c"], Roughness=.32, Specular=.6, Translucency=.45),
        "young": instance(F, "MI_ShootYoung", T["young"], Roughness=.7, Specular=.35, Translucency=.6),
        "bud": instance(F, "MI_ShootBud", T["bud"], Roughness=.8, Specular=.3, Translucency=.55),
        "wood": instance(O, "MI_Wood", T["wood"], Roughness=.85),
        "twig": instance(O, "MI_Twig", T["twig"], Roughness=.8),
        "metal": instance(O, "MI_ArchMetal", T["metal"], Roughness=.42, Metallic=.7),
        "ground": instance(O, "MI_Ground", T["ground"], Roughness=.95, Specular=.3),
        "grass": instance(F, "MI_Grass", T["grass"], Roughness=.6, Specular=.4, Translucency=.5),
        "fern": instance(F, "MI_Fern", T["fern"], Roughness=.55, Specular=.45, Translucency=.5),
        "weeds": instance(F, "MI_Weed", T["weed"], Roughness=.55, Specular=.45, Translucency=.5),
        "litter": instance(O, "MI_Litter", T["litter"], Roughness=.9),
        "hills": instance(O, "MI_Hills", T["hills"], Roughness=.95, Specular=.2),
        "net": instance(get_material("M_NetTrans", "translucent"), "MI_ShadeNet2", color=W["net"]["color"], Opacity=W["net"]["opacity"], Roughness=1.0, Specular=0.0),
    }
    level = f"{GAME}/TeaGreenhouse"
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(level):
        les.load_level(level)
        for a in eas.get_all_level_actors():
            eas.destroy_actor(a)
    else:
        les.new_level(level)
    # sky, sun, fog, exposure
    sun = eas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 800), unreal.Rotator(0, -52, 125))
    lc = sun.light_component
    lc.set_editor_property("intensity", 10.0); lc.set_editor_property("atmosphere_sun_light", True)
    lc.set_editor_property("light_source_angle", 0.53)
    eas.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0))
    sky = eas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 500))
    sky.light_component.set_editor_property("real_time_capture", True); sky.light_component.set_editor_property("intensity", 1.0)
    fog = eas.spawn_actor_from_class(unreal.ExponentialHeightFog, unreal.Vector(0, 0, 0))
    fc = fog.component
    fc.set_editor_property("fog_density", 0.03); fc.set_editor_property("fog_height_falloff", 0.08)
    fc.set_editor_property("start_distance", 1500.0)
    fc.set_editor_property("fog_inscattering_luminance", unreal.LinearColor(.55, .66, .80, 1))
    eas.spawn_actor_from_class(unreal.VolumetricCloud, unreal.Vector(0, 0, 0))
    ppv = eas.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    ppv.set_editor_property("unbound", True)
    s = ppv.settings
    for k, v in (("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL), ("auto_exposure_apply_physical_camera_exposure", False),
                 ("auto_exposure_bias", float(os.environ.get("TEA_EV_BIAS", "1.6")))):
        s.set_editor_property("override_" + k, True); s.set_editor_property(k, v)
    ppv.set_editor_property("settings", s)
    # environment
    for n in ("ground", "hills", "arches", "rail"):
        mat = {"arches": "metal", "rail": "metal"}.get(n, n)
        place(import_file(os.path.join(edir, f"{n}.obj"), f"{GAME}/Meshes"), MI[mat], label=n.capitalize())
    place(import_file(os.path.join(edir, "net.obj"), f"{GAME}/Meshes"), MI["net"], label="ShadeNet", shadows=False)
    for n in ("grass", "weeds", "ferns", "litter"):
        place(import_file(os.path.join(edir, f"{n}.obj"), f"{GAME}/Meshes"), MI[{"ferns": "fern"}.get(n, n)], label="Cover_" + n)
    # tea bushes
    meta = json.load(open(os.path.join(pdir, "bushes.json")))
    parts = (("wood", "wood"), ("twig", "twig"), ("bud", "bud"), ("leaf_a", "leaf_a"), ("leaf_b", "leaf_b"), ("leaf_c", "leaf_c"),
             ("shoot_stem", "twig"), ("shoot_bud", "bud"), ("shoot_young", "young"))
    meshes = {(v, p): import_file(os.path.join(pdir, f"bush{v}_{p}.obj"), f"{GAME}/Meshes") for v in range(len(meta["bushes"])) for p, _ in parts}
    g = W["greenhouse"]
    placed = []
    x = g["x_min"] + .5
    while x < g["x_min"] + g["length"] - .3:
        for y in W["rows_y"]:
            placed.append((rnd.randrange(len(meta["bushes"])), [x + rnd.uniform(-.12, .12), y + rnd.uniform(-.12, .12), 0], rnd.uniform(.9, 1.08), rnd.uniform(0, 360)))
        x += W["bush_spacing"]
    n_inside = len(placed)
    o = W["outside"]
    while len(placed) < n_inside + o["count"]:
        r, a = rnd.uniform(o["inner"], o["outer"]), rnd.uniform(0, 2 * math.pi)
        px, py = r * math.cos(a) * 1.3, r * math.sin(a)
        if abs(px) < g["length"] / 2 + .8 and abs(py) < g["width"] / 2 + .8:
            continue
        placed.append((rnd.randrange(len(meta["bushes"])), [px, py, 0], rnd.uniform(.95, 1.5) * (2.2 if rnd.random() < .12 else 1), rnd.uniform(0, 360)))
    for i, (v, loc, sc, yaw) in enumerate(placed):
        for p, mat in parts:
            place(meshes[(v, p)], MI[mat], loc, sc, yaw, label=f"Bush{i:03d}_{p}")
    les.save_current_level()
    log(f"level saved: {n_inside} bushes inside, {len(placed) - n_inside} outside")
    # camera views: overview, inside the row, and close-ups of real shoot tips on the centre row
    centre = [b for b in placed[:n_inside] if abs(b[1][1]) < .3]
    v, loc, sc, yaw = centre[len(centre) // 2]
    tips = sorted(meta["bushes"][v]["tips"], key=lambda t: -t["pos"][2])
    tip = bush_tf(tips[0]["pos"], loc, sc, yaw)
    tip2 = bush_tf(tips[min(3, len(tips) - 1)]["pos"], loc, sc, yaw)
    views = {
        "overview": [[7.5, -6.5, 2.4], [0.0, 0.0, 1.3], 60],
        "inside_row": [[-4.3, -0.85, 1.65], [0.5, 0.2, 1.55], 72],
        "canopy": [[tip[0] + .55, tip[1] - .75, tip[2] + .30], tip, 55],
        "shoot_close": [[tip[0] + .16, tip[1] - .20, tip[2] + .03], [tip[0], tip[1], tip[2] + .03], 45],
        "shoot_close2": [[tip2[0] - .14, tip2[1] - .18, tip2[2] + .06], [tip2[0], tip2[1], tip2[2] + .02], 50],
    }
    json.dump(views, open(os.path.join(OUT, "ue_views.json"), "w"), indent=1)
    log("views written")


main()
