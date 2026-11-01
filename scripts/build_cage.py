"""Generate editable STEP, STL, and concavity-preserving MJCF from one config.

CAD uses millimetres; MuJoCo uses metres, radians and kilograms.
The mechanism is a geometry prototype: independent ideal petal actuators,
blunt cutter placeholders, no completed servo linkage or cutting mechanics.
"""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import cadquery as cq
import numpy as np
import trimesh
from arm_kinematics import inverse
from greenhouse import frame_segments

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "cage"


def v(values):
    return " ".join(f"{float(x):.9g}" for x in values)


def node(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(x) for k, x in attrs.items()})


def spherical(r, theta, phi=0):
    return np.array([r * math.sin(theta) * math.cos(phi),
                     r * math.sin(theta) * math.sin(phi), r * math.cos(theta)])


def build():
    cfg = json.loads((ROOT / "config/cage.json").read_text())
    for folder in [OUT, OUT / "stl", OUT / "step", OUT / "meshes"]:
        folder.mkdir(parents=True, exist_ok=True)
    scale = cfg["radius_mm"]/50
    r, thick = 50, cfg["wall_mm"]/scale
    count = cfg["petals"]
    a, b = np.radians([cfg["top_angle_deg"], cfg["bottom_angle_deg"]])
    span = 360 / count - cfg["slot_angle_deg"]
    pivot = spherical(r, a)
    def point(radius, angle):
        p = spherical(radius, angle)
        return (float(p[0]), float(p[2]))
    # Revolve an annular circular arc, preserving the open cavity.
    petal = (cq.Workplane("XZ").moveTo(*point(r + thick, a))
             .threePointArc(point(r + thick, (a+b)/2), point(r + thick, b))
             .lineTo(*point(r, b)).threePointArc(point(r, (a+b)/2), point(r, a))
             .close().revolve(span, (0, 0), (0, 1))
             .rotate((0, 0, 0), (0, 0, 1), -span/2))
    # Relieve the shoulders around the fixed clevis, retaining a central tongue.
    for sign in [-1,1]:
        relief = cq.Workplane("XY").box(200,100,100).translate((0,sign*53.5,pivot[2]+42))
        petal = petal.cut(relief)
    # Hinge knuckle and clevis use a tangential (Y) pin axis.
    plane = cq.Plane(origin=tuple(pivot), xDir=(1, 0, 0), normal=(0, 1, 0))
    knuckle = cq.Workplane(plane).circle(5).extrude(3.5, both=True)
    bore = cq.Workplane(plane).circle(cfg["hinge_bore_mm"]/scale/2).extrude(12, both=True)
    petal = petal.union(knuckle).cut(bore)
    base = cq.Workplane("XY").workplane(offset=pivot[2]+8).circle(pivot[0]+7).circle(10).extrude(6)
    bracket = None
    for sy in [-1, 1]:
        lug = cq.Workplane("XY").box(10, 3, 14).translate((pivot[0], sy*5.5, pivot[2]+4)).cut(bore)
        bracket = lug if bracket is None else bracket.union(lug)
    for i in range(count):
        base = base.union(bracket.rotate((0, 0, 0), (0, 0, 1), i*360/count))
    blade = cq.Workplane("XY").box(20, 6, 2)
    petal, base, blade = [cq.Workplane("XY").newObject([part.val().scale(scale)]) for part in [petal,base,blade]]
    pivot *= scale
    r, thick = r*scale, thick*scale
    assembly = cq.Assembly(name="cage_closed_geometry_prototype")
    assembly.add(base, name="base", color=cq.Color(0.15, 0.23, 0.29))
    parts = {"petal": petal, "base": base, "cutter_placeholder": blade}
    for name, part in parts.items():
        if not part.val().isValid() or len(part.solids().vals()) != 1:
            raise ValueError(f"Invalid or disconnected CAD part: {name}")
        cq.exporters.export(part, str(OUT / "step" / f"{name}.step"))
        cq.exporters.export(part, str(OUT / "stl" / f"{name}.stl"), tolerance=0.08, angularTolerance=0.08)
    for i in range(count):
        assembly.add(petal.rotate((0,0,0), (0,0,1), i*360/count), name=f"petal_{i+1}", color=cq.Color(.12,.65,.56))
    for i, x in enumerate([-30, 30]):
        assembly.add(blade.translate((x*scale, 0, -55*scale)), name=f"cutter_{i+1}", color=cq.Color(.9,.48,.2))
    assembly.save(str(OUT / "step" / "cage_assembly.step"))

    arm=cfg["arm"]
    q0=inverse(arm,np.array(arm["pick_wrist_mm"])*.001)
    link_parts=[]
    for i,length in enumerate(arm["links_mm"]):
        part=cq.Workplane("XZ").center(length/2,0).slot2D(length+9,9).extrude(3.5,both=True)
        holes=cq.Workplane("XZ").pushPoints([(0,0),(length,0)]).circle(2).extrude(5,both=True)
        part=part.cut(holes)
        link_parts.append(part)
        for folder,ext in [("step","step"),("stl","stl")]:
            cq.exporters.export(part,str(OUT/folder/f"arm_link_{i+1}.{ext}"))
    whole=cq.Assembly(name="two_joint_arm_concept")
    basepos=np.array(arm["base_mm"])
    elbow=basepos+np.array([arm["links_mm"][0]*math.cos(q0[0]),0,arm["links_mm"][0]*math.sin(q0[0])])
    for i,(part,pos,angle) in enumerate(zip(link_parts,[basepos,elbow],[q0[0],sum(q0)])):
        whole.add(part.rotate((0,0,0),(0,-1,0),math.degrees(angle)).translate(tuple(pos)),name=f"link_{i+1}",color=cq.Color(.35,.51,.75))
    headpos=np.array(arm["pick_wrist_mm"])-[0,0,arm["mount_drop_mm"]]
    whole.add(assembly,name="gripper",loc=cq.Location(cq.Vector(*headpos)))
    trolley_z=cfg["greenhouse"]["rail_height_mm"]-5
    trolley_part=cq.Workplane("XY").box(28,24,6).translate((basepos[0],0,trolley_z))
    whole.add(trolley_part,name="overhead_trolley",color=cq.Color(.23,.3,.4))
    hub=cq.Workplane("XZ").circle(7).extrude(6,both=True)
    for name,pos in [("shoulder_housing",basepos),("elbow_housing",elbow)]:
        whole.add(hub.translate(tuple(pos)),name=name,color=cq.Color(.88,.55,.22))
    mount_block=cq.Workplane("XY").box(8,8,4).translate(tuple(np.array(arm["pick_wrist_mm"])-[0,0,2]))
    whole.add(mount_block,name="fixed_tool_mount",color=cq.Color(.25,.32,.4))
    whole.save(str(OUT/"step/arm_assembly.step"))
    house=cq.Assembly(name="greenhouse_overhead_harvester")
    house.add(whole,name="hanging_arm")
    segments=frame_segments(cfg)
    for i,(a0,b0,r0) in enumerate(segments):
        direction=np.array(b0)-a0
        bar=cq.Solid.makeCylinder(r0,float(np.linalg.norm(direction)),cq.Vector(*a0),cq.Vector(*direction))
        house.add(bar,name=f"frame_{i}",color=cq.Color(.60,.68,.70))
    house.save(str(OUT/"step/greenhouse_assembly.step"))

    # Simulation petal mesh is referenced to its hinge, not sphere centre.
    cq.exporters.export(petal.translate(tuple(-pivot)), str(OUT / "meshes/petal.stl"), tolerance=.15, angularTolerance=.15)
    cq.exporters.export(base, str(OUT / "meshes/base.stl"), tolerance=.15, angularTolerance=.15)
    visual_mesh = trimesh.load(OUT / "meshes/petal.stl", force="mesh")
    base_mesh = trimesh.load(OUT / "meshes/base.stl", force="mesh")
    preview = {
        "config": cfg, "pivot_mm": pivot.tolist(),
        "petal": {"vertices": np.round(visual_mesh.vertices, 4).tolist(), "faces": visual_mesh.faces.tolist()},
        "base": {"vertices": np.round(base_mesh.vertices, 4).tolist(), "faces": base_mesh.faces.tolist()},
    }
    m = ET.Element("mujoco", model="tea_cage_v5_single_leaf")
    node(m, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    node(m, "option", timestep=cfg["timestep_s"], gravity="0 0 -9.81", integrator="implicitfast", iterations="80")
    visual = node(m, "visual")
    node(visual, "global", offwidth="1280", offheight="960")
    node(visual, "headlight", ambient=".5 .5 .5", diffuse=".7 .7 .7")
    default = node(m, "default")
    node(default, "geom", friction=".55 .005 .0001", solref=".008 1", solimp=".95 .99 .001", margin="0.0001")
    asset = node(m, "asset")
    node(asset, "mesh", name="petal_visual", file="petal.stl", scale=".001 .001 .001")
    node(asset, "mesh", name="base_visual", file="base.stl", scale=".001 .001 .001")
    # Each small patch has its own convex hull. A single shell hull would fill
    # the entire cavity and give invalid capture behaviour.
    ts = np.linspace(a, b, cfg["collision_rows"]+1)
    ps = np.linspace(-math.radians(span)/2, math.radians(span)/2, cfg["collision_columns"]+1)
    patches = []
    for j in range(len(ts)-1):
        for k in range(len(ps)-1):
            vertices = [(spherical(rad, t, p)-pivot)*.001
                        for rad in [r, r+thick] for t in ts[j:j+2] for p in ps[k:k+2]]
            center_mm=np.mean(vertices,axis=0)*1000+pivot
            if abs(center_mm[1])>3.5*scale and center_mm[2]>pivot[2]-8*scale:
                continue
            name = f"patch_{j}_{k}"
            node(asset, "mesh", name=name, vertex=v(np.array(vertices).reshape(-1)))
            patches.append(name)
    world = node(m, "worldbody")
    node(world, "light", pos=".1 -.3 .8", dir="0 .2 -1", diffuse=".8 .8 .8")
    node(world, "geom", name="floor", type="plane", size=".5 .5 .01", rgba=".88 .90 .9 1")
    node(world, "camera", name="overview", pos=".48 -.60 .43", xyaxes=".78 .625 0 -.20 .25 .94")
    greenhouse=node(world,"body",name="greenhouse")
    for i,(a0,b0,r0) in enumerate(segments):
        node(greenhouse,"geom",name=f"frame_{i}",type="capsule",fromto=v(np.r_[a0,b0]*.001),size=r0*.001,rgba=".6 .68 .7 1",contype="32",conaffinity="26")
    # Collection tray; inner floor z=30 mm, rim z=100 mm.
    node(world, "geom", name="bin_floor", type="box", pos=".2 0 .026", size=".07 .065 .004", rgba=".16 .27 .32 1")
    for sy in [-1,1]:
        node(world, "geom", name=f"bin_y_{sy}", type="box", pos=v([.2,sy*.068,.065]), size=".076 .003 .035", rgba=".3 .57 .65 .35")
    for sx in [-1,1]:
        node(world, "geom", name=f"bin_x_{sx}", type="box", pos=v([.2+sx*.073,0,.065]), size=".003 .065 .035", rgba=".3 .57 .65 .35")
    protected = node(world, "body", name="protected_plant")
    node(protected,"geom",name="parent_branch",type="capsule",fromto="0 0 .02 0 0 .115",size=".0035",rgba=".38 .25 .12 1",contype="4",conaffinity="10")
    node(protected,"geom",name="retained_twig",type="capsule",fromto="0 0 .115 0 0 .145",size=".001",rgba=".26 .42 .12 1",contype="4",conaffinity="10")
    for name,pos,sz,rot in [("old_leaf_a",".025 0 .109",".026 .010 .002","0 -.25 0"),("old_leaf_b","-.028 0 .119",".027 .010 .002","0 .2 0"),("old_leaf_c","0 .029 .101",".009 .028 .002",".2 0 0")]:
        node(protected,"geom",name=name,type="ellipsoid",pos=pos,size=sz,euler=rot,rgba=".12 .35 .19 1",contype="4",conaffinity="10")
    node(protected,"geom",name="neighbor_branch",type="capsule",fromto="0 0 .085 -.075 0 .138",size=".002",rgba=".38 .25 .12 1",contype="4",conaffinity="10")
    node(protected,"geom",name="neighbor_bud",type="ellipsoid",pos="-.075 0 .15",size=".004 .004 .012",rgba=".23 .52 .27 1",contype="4",conaffinity="10")
    lengths=np.array(arm["links_mm"])*.001
    trolley=node(world,"body",name="trolley",pos=v([basepos[0]*.001,0,trolley_z*.001]))
    node(trolley,"joint",name="rail",type="slide",axis="1 0 0",range=v([0,cfg["greenhouse"]["carriage_travel_mm"]*.001]),damping="2")
    node(trolley,"geom",name="trolley_block",type="box",size=".014 .012 .003",mass=".05",rgba=".23 .3 .4 1",contype="0",conaffinity="0")
    arm_base=node(trolley,"body",name="arm_base",pos=v([0,0,(basepos[2]-trolley_z)*.001]))
    upper=node(arm_base,"body",name="upper_arm",euler=v([0,-q0[0],0]),gravcomp="1")
    forearm=node(upper,"body",name="forearm",pos=v([lengths[0],0,0]),euler=v([0,-q0[1],0]),gravcomp="1")
    for i,(body,name,key) in enumerate(zip([upper,forearm],["shoulder","elbow"],["shoulder_limits_deg","elbow_limits_deg"])):
        node(body,"joint",name=name,type="hinge",axis="0 -1 0",ref=q0[i],range=v(np.radians(arm[key])),damping=".12",armature=".002")
        node(body,"geom",name=f"arm_link_{i}",type="capsule",fromto=v([0,0,0,lengths[i],0,0]),size=".0045",mass=".015",rgba=".32 .5 .76 1",contype="16",conaffinity="5")
        node(body,"geom",name=f"arm_joint_{i}",type="cylinder",size=".007 .006",euler=v([math.pi/2,0,0]),mass=".012",rgba=".88 .55 .22 1",contype="16",conaffinity="5")
    mount=node(forearm,"body",name="tool_mount",pos=v([lengths[1],0,0]),euler=v([0,sum(q0),0]))
    node(mount,"geom",name="arm_mount",type="box",pos="0 0 -.002",size=".004 .004 .002",mass=".003",rgba=".25 .32 .4 1",contype="16",conaffinity="5")
    carrier = node(mount, "body", name="carrier", pos=v([0,0,-arm["mount_drop_mm"]*.001]), gravcomp="1")
    node(carrier, "inertial", pos="0 0 .015", mass=".02", diaginertia=".000002 .000002 .000002")
    node(carrier, "geom", name="base", type="mesh", mesh="base_visual", contype="0", conaffinity="0", rgba=".15 .23 .29 1")
    for i in range(count):
        phi = i * 2*math.pi/count
        p = spherical(r, a, phi)*.001
        body = node(carrier, "body", name=f"petal_{i}", pos=v(p), euler=v([0,0,phi]), gravcomp="1")
        node(body, "joint", name=f"hinge_{i}", type="hinge", axis="0 -1 0", range=v([0,math.radians(cfg["open_angle_deg"])]), damping=".015", armature=".0001")
        solid = petal.val()
        mass = solid.Volume()*1e-9*cfg["density_kg_m3"]
        center = (np.array(solid.Center().toTuple())-pivot)*.001
        inertia = np.array(cq.Shape.matrixOfInertia(solid))*1e-15*cfg["density_kg_m3"]
        node(body, "inertial", pos=v(center), mass=mass, fullinertia=v([inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]))
        node(body, "geom", name=f"petal_surface_{i}", type="mesh", mesh="petal_visual", contype="0", conaffinity="0", group="1", rgba=".10 .65 .54 .32")
        for patch in patches:
            node(body, "geom", name=f"p{i}_{patch}", type="mesh", mesh=patch, contype="2", conaffinity="1", group="3", rgba=".1 .8 .5 .1", mass="0")
    for i, sign in enumerate([-1,1]):
        body = node(carrier, "body", name=f"blade_{i}", pos=v([sign*.03*scale,0,-.055*scale]), gravcomp="1")
        node(body, "joint", name=f"cut_{i}", type="slide", axis=v([-sign,0,0]), range=v([0,.02*scale]), damping=".2")
        node(body, "geom",name=f"cutter_{i}",type="box", size=v(np.array([.01,.003,.001])*scale), mass=".00065", rgba=".93 .5 .17 1", contype="8", conaffinity="4")
    shoot = node(world, "body", name="shoot", pos="0 0 .16")
    node(shoot, "freejoint", name="shoot_free")
    # One blade and its short petiole only. Never release a bud/two-leaf cluster.
    node(shoot, "geom", name="leaf_petiole", type="capsule", fromto="0 0 -.013 0 0 .002", size=".001", mass=".00004", rgba=".55 .7 .12 1")
    node(shoot, "geom", name="target_leaf", type="ellipsoid", pos="0 0 .018", size=".006 .002 .018", mass=".00024", condim="6", friction=".55 .0001 .00002", rgba=".78 .9 .20 1")
    node(protected,"geom",name="retained_leaf_petiole",type="capsule",fromto="0 0 .115 .016 -.006 .125",size=".001",rgba=".26 .42 .12 1",contype="4",conaffinity="10")
    node(protected,"geom",name="retained_young_leaf",type="ellipsoid",pos=".022 -.006 .126",size=".011 .0045 .0015",euler="0 -.3 0",rgba=".24 .55 .22 1",contype="4",conaffinity="10")
    ps=cfg["plant_scale"]
    # Only scale the plant and tray; the cage and arm already use current CAD dimensions.
    scaled=[shoot]+list(shoot.findall("geom"))+list(protected.findall("geom"))+[g for g in world.findall("geom") if g.get("name","").startswith("bin_")]
    for element in scaled:
        for key in ["pos","size","fromto"]:
            if key in element.attrib:
                element.set(key,v(np.fromstring(element.get(key),sep=" ")*ps))
        if "mass" in element.attrib:
            element.set("mass",str(float(element.get("mass"))*ps**3))
    # Use the same primitive descriptions for the browser and the dynamics.
    def descriptors(body):
        return [{k: ([float(x) for x in value.split()] if k in ["size","pos","euler","fromto","rgba"] else value) for k,value in g.attrib.items() if k in ["name","type","size","pos","euler","fromto","rgba"]} for g in body.findall("geom")]
    preview.update({"greenhouse_geoms":descriptors(greenhouse),"bin_geoms":[d for d in descriptors(world) if d["name"].startswith("bin_")],"protected_geoms":descriptors(protected),"shoot_geoms":descriptors(shoot),"arm_geoms":{name:descriptors(body) for name,body in [("trolley",trolley),("arm_base",arm_base),("upper_arm",upper),("forearm",forearm),("tool_mount",mount)]},"shoot_origin_m":[0,0,.16*ps],"cut_point_m":[0,0,.147*ps],"carrier_origin_m":(headpos*.001).tolist()})
    (OUT / "geometry.json").write_text(json.dumps(preview,separators=(",",":")))
    equality = node(m, "equality")
    node(equality, "weld", name="stem_attachment", body1="shoot", solref=".004 1")
    contact=node(m,"contact")
    # Intentional pivot/mount overlap is not an obstacle to its own joint.
    node(contact,"exclude",body1="arm_base",body2="upper_arm")
    node(contact,"exclude",body1="upper_arm",body2="forearm")
    actuators = node(m, "actuator")
    for name in ["shoulder", "elbow"]:
        node(actuators, "position", name=name, joint=name, kp="120", kv="2", forcerange="-6 6")
    for i in range(count):
        node(actuators, "position", name=f"open_{i}", joint=f"hinge_{i}", kp="2", kv=".07", forcerange="-.2 .2")
    for i in range(2):
        node(actuators, "position", name=f"blade_{i}", joint=f"cut_{i}", kp="80", kv="1", forcerange="-3 3")
    node(actuators,"position",name="rail",joint="rail",kp="800",kv="15",forcerange="-10 10")
    ET.indent(m)
    ET.ElementTree(m).write(OUT / "scene.xml", encoding="unicode")
    report = {"config": cfg, "parts": {name: {"valid": part.val().isValid(), "solids": len(part.solids().vals()), "volume_mm3": part.val().Volume()} for name,part in parts.items()},
              "petal_mass_g": mass*1000, "shell_outer_diameter_mm": 2*(r+thick),
              "bottom_opening_mm": 2*r*math.sin(b), "equator_slot_mm": r*math.radians(cfg["slot_angle_deg"]),
              "collision_patches_per_petal": len(patches), "status": "geometry prototype; drive linkage and cutter mount not engineered"}
    (OUT / "build-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    build()
