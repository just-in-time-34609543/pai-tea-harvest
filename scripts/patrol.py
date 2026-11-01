"""Continuous MuJoCo patrol. Multiple leaves per shared plant, one per trip.

Uses simulator ground truth for interlocks, not camera/AI observations.
The v5 single-leaf benchmark remains available separately.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from arm_kinematics import inverse, head_center

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/cage"


def numbers(x):
    return " ".join(str(float(v)) for v in x)


def targets(cfg):
    return [{"plant":p,"leaf_on_plant":j,"x_mm":x+dx}
            for p,x in enumerate(cfg["patrol"]["plant_centers_mm"])
            for j,dx in enumerate(cfg["patrol"]["leaf_offsets_mm"])]


def make_plant(cfg, index, offset, intrusion):
    """One shared root/trunk and two branch tips; all stay after harvest.

    Assumed test layout, not measured botanical morphology. SI units.
    """
    x=cfg["patrol"]["plant_centers_mm"][index]*.001+offset[0]
    body=ET.Element("body",name=f"plant{index}_protected_plant",pos=numbers([x,offset[1],0]))
    def geom(name,typ,**attrs):
        ET.SubElement(body,"geom",name=f"plant{index}_{name}",type=typ,
                      contype="4",conaffinity="10",**attrs)
    geom("parent_branch","capsule",fromto="0 0 .01 0 0 .05",size=".00175",rgba=".38 .25 .12 1")
    for j,dx in enumerate(cfg["patrol"]["leaf_offsets_mm"]):
        dx*=.001
        geom(f"branch_{j}","capsule",fromto=numbers([0,0,.05,dx,0,.069]),size=".0007",rgba=".38 .25 .12 1")
        geom(f"tip_{j}","capsule",fromto=numbers([dx,0,.069,dx,0,.0725]),size=".0005",rgba=".26 .42 .12 1")
    geom("old_leaf_a","ellipsoid",pos=".012 0 .039",size=".013 .005 .001",euler="0 -.25 0",rgba=".18 .38 .12 1")
    geom("old_leaf_b","ellipsoid",pos="-.014 0 .043",size=".0135 .005 .001",euler="0 .2 0",rgba=".18 .38 .12 1")
    geom("neighbor_branch","capsule",fromto="0 0 .03 0 .025 .055",size=".001",rgba=".38 .25 .12 1")
    geom("neighbor_bud","ellipsoid",pos="0 .025 .06",size=".002 .002 .006",rgba=".48 .65 .12 1")
    if intrusion and index==0:
        # Obstruct the second leaf of the FIRST plant, leaving its first reachable.
        g=body.find(f"geom[@name='plant{index}_old_leaf_a']")
        g.set("pos",numbers([cfg["patrol"]["leaf_offsets_mm"][1]*.001,.009,.0825]))
    return body


def descriptors(body):
    numeric={"size","pos","euler","fromto","rgba"}
    return [{k:([float(x) for x in v.split()] if k in numeric else v)
             for k,v in g.attrib.items() if k in numeric|{"name","type"}}
            for g in body.findall("geom")]


def make_scene(cfg, offset=(0,0), intrusion=False):
    scene=ET.parse(OUT/"scene.xml").getroot()
    scene.set("model","multi_leaf_plant_patrol_v7")
    world=scene.find("worldbody")
    plant=world.find("body[@name='protected_plant']")
    leaf=world.find("body[@name='shoot']")
    world.remove(plant); world.remove(leaf)
    eq=scene.find("equality"); eq.clear()
    names=[]
    for p in range(len(cfg["patrol"]["plant_centers_mm"])):
        world.append(make_plant(cfg,p,offset,intrusion))
    for i,target in enumerate(targets(cfg)):
        body=deepcopy(leaf)
        pos=np.fromstring(body.get("pos","0 0 0"),sep=" ")
        pos+=np.array([target["x_mm"]*.001+offset[0],offset[1],0])
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name",f"leaf{i}_{element.get('name')}")
        body.set("pos",numbers(pos)); world.append(body)
        names.append(body.get("name"))
        # Local fixtures support only uncut leaves, never attached to gripper.
        ET.SubElement(world,"body",name=f"fixture_{i}",pos=numbers(pos))
        ET.SubElement(eq,"weld",name=f"attachment_{i}",body1=body.get("name"),body2=f"fixture_{i}",solref=".004 1")
    delta=cfg["patrol"]["bin_x_mm"]*.001-.2*cfg["plant_scale"]
    for g in world.findall("geom"):
        if g.get("name","").startswith("bin_"):
            pos=np.fromstring(g.get("pos"),sep=" "); pos[0]+=delta
            g.set("pos",numbers(pos))
    scene.find(".//joint[@name='rail']").set("range",numbers(np.array(cfg["patrol"]["rail_limits_mm"])*.001))
    return scene,names


def smooth(x):
    x=np.clip(x,0,1)
    return float(x*x*(3-2*x))


def run(record=False, video=False, viewer=False, offset=(0,0), timestep=None, intrusion=False, hold_open=False):
    cfg=json.loads((ROOT/"config/cage.json").read_text())
    pc=cfg["patrol"]
    planned=targets(cfg)
    plant_names=[f"plant{i}_protected_plant" for i in range(len(pc["plant_centers_mm"]))]
    scene,names=make_scene(cfg,offset,intrusion)
    if record:
        ET.indent(scene)
        ET.ElementTree(scene).write(OUT/"scene-patrol.xml",encoding="unicode")
    scene.find("compiler").set("meshdir",str(OUT/"meshes"))
    m=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding="unicode"))
    if timestep: m.opt.timestep=timestep
    d=mujoco.MjData(m)
    arm=cfg["arm"]
    qp=inverse(arm,np.array(arm["pick_wrist_mm"])*.001)
    ql=inverse(arm,np.array(arm["pick_wrist_mm"])*.001+[0,0,cfg["greenhouse"]["lift_mm"]*.001])
    qt=np.radians(pc["travel_angles_deg"])
    aids=[m.actuator(n).id for n in ["shoulder","elbow"]]
    qids=[m.joint(n).qposadr[0] for n in ["shoulder","elbow"]]
    pa=[m.actuator(f"open_{i}").id for i in range(cfg["petals"])]
    ca=[m.actuator(f"blade_{i}").id for i in range(2)]
    ra=m.actuator("rail").id
    rqi=m.joint("rail").qposadr[0]
    opening=math.radians(cfg["open_angle_deg"])
    binx=pc["bin_x_mm"]*.001
    d.qpos[qids]=qt; d.ctrl[aids]=qt
    d.qpos[rqi]=binx; d.ctrl[ra]=binx
    bids=[m.body(n).id for n in names]
    cid=m.body("carrier").id
    leafsets=[set(np.flatnonzero(m.geom_bodyid==b)) for b in bids]
    for ids in leafsets:
        assert len(ids)==2 and sum(m.geom(i).name.endswith("target_leaf") for i in ids)==1
    protected={i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("plant")}
    structure=set(np.flatnonzero(m.geom_bodyid==m.body("greenhouse").id))
    tools={i for i in range(m.ngeom) if (m.geom(i).name or "").startswith(("p0_","p1_","p2_","p3_","p4_","p5_","cutter_","arm_"))}
    bodies=["carrier"]+[f"petal_{i}" for i in range(cfg["petals"])]+["blade_0","blade_1","arm_base","upper_arm","forearm","tool_mount","trolley"]+names+plant_names
    bodyids=[m.body(b).id for b in bodies]
    mujoco.mj_forward(m,d)
    # Boxes have the same dimensions as v5, translated along the rail.
    def in_bin(i):
        p=d.xpos[bids[i]]
        return bool(abs(p[0]-binx)<.035 and abs(p[1])<.0325 and .0145<p[2]<.05)
    def enclosed(i):
        rel=d.xmat[cid].reshape(3,3).T@(d.xpos[bids[i]]-d.xpos[cid])
        return bool(np.linalg.norm(rel[:2])<.014 and -.020<rel[2]<.015)
    stages=[("goto",3), ("unfold",1.2), ("open",1.2), ("descend",1.2),
            ("enclose",1), ("cut",.6), ("lift",1.2), ("fold",1.2),
            ("transfer",3), ("unfold_bin",1.2), ("lower_bin",1.2),
            ("release",2), ("collect",pc["bin_timeout_s"]), ("retract",1.2), ("fold_empty",1.2)]
    phase_map={"goto":"approach","open":"approach","unfold":"approach","descend":"approach","enclose":"enclose","cut":"cut","lift":"lift","fold":"lift","transfer":"transfer","unfold_bin":"transfer","lower_bin":"transfer","release":"release","collect":"collect","retract":"return","fold_empty":"return","home":"return","done":"done"}
    cycle=si=0; stage_start=0.; dwell=0.; rail_from=binx
    qcmd=qt.copy(); railcmd=binx; opencmd=0.
    cuts=[None]*len(names); collected=[]; events=[]; frames=[]
    def plant_counts():
        return [sum(planned[i]["plant"]==p for i in collected) for p in range(len(plant_names))]
    contact_pairs=set(); multiple=0; retained=[True]*len(names)
    max_error=0.; arm_limits_ok=True; rail_limits_ok=True
    stopped=bool(np.linalg.norm(offset)>.00125)
    reason="known_target_misaligned" if stopped else None
    ui=renderer=writer=None
    if viewer:
        from mujoco import viewer as mv
        ui=mv.launch_passive(m,d); ui.cam.lookat[:]=[.1,0,.07]; ui.cam.distance=.65
        ui.cam.azimuth=120; ui.cam.elevation=-18; ui.opt.geomgroup[3]=0
    if video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,height=720,width=960)
        camera=mujoco.MjvCamera(); camera.lookat[:]=[.1,0,.07]; camera.distance=.65
        camera.azimuth=125; camera.elevation=-20
        vo=mujoco.MjvOption(); vo.geomgroup[3]=0
        writer=imageio.get_writer(OUT/"patrol.mp4",fps=25,codec="libx264",quality=8)
    wall=time.monotonic(); next_frame=0.; done_at=None
    for _ in range(round((len(names)*35+5)/m.opt.timestep)):
        t=d.time
        stage,duration=stages[si] if cycle<len(names) else ("home",3)
        elapsed=t-stage_start; s=smooth(elapsed/duration)
        target=planned[min(cycle,len(names)-1)]["x_mm"]*.001
        if not stopped:
            if stage=="goto": qcmd=qt; railcmd=rail_from+(target-rail_from)*s; opencmd=0
            elif stage=="unfold": qcmd=qt+(ql-qt)*s; opencmd=0
            elif stage=="open": qcmd=ql; opencmd=opening*s
            elif stage=="descend": qcmd=ql+(qp-ql)*s
            elif stage=="enclose": qcmd=qp; opencmd=opening*(1-s)
            elif stage=="cut": qcmd=qp; opencmd=0
            elif stage in ["lift","retract"]: qcmd=qp+(ql-qp)*s
            elif stage in ["fold","fold_empty"]: qcmd=ql+(qt-ql)*s; opencmd=0
            elif stage=="transfer": qcmd=qt; railcmd=target+(binx-target)*s
            elif stage=="unfold_bin": qcmd=qt+(ql-qt)*s
            elif stage=="lower_bin": qcmd=ql+(qp-ql)*s
            elif stage=="release": qcmd=qp; opencmd=opening*s
            elif stage=="collect": opencmd=opening
            elif stage=="home": qcmd=qt; railcmd=binx; opencmd=0
        d.ctrl[aids]=qcmd; d.ctrl[ra]=railcmd
        d.ctrl[pa]=opening if hold_open else opencmd
        d.ctrl[ca]=.02*cfg["radius_mm"]/50*min(1,elapsed/.4) if stage=="cut" and not stopped else 0
        if stage=="cut" and elapsed>.45 and cuts[cycle] is None and not stopped:
            blades=[d.qpos[m.joint(f"cut_{i}").qposadr[0]] for i in range(2)]
            if min(blades)>.019*cfg["radius_mm"]/50:
                d.eq_active[cycle]=False; cuts[cycle]=float(t)
                events.append({"t":float(t),"event":"cut_one_leaf","leaf":cycle,"plant":planned[cycle]["plant"],"previous_collected":list(collected)})
        mujoco.mj_step(m,d)
        for c in d.contact[:d.ncon]:
            pair={int(c.geom1),int(c.geom2)}
            forbidden=bool(pair & protected or pair & structure)
            forbidden |= any(j!=cycle and j not in collected and pair & ids for j,ids in enumerate(leafsets))
            if pair & tools and forbidden:
                contact_pairs.add(tuple(sorted(pair))); stopped=True; reason="unexpected_contact"
        inside=[j for j in range(len(names)) if cuts[j] is not None and enclosed(j)]
        multiple=max(multiple,len(inside))
        if len(inside)>1: stopped=True; reason="multiple_leaves_in_cage"
        if cycle<len(names) and cuts[cycle] is not None and stage in ["lift","fold","transfer","unfold_bin","lower_bin"]:
            if not enclosed(cycle): retained[cycle]=False; stopped=True; reason="leaf_lost_in_transfer"
        expected=head_center(arm,qcmd)+[railcmd,0,0]
        max_error=max(max_error,float(np.linalg.norm(d.xpos[cid]-expected)))
        limits=np.radians([arm["shoulder_limits_deg"],arm["elbow_limits_deg"]])
        arm_limits_ok &= bool(np.all(d.qpos[qids]>=limits[:,0]-.002) and np.all(d.qpos[qids]<=limits[:,1]+.002))
        rail_limits_ok &= bool(pc["rail_limits_mm"][0]*.001-.001<=d.qpos[rqi]<=pc["rail_limits_mm"][1]*.001+.001)
        if stage=="collect" and not stopped:
            dwell=dwell+m.opt.timestep if in_bin(cycle) and not inside else 0
            if dwell>=pc["bin_dwell_s"]:
                collected.append(cycle); events.append({"t":float(d.time),"event":"collected_and_empty","leaf":cycle,"plant":planned[cycle]["plant"]})
                si+=1; stage_start=d.time; dwell=0
            elif elapsed>=duration:
                stopped=True; reason="collection_timeout"
        elif elapsed>=duration and not stopped:
            if stage=="cut" and cuts[cycle] is None: stopped=True; reason="cut_not_confirmed"
            elif stage=="home":
                done_at=float(t); break
            else:
                si+=1; stage_start=d.time
                if si==len(stages):
                    cycle+=1; si=0; rail_from=binx
        if d.time>=next_frame or stopped:
            if record:
                frames.append({"t":round(float(d.time),4),"phase":"stopped" if stopped else phase_map[stage],"stage":stage,
                               "target_index":min(cycle,len(names)-1),"collected":len(collected),"cut":cuts[min(cycle,len(names)-1)] is not None,
                               "collected_per_plant":plant_counts(),
                               "aborted":stopped,"rail_mm":float(d.qpos[rqi])*1000,"arm_deg":np.degrees(d.qpos[qids]).tolist(),
                               "poses":[[*np.round(d.xpos[b],7),*np.round(d.xquat[b],7)] for b in bodyids]})
            if renderer:
                renderer.update_scene(d,camera=camera,scene_option=vo); writer.append_data(renderer.render())
            if ui:
                if not ui.is_running(): stopped=True; reason="viewer_closed"
                ui.sync(); delay=wall+d.time-time.monotonic()
                if delay>0: time.sleep(delay)
            next_frame+=.04
        if stopped: break
    if ui: ui.close()
    if renderer: writer.close(); renderer.close()
    if done_at is None and not stopped:
        stopped=True; reason="patrol_timeout"
    warnings={str(i):int(w.number) for i,w in enumerate(d.warning) if w.number}
    finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
    positions=[d.xpos[b].tolist() for b in bids]
    rates=[float(np.max(np.abs(d.qvel[m.joint(f"leaf{i}_shoot_free").dofadr[0]:m.joint(f"leaf{i}_shoot_free").dofadr[0]+6]))) for i in range(len(names))]
    passed=bool(len(collected)==len(names) and done_at is not None and not stopped and all(in_bin(i) for i in collected) and not warnings and finite and arm_limits_ok and rail_limits_ok and max_error<.01)
    report={"revision":cfg["revision"],"mode":"patrol","passed":passed,"offset_mm":(np.array(offset)*1000).tolist(),"timestep_s":float(m.opt.timestep),
            "collected_count":len(collected),"target_count":len(names),"collected_per_plant":plant_counts(),"targets":planned,"cut_times_s":cuts,"events":events,"max_simultaneously_carried":multiple,
            "retained":retained,"aborted":stopped,"abort_reason":reason,"contact_pairs":[[m.geom(a).name,m.geom(b).name] for a,b in sorted(contact_pairs)],
            "max_head_tracking_error_mm":max_error*1000,"arm_limits_ok":arm_limits_ok,"rail_limits_ok":rail_limits_ok,"final_positions_m":positions,
            "final_leaf_max_velocity_components":rates,"final_leaf_strict_rest":[v<.01 for v in rates],
            "warnings":warnings,"finite":finite,"simulated_seconds":float(d.time),"wall_seconds":time.monotonic()-wall,
            "limits":"Known targets; 1s box residence and empty-cage check, not strict rest. Rigid leaves. No AI, tissue cutting or real sensors."}
    if record:
        if done_at is not None:
            frames.append({"t":round(float(d.time),4),"phase":"done","stage":"done","target_index":len(names)-1,
                           "collected":len(collected),"cut":True,"aborted":False,"rail_mm":float(d.qpos[rqi])*1000,
                           "collected_per_plant":plant_counts(),
                           "arm_deg":np.degrees(d.qpos[qids]).tolist(),"poses":[[*np.round(d.xpos[b],7),*np.round(d.xquat[b],7)] for b in bodyids]})
        (OUT/"trajectory-patrol.json").write_text(json.dumps({"mode":"patrol","bodies":bodies,"leaf_bodies":names,"targets":planned,"stations_mm":[t["x_mm"] for t in planned],"frames":frames},separators=(",",":")))
        (OUT/"patrol-report.json").write_text(json.dumps(report,indent=2))
        geometry=json.loads((OUT/"geometry.json").read_text()); geometry["config"]=cfg
        geometry["plant_geoms"]={n:descriptors(scene.find(f"worldbody/body[@name='{n}']")) for n in plant_names}
        for g in geometry["bin_geoms"]: g["pos"][0]+=binx-.2*cfg["plant_scale"]
        (OUT/"geometry-patrol.json").write_text(json.dumps(geometry,separators=(",",":")))
    return report
