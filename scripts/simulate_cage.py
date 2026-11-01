"""Execute actual MuJoCo dynamics and save body poses for a 3D replay.

No artificial attachment to the cage is used after the simulated cut.
The only released weld is the initial stem-to-world attachment.
"""
from pathlib import Path
import argparse
import json
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from arm_kinematics import inverse, head_center

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/cage"


def smooth(t, lo, hi):
    x = np.clip((t-lo)/(hi-lo), 0, 1)
    return float(x*x*(3-2*x))


def phase(t):
    for end, name in [(1,"approach"),(2,"enclose"),(2.6,"cut"),(3.6,"lift"),
                      (6,"transfer"),(8,"release"),(10,"collect")]:
        if t < end:
            return name
    return "collect"


def run(offset=(0,0), hold_open=False, record=False, viewer=False, video=False, timestep=None, intrusion=False):
    cfg = json.loads((ROOT / "config/cage.json").read_text())
    ps=cfg["plant_scale"]
    scene=ET.parse(OUT/"scene.xml").getroot()
    scene.find("compiler").set("meshdir",str(OUT/"meshes"))
    plant=scene.find(".//body[@name='protected_plant']")
    plant.set("pos",f"{offset[0]} {offset[1]} 0")
    if intrusion:
        plant.find("geom[@name='old_leaf_a']").set("pos",f"0 {.018*ps} {.165*ps}")
    # Static collision acceleration structures must be compiled AFTER placement.
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding="unicode"))
    if timestep is not None:
        model.opt.timestep = timestep
    data = mujoco.MjData(model)
    arm=cfg["arm"]
    pick_wrist=np.array(arm["pick_wrist_mm"])*.001
    start_angles=inverse(arm,pick_wrist)
    arm_indices=[model.joint(name).qposadr[0] for name in ["shoulder","elbow"]]
    rail_act=model.actuator("rail").id
    blade_act=[model.actuator(f"blade_{i}").id for i in range(2)]
    data.qpos[arm_indices]=start_angles
    data.ctrl[:2]=start_angles
    joint_indices = [model.joint(f"hinge_{i}").qposadr[0] for i in range(cfg["petals"])]
    opening = math.radians(cfg["open_angle_deg"])
    data.qpos[joint_indices] = opening
    free = model.joint("shoot_free").qposadr[0]
    data.qpos[free:free+2] += offset
    # Move the fixture target with the initial shoot perturbation.
    model.eq_data[0,3:5] -= offset
    data.ctrl[2:2+cfg["petals"]] = opening
    mujoco.mj_forward(model, data)
    bid = model.body("shoot").id
    target_geoms=[model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i]==bid]
    if sorted(target_geoms)!=["leaf_petiole","target_leaf"]:
        raise ValueError("Harvest body must contain exactly one leaf and its petiole")
    cid = model.body("carrier").id
    bodies = ["carrier"]+[f"petal_{i}" for i in range(cfg["petals"])]+["blade_0","blade_1","shoot","protected_plant","arm_base","upper_arm","forearm","tool_mount","trolley"]
    bodyids = [model.body(b).id for b in bodies]
    frames, measurements = [], []
    cut_time = None
    captured_at = []
    protected_ids=set(np.flatnonzero(model.geom_bodyid==model.body("protected_plant").id))
    tool_ids={i for i in range(model.ngeom) if (model.geom(i).name or "").startswith(("p0_","p1_","p2_","p3_","p4_","p5_","cutter_","arm_"))}
    contacts_seen=set()
    contact_steps=0
    structure_ids=set(np.flatnonzero(model.geom_bodyid==model.body("greenhouse").id))
    structure_contacts=set()
    structure_steps=0
    aborted = bool(np.linalg.norm(offset)>.0025*ps)
    abort_reason = "known_cut_point_outside_tolerance" if aborted else None
    scale=cfg["radius_mm"]/50
    tracking_errors=[]
    joint_history=[]
    ui = None
    if viewer:
        from mujoco import viewer as mjviewer
        ui = mjviewer.launch_passive(model, data)
        ui.cam.lookat[:] = [.1,0,.07]
        ui.cam.distance = .65
        ui.cam.azimuth = 120
        ui.cam.elevation = -18
        ui.opt.geomgroup[3] = 0
    renderer = writer = None
    if video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=720, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.1,0,.07]
        camera.distance = .65
        camera.azimuth = 125
        camera.elevation = -20
        scene_opt = mujoco.MjvOption()
        scene_opt.geomgroup[3] = 0
        writer = imageio.get_writer(OUT / "demo.mp4", fps=25, codec="libx264", quality=8)
    start = time.monotonic()
    next_frame = 0
    for _ in range(round(cfg["duration_s"]/model.opt.timestep)):
        t = data.time
        rail_target=0 if aborted else cfg["greenhouse"]["carriage_travel_mm"]*.001*smooth(t,3.6,5)
        data.ctrl[rail_act]=rail_target
        wrist=pick_wrist+np.array([0,0,cfg["greenhouse"]["lift_mm"]*.001*smooth(t,2.6,3.6)*(1-smooth(t,5,6))])
        target_angles=start_angles if aborted else inverse(arm,wrist)
        data.ctrl[:2]=target_angles
        close = 1-smooth(t,1,2)
        reopen = smooth(t,6,8)
        data.ctrl[2:2+cfg["petals"]] = opening*(1 if hold_open or aborted else max(close,reopen))
        data.ctrl[blade_act] = 0 if aborted else .02*scale*smooth(t,2,2.4)*(1-smooth(t,2.5,2.6))
        if cut_time is None and t>=2.4 and not aborted:
            actual_blades = [data.qpos[model.joint(f"cut_{i}").qposadr[0]] for i in range(2)]
            if min(actual_blades)>.019*scale:
                data.eq_active[model.equality("stem_attachment").id] = False
                cut_time = float(t)
        mujoco.mj_step(model,data)
        bad_pairs=[]
        frame_pairs=[]
        for c in data.contact[:data.ncon]:
            pair={int(c.geom1),int(c.geom2)}
            if pair & protected_ids and pair & tool_ids:
                bad_pairs.append(tuple(sorted(pair)))
            if pair & structure_ids and pair & tool_ids:
                frame_pairs.append(tuple(sorted(pair)))
        if frame_pairs:
            structure_steps+=1
            structure_contacts.update(frame_pairs)
            if cut_time is None:
                aborted=True
                abort_reason="greenhouse_contact"
        if bad_pairs:
            contact_steps+=1
            contacts_seen.update(bad_pairs)
            if cut_time is None:
                aborted=True
                abort_reason="protected_plant_contact"
        rel = data.xmat[cid].reshape(3,3).T@(data.xpos[bid]-data.xpos[cid])
        tracking_errors.append(float(np.linalg.norm(data.xpos[cid]-(head_center(arm,target_angles)+[rail_target,0,0]))))
        joint_history.append(data.qpos[arm_indices].copy())
        if 3<=t<=5.95:
            # An object is retained only when it travels inside the cage.
            captured_at.append(bool(np.linalg.norm(rel[:2])<.028*ps and -.040*ps<rel[2]<.030*ps))
        if data.time>=next_frame:
            if record:
                frames.append({"t": round(float(data.time),4), "phase": phase(data.time),
                               "cut": cut_time is not None,"aborted":aborted,"rail_mm":float(data.qpos[model.joint("rail").qposadr[0]])*1000,"arm_deg":np.degrees(data.qpos[arm_indices]).tolist(),
                               "poses": [[*np.round(data.xpos[b],7),*np.round(data.xquat[b],7)] for b in bodyids]})
                measurements.append({"t":round(float(data.time),4),"shoot":data.xpos[bid].tolist(),"carrier":data.xpos[cid].tolist(),"arm_target_deg":np.degrees(target_angles).tolist(),"arm_actual_deg":np.degrees(data.qpos[arm_indices]).tolist(),"arm_torque_Nm":data.actuator_force[:2].tolist()})
            if renderer:
                renderer.update_scene(data,camera=camera,scene_option=scene_opt)
                writer.append_data(renderer.render())
            if ui:
                if not ui.is_running():
                    break
                ui.sync()
                delay = start+data.time-time.monotonic()
                if delay>0:
                    time.sleep(delay)
            next_frame+=.04
    if ui:
        ui.close()
    if renderer:
        writer.close()
        renderer.close()
    final = data.xpos[bid].copy()
    final_speed = float(np.max(np.abs(data.qvel)))
    received = bool(.13*ps<final[0]<.27*ps and abs(final[1])<.065*ps and .03*ps<final[2]<.12*ps and final_speed<.01)
    warnings = {str(i):int(w.number) for i,w in enumerate(data.warning) if w.number}
    joint_history=np.array(joint_history)
    limits=np.radians([arm["shoulder_limits_deg"],arm["elbow_limits_deg"]])
    joint_limits_ok=bool(np.all(joint_history>=limits[:,0]-.002) and np.all(joint_history<=limits[:,1]+.002))
    result = {"offset_mm":[x*1000 for x in offset],"hold_open":hold_open,"cut_time_s":cut_time,
              "arm_joint_names":["shoulder","elbow"],"arm_joint_ranges_deg":np.degrees(np.array([joint_history.min(axis=0),joint_history.max(axis=0)]).T).tolist(),
              "arm_limits_ok":joint_limits_ok,"max_head_tracking_error_mm":max(tracking_errors)*1000,
              "intrusion":intrusion,"aborted":aborted,"abort_reason":abort_reason,
              "structure_contact_steps":structure_steps,"structure_contact_pairs":[[model.geom(a).name,model.geom(b).name] for a,b in sorted(structure_contacts)],
              "rail_final_mm":float(data.qpos[model.joint("rail").qposadr[0]])*1000,
              "protected_contact_steps":contact_steps,"protected_contact_pairs":[[model.geom(a).name,model.geom(b).name] for a,b in sorted(contacts_seen)],
              "released_body":"shoot" if cut_time is not None else None,
              "harvest_unit":"one_leaf_with_short_petiole","harvested_leaf_count":1 if cut_time is not None else 0,
              "harvested_bud_count":0,"target_geometry_names":target_geoms,
              "protected_plant_fixed":bool(model.body("protected_plant").jntnum==0),
              "retained_during_transfer": bool(cut_time is not None and captured_at and all(captured_at)),
              "received_in_bin":received,"final_shoot_position_m":final.tolist(),
              "warnings":warnings,"finite":bool(np.isfinite(data.qpos).all()),
              "final_max_abs_velocity":final_speed,"simulated_seconds":float(data.time),"timestep_s":float(model.opt.timestep),
              "wall_seconds":time.monotonic()-start}
    if record:
        (OUT / "trajectory.json").write_text(json.dumps({"bodies":bodies,"frames":frames},separators=(",",":")))
        (OUT / "measurements.json").write_text(json.dumps(measurements,indent=2))
        (OUT / "run-report.json").write_text(json.dumps(result,indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer",action="store_true",help="macOS: launch with .venv/bin/mjpython")
    parser.add_argument("--video",action="store_true")
    parser.add_argument("--test",action="store_true",help="Five predefined positions plus an open-cage negative control")
    parser.add_argument("--patrol",action="store_true",help="Finish each plant's known leaves, one per trip")
    args = parser.parse_args()
    if args.patrol:
        from patrol import run as run_patrol
        if args.test:
            cases=[run_patrol(),run_patrol(timestep=.0005),run_patrol(offset=(0,.001))]
            blocked=run_patrol(intrusion=True)
            open_case=run_patrol(hold_open=True)
            passed=(all(c["passed"] for c in cases) and blocked["aborted"]
                    and blocked["collected_per_plant"]==[1,0,0]
                    and all(t is None for t in blocked["cut_times_s"][1:])
                    and open_case["aborted"] and open_case["collected_count"]==0
                    and all(t is None for t in open_case["cut_times_s"][1:]))
            report={"passed":passed,"cases":cases,"intrusion_control":blocked,"open_control":open_case}
            (OUT/"patrol-test-report.json").write_text(json.dumps(report,indent=2))
        else:
            report=run_patrol(record=True,video=args.video,viewer=args.viewer)
        print(json.dumps(report,indent=2))
        raise SystemExit(0 if report["passed"] else 1)
    if args.test:
        cfg=json.loads((ROOT/"config/cage.json").read_text())
        ps=cfg["plant_scale"]
        offsets=[(x*ps,y*ps) for x,y in [(0,0),(.002,0),(-.002,0),(0,.002),(0,-.002)]]
        results=[run(p) for p in offsets]
        dt=json.loads((ROOT/"config/cage.json").read_text())["timestep_s"]
        refined=[run(p,timestep=dt/2) for p in offsets]
        control=run(hold_open=True)
        obstruction=run(intrusion=True)
        misaligned=run((.005*ps,0))
        success = all(x["cut_time_s"] is not None and x["retained_during_transfer"] and x["received_in_bin"] and x["finite"] and not x["warnings"] and not x["protected_contact_steps"] and not x["aborted"] and x["arm_limits_ok"] and x["max_head_tracking_error_mm"]<10 for x in results+refined)
        success &= control["cut_time_s"] is not None and control["finite"] and not control["warnings"] and not control["retained_during_transfer"] and not control["received_in_bin"]
        success &= obstruction["aborted"] and obstruction["cut_time_s"] is None and obstruction["protected_contact_steps"]>0 and obstruction["finite"] and not obstruction["warnings"]
        success &= misaligned["aborted"] and misaligned["cut_time_s"] is None and not misaligned["received_in_bin"]
        success &= all(not x["structure_contact_steps"] and abs(x["rail_final_mm"]-cfg["greenhouse"]["carriage_travel_mm"])<1 for x in results+refined)
        report={"revision":cfg["revision"],"passed":bool(success),"mujoco_version":mujoco.__version__,"cases":results,"half_timestep_cases":refined,"open_cage_control":control,"protected_leaf_control":obstruction,"misaligned_control":misaligned,
                "limits":"Known single leaf and petiole cut point; one harvest cycle. Rigid leaf, fixed protected neighbors. No AI selection, fracture, damage or deformation model."}
        (OUT / "test-report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        raise SystemExit(0 if success else 1)
    print(json.dumps(run(record=True,viewer=args.viewer,video=args.video),indent=2))


if __name__=="__main__":
    main()
