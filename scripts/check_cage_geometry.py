"""Independent mesh validity and sampled adjacent-petal/base interference checks."""
from pathlib import Path
import json
import math
import cadquery as cq
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/cage"

def main():
    cfg=json.loads((ROOT/"config/cage.json").read_text())
    meshes={}
    for p in sorted((OUT/"stl").glob("*.stl")):
        mesh=trimesh.load(p,force="mesh")
        meshes[p.name]={"watertight":bool(mesh.is_watertight),"positive_volume":bool(mesh.volume>0),"bounds_mm":mesh.bounds.tolist()}
    petal=cq.importers.importStep(str(OUT/"step/petal.step"))
    base=cq.importers.importStep(str(OUT/"step/base.step"))
    a=math.radians(cfg["top_angle_deg"])
    pivot=(cfg["radius_mm"]*math.sin(a),0,cfg["radius_mm"]*math.cos(a))
    end=(pivot[0],-1,pivot[2])
    checks=[]
    for angle in np.linspace(0,cfg["open_angle_deg"],6):
        moving=petal.rotate(pivot,end,float(angle))
        neighbour=moving.rotate((0,0,0),(0,0,1),360/cfg["petals"])
        volumes={"base_overlap_mm3":sum(s.Volume() for s in moving.intersect(base).solids().vals()),
                 "adjacent_overlap_mm3":sum(s.Volume() for s in moving.intersect(neighbour).solids().vals())}
        checks.append({"opening_deg":float(angle),**volumes})
    passed=all(x["watertight"] and x["positive_volume"] for x in meshes.values()) and all(x["base_overlap_mm3"]<.01 and x["adjacent_overlap_mm3"]<.01 for x in checks)
    report={"passed":passed,"meshes":meshes,"sampled_interference":checks,"scope":"6 synchronous opening positions. Does not validate cutter mounts, drive links, printing clearance, strength, or asynchronous motion."}
    (OUT/"geometry-report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if passed else 1)

if __name__=="__main__": main()
