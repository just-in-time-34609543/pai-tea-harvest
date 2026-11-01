"""Patrol topology, independent fixtures and safe sequence evidence."""
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from patrol import make_scene, targets

ROOT=Path(__file__).resolve().parents[1]


class PatrolTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/"config/cage.json").read_text())
        self.scene,self.names=make_scene(self.cfg)
        self.scene.find("compiler").set("meshdir",str(ROOT/"output/cage/meshes"))
        self.m=mujoco.MjModel.from_xml_string(ET.tostring(self.scene,encoding="unicode"))

    def test_six_independent_one_leaf_bodies(self):
        self.assertEqual(len(self.names),6)
        self.assertEqual(self.m.neq,6)
        for i,name in enumerate(self.names):
            body=self.m.body(name)
            geoms=[self.m.geom(j).name for j in range(self.m.ngeom) if self.m.geom_bodyid[j]==body.id]
            self.assertEqual(sorted(geoms),[f"leaf{i}_leaf_petiole",f"leaf{i}_target_leaf"])
            self.assertEqual(self.m.eq_obj1id[i],body.id)
            self.assertEqual(self.m.eq_obj2id[i],self.m.body(f"fixture_{i}").id)
            np.testing.assert_allclose(self.m.body_pos[body.id],self.m.body_pos[self.m.eq_obj2id[i]])

    def test_releasing_one_does_not_release_others(self):
        d=mujoco.MjData(self.m); d.eq_active[0]=False
        np.testing.assert_equal(d.eq_active,[False,True,True,True,True,True])

    def test_two_leaves_share_each_plant(self):
        plants=self.scene.findall("worldbody/body")
        self.assertEqual(sum(b.get("name","").startswith("plant") for b in plants),3)
        planned=targets(self.cfg)
        self.assertEqual([t["plant"] for t in planned],[0,0,1,1,2,2])
        for i,t in enumerate(planned):
            plant=self.scene.find(f"worldbody/body[@name='plant{t['plant']}_protected_plant']")
            tip=plant.find(f"geom[@name='plant{t['plant']}_tip_{t['leaf_on_plant']}']")
            end=np.fromstring(tip.get("fromto"),sep=" ")[3:]+np.fromstring(plant.get("pos"),sep=" ")
            leaf=self.m.body(self.names[i])
            # Capsule endpoint radii touch: support .5mm + petiole .5mm.
            np.testing.assert_allclose(end,self.m.body_pos[leaf.id]+[0,0,-.0075],atol=1e-9)

    def test_unharvested_leaves_remain_at_fixture(self):
        d=mujoco.MjData(self.m)
        d.qpos[self.m.joint("rail").qposadr[0]]=.23
        d.ctrl[self.m.actuator("rail").id]=.23
        for name,q in zip(["shoulder","elbow"],np.radians(self.cfg["patrol"]["travel_angles_deg"])):
            d.qpos[self.m.joint(name).qposadr[0]]=q; d.ctrl[self.m.actuator(name).id]=q
        for _ in range(1000): mujoco.mj_step(self.m,d)
        for name in self.names:
            b=self.m.body(name).id
            np.testing.assert_allclose(d.xpos[b],self.m.body_pos[b],atol=.0001)

    def test_one_at_a_time_recorded_sequence(self):
        report=json.loads((ROOT/"output/cage/patrol-report.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_simultaneously_carried"],1)
        finished=[]
        for event in report["events"]:
            if event["event"]=="cut_one_leaf":
                self.assertEqual(event["previous_collected"],finished)
                self.assertEqual(event["leaf"],len(finished))
            else:
                finished.append(event["leaf"])
        self.assertEqual(finished,list(range(6)))
        self.assertEqual(report["collected_per_plant"],[2,2,2])
        cuts=[e for e in report["events"] if e["event"]=="cut_one_leaf"]
        self.assertEqual([e["plant"] for e in cuts],[0,0,1,1,2,2])
        for event in cuts:
            self.assertTrue(all(i in event["previous_collected"] for i in range(event["plant"]*2)))


if __name__=="__main__": unittest.main()
