"""Check 2R kinematics against MuJoCo's actual fixed-wrist body tree."""
from pathlib import Path
import json
import unittest
import mujoco
import numpy as np
from arm_kinematics import inverse, forward, head_center
from greenhouse import frame_segments

ROOT=Path(__file__).resolve().parents[1]


class ArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm=json.loads((ROOT/"config/cage.json").read_text())["arm"]
        cls.model=mujoco.MjModel.from_xml_path(str(ROOT/"output/cage/scene.xml"))

    def test_forward_inverse(self):
        for x in np.linspace(-.005,.005,11):
            for z in [.11,.125,.14]:
                wrist=np.array([x,0,z])
                np.testing.assert_allclose(forward(self.arm,inverse(self.arm,wrist)),wrist,atol=1e-10)

    def test_mujoco_matches_two_joint_math(self):
        data=mujoco.MjData(self.model)
        for wrist in [[0,0,.11],[0,0,.14],[.005,0,.125],[-.005,0,.11]]:
            q=inverse(self.arm,wrist)
            for name,value in zip(["shoulder","elbow"],q):
                data.qpos[self.model.joint(name).qposadr[0]]=value
            mujoco.mj_forward(self.model,data)
            np.testing.assert_allclose(data.xpos[self.model.body("carrier").id],head_center(self.arm,q),atol=1e-8)

    def test_no_hidden_arm_joint(self):
        names=[self.model.joint(i).name for i in range(self.model.njnt)]
        arm_names=[n for n in names if not n.startswith(("hinge_","cut_")) and n not in ["shoot_free","rail"]]
        self.assertEqual(arm_names,["shoulder","elbow"])
        for n in arm_names:
            self.assertEqual(self.model.joint(n).type[0],mujoco.mjtJoint.mjJNT_HINGE)
        self.assertEqual(self.model.body("tool_mount").jntnum,0)
        self.assertEqual(self.model.body("carrier").jntnum,0)

    def test_overhead_rail_moves_whole_arm(self):
        data=mujoco.MjData(self.model)
        q=inverse(self.arm,np.array(self.arm["pick_wrist_mm"])*.001)
        for name,value in zip(["shoulder","elbow"],q):
            data.qpos[self.model.joint(name).qposadr[0]]=value
        data.qpos[self.model.joint("rail").qposadr[0]]=.05
        mujoco.mj_forward(self.model,data)
        np.testing.assert_allclose(data.xpos[self.model.body("carrier").id],head_center(self.arm,q)+[.05,0,0],atol=1e-8)
        self.assertEqual(self.model.joint("rail").type[0],mujoco.mjtJoint.mjJNT_SLIDE)
        self.assertFalse(any(self.model.geom(i).name=="arm_pedestal" for i in range(self.model.ngeom)))

    def test_unreachable_targets_rejected(self):
        for target in [[2,0,.11],[0,.01,.11],np.array(self.arm["base_mm"])*.001]:
            with self.assertRaises(ValueError):
                inverse(self.arm,target)

    def test_exactly_one_leaf_is_releasable(self):
        m=self.model
        target=m.body("shoot").id
        names={m.geom(i).name for i in range(m.ngeom) if m.geom_bodyid[i]==target}
        self.assertEqual(names,{"target_leaf","leaf_petiole"})
        self.assertEqual(m.neq,1)
        self.assertEqual(m.eq_obj1id[0],target)
        for name in ["retained_young_leaf","neighbor_bud","parent_branch"]:
            body=m.geom(name).bodyid[0]
            self.assertEqual(m.body(body).name,"protected_plant")
            self.assertEqual(m.body(body).jntnum,0)

    def test_greenhouse_outer_dimensions(self):
        cfg=json.loads((ROOT/"config/cage.json").read_text())
        g=cfg["greenhouse"]
        for a,b,r in frame_segments(cfg):
            lower=np.minimum(a,b)-r
            upper=np.maximum(a,b)+r
            self.assertTrue(np.all(lower >= [g["x_min_mm"],-g["width_mm"]/2,0]))
            self.assertTrue(np.all(upper <= [g["x_min_mm"]+g["length_mm"],g["width_mm"]/2,g["height_mm"]]))


if __name__=="__main__":
    unittest.main()
