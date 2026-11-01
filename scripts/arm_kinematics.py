"""Two actuated pitch joints in the XZ plane. No hidden wrist actuator."""
import math
import numpy as np


def inverse(arm, wrist_m):
    p=np.asarray(wrist_m)-np.asarray(arm["base_mm"])*.001
    if abs(p[1])>1e-9:
        raise ValueError("Two pitch joints cannot reach outside their XZ plane")
    l1,l2=np.asarray(arm["links_mm"])*.001
    c=(p[0]**2+p[2]**2-l1*l1-l2*l2)/(2*l1*l2)
    if not -1<=c<=1:
        raise ValueError("Wrist target outside two-link workspace")
    q2=arm.get("elbow_branch",-1)*math.acos(c)
    q1=math.atan2(p[2],p[0])-math.atan2(l2*math.sin(q2),l1+l2*math.cos(q2))
    q=np.array([q1,q2])
    for angle,key in zip(q,["shoulder_limits_deg","elbow_limits_deg"]):
        lo,hi=np.radians(arm[key])
        if not lo<=angle<=hi:
            raise ValueError(f"Target violates {key}")
    return q


def forward(arm, q):
    l1,l2=np.asarray(arm["links_mm"])*.001
    return np.asarray(arm["base_mm"])*.001+np.array([
        l1*math.cos(q[0])+l2*math.cos(sum(q)),0,
        l1*math.sin(q[0])+l2*math.sin(sum(q))])


def head_center(arm,q):
    q0=inverse(arm,np.asarray(arm["pick_wrist_mm"])*.001)
    tilt=sum(q)-sum(q0)
    h=arm["mount_drop_mm"]*.001
    return forward(arm,q)+np.array([h*math.sin(tilt),0,-h*math.cos(tilt)])
