"""Shared greenhouse centre-lines in mm; nominal outer envelope 500x250x150."""
import math
import numpy as np


def frame_segments(cfg):
    g=cfg["greenhouse"]
    radius=1.25
    lo=g["x_min_mm"]+radius
    hi=g["x_min_mm"]+g["length_mm"]-radius
    half=g["width_mm"]/2-radius
    eave=g["eave_mm"]
    rise=g["height_mm"]-radius-eave
    segments=[]
    for x in np.linspace(lo,hi,5):
        points=[[x,half*math.cos(t),eave+rise*math.sin(t)] for t in np.linspace(0,math.pi,25)]
        points=[[x,half,radius]]+points+[[x,-half,radius]]
        segments.extend((a,b,radius) for a,b in zip(points,points[1:]))
    for y,z in [(half,eave),(-half,eave),(0,g["height_mm"]-radius)]:
        segments.append(([lo,y,z],[hi,y,z],radius))
    # Two longitudinal tracks; travelling trolley is below these rails.
    for y in [-10,10]:
        segments.append(([lo+.25,y,g["rail_height_mm"]],[hi-.25,y,g["rail_height_mm"]],1.5))
    # Short hangers connect tracks to each roof arch, rather than floating rails.
    roof_at_track=eave+rise*math.sqrt(1-(10/half)**2)
    for x in np.linspace(lo,hi,5):
        for y in [-10,10]:
            segments.append(([x,y,g["rail_height_mm"]],[x,y,roof_at_track],.8))
    return segments
