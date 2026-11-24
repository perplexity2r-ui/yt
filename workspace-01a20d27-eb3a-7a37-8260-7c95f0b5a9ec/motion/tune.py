import numpy as np, math, itertools
from engine import *
from engine import _compose
import scene as S

def bbox_at(t, R, cy, ty, fov):
    # rebuild camera
    objs_fn = S.build
    sc = objs_fn(t)
    az = 0.52 - 0.035*t
    tgt = np.array([0.0, ty, 0.0])
    eye = tgt + np.array([R*math.sin(az), cy, R*math.cos(az)])
    view = look_at(eye, tgt); proj = perspective(fov, S.W/S.H, 0.05, 200)
    xs, ys = [], []
    for o in sc["objects"]:
        if o["mesh"] == "plane": continue
        M = _compose(o) if True else None
        s = o["size"]
        # 8 corners of the local box
        for sx, sy, sz in itertools.product((-1,1),repeat=3):
            p = M @ np.array([sx*s[0]/2, sy*s[1]/2, sz*s[2]/2, 1.0])
            v = proj @ view @ p
            if v[3] <= 0.01: continue
            ndc = v[:3]/v[3]
            xs.append((ndc[0]*0.5+0.5)*S.W); ys.append((1-(ndc[1]*0.5+0.5))*S.H)
    if not xs: return None
    return min(xs), max(xs), min(ys), max(ys)

def score(t, R, cy, ty, fov, want_cy=1150, want_h=620):
    b = bbox_at(t, R, cy, ty, fov)
    if b is None: return 1e9, None
    x0,x1,y0,y1 = b
    h = y1-y0; c = (y0+y1)/2
    w = x1-x0
    # want: height ~620, center ~1150, width >= 0.72*W
    s = abs(h-want_h)*1.0 + abs(c-want_cy)*1.2 + max(0, 0.72*S.W - w)*1.5
    return s, b

for t, want_h in [(0.9,660),(2.2,520),(3.7,560),(5.0,560),(6.9,620),(8.3,420),(9.7,700)]:
    best=(1e9,None,None)
    for R in np.arange(2.8, 7.0, 0.1):
        for ty in np.arange(-0.2, 1.6, 0.05):
            s,b = score(t, R, 2.0, ty, 31.0, want_h=want_h)
            if s < best[0]: best=(s,R,ty,b)
    print("t=%.1f  R=%.2f ty=%.2f  bbox=%s score=%.0f" % (t, best[1], best[2], np.round(best[3]).astype(int), best[0]))
