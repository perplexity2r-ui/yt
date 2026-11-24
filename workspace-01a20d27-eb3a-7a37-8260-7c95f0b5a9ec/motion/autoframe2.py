import numpy as np, math, itertools
from engine import *
from engine import _compose
import scene as S

W, H = S.W, S.H

def analytic(t, dist, ty, pitch):
    sc = S.build(t)
    az = 0.52 - 0.035 * t
    tgt = np.array([0.0, ty, 0.0])
    eye = tgt + np.array([dist * math.cos(pitch) * math.sin(az), dist * math.sin(pitch),
                          dist * math.cos(pitch) * math.cos(az)])
    view = look_at(eye, tgt); proj = perspective(31.0, W / H, 0.05, 200)
    xs, ys = [], []
    for o in sc["objects"]:
        if o["mesh"] == "plane": continue
        M = _compose(o)
        s = o["size"]
        for sx, sy, sz in itertools.product((-1, 1), repeat=3):
            v = proj @ view @ (M @ np.array([sx*s[0]/2, sy*s[1]/2, sz*s[2]/2, 1.0]))
            if v[3] <= 0.05: return None
            n = v[:3] / v[3]
            xs.append((n[0]*0.5+0.5)*W); ys.append((1-(n[1]*0.5+0.5))*H)
    return min(xs), max(xs), min(ys), max(ys)

def solve(t, want_h, want_cy, max_w=980, verbose=True):
    best = (1e9, None)
    for dist in np.arange(1.8, 9.0, 0.1):
        for pitch in np.arange(0.16, 0.52, 0.03):
            for ty in np.arange(-0.6, 2.2, 0.06):
                b = analytic(t, dist, ty, pitch)
                if b is None: continue
                x0, x1, y0, y1 = b
                h = y1 - y0; c = (y0+y1)/2; w = x1 - x0
                sc = abs(h-want_h) + 1.4*abs(c-want_cy) + 2.0*max(0, w-max_w) + 1.5*max(0, 120-x0) + 1.5*max(0, x1-(W-120))
                if sc < best[0]: best = (sc, (dist, ty, pitch, b))
    d, ty_, p_, b = best[1]
    if verbose:
        print("t=%.2f dist=%.2f ty=%.2f pitch=%.3f -> h=%d c=%d x[%d,%d] score=%.0f" %
              (t, d, ty_, p_, b[3]-b[2], (b[2]+b[3])/2, b[0], b[1], best[0]))
    return d, ty_, p_

TARGETS = [
    (0.95, 760, 1110),   # hero
    (2.25, 450, 1170),   # machines
    (3.80, 600, 1140),   # haiku
    (5.15, 520, 1160),   # sonnet
    (7.00, 640, 1120),   # opus
    (8.40, 250, 1230),   # conclusion pad
    (9.75, 720, 1120),   # CTA
]
for t, h, c in TARGETS:
    solve(t, h, c)
