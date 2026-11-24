import numpy as np, math, re
from PIL import Image
from engine import *
import scene as S

WANT = [  # t, target content height in px (of 1920), target center y
    (0.90, 700, 1120),
    (2.20, 470, 1150),
    (3.70, 560, 1150),
    (5.10, 500, 1150),
    (6.90, 620, 1130),
    (8.30, 360, 1180),
    (9.70, 680, 1120),
]
FK = (1920 / 2) / math.tan(math.radians(31.0) / 2)   # focal in px

def measure(r, t):
    sc = S.build(t)
    sc["objects"] = [o for o in sc["objects"] if o["mesh"] != "plane"]
    sc["bg_glow"] = ((0, 0, 0), (0.5, 0.6), 0.0)
    sc["bloom"] = 0.0
    sc["camera"]["dof"]["aperture"] = 0.0
    im = r.render(sc)
    d = np.frombuffer(r.depth_tex.read(), dtype='f4').reshape(r.H, r.W)
    mask = d < 0.999
    ys, xs = np.where(mask)
    if len(ys) < 50: return None
    return (np.percentile(xs, 1), np.percentile(xs, 99),
            np.percentile(ys, 1), np.percentile(ys, 99))

def run():
    r = Renderer(int(S.W * 0.4), int(S.H * 0.4))
    keys = [list(k) for k in S.CAM_KEYS]
    for it in range(4):
        print("--- iter", it)
        newkeys = [list(k) for k in keys]
        for i, (t, wh, wc) in enumerate(WANT):
            b = measure(r, t)
            if b is None: print(t, "no content"); continue
            x0, x1, y0, y1 = [v / 0.4 for v in b]
            hh = y1 - y0; c = (y0 + y1) / 2
            # find nearest key
            ki = min(range(len(keys)), key=lambda j: abs(keys[j][0] - t))
            R, cy, ty, fov = keys[ki][1:]
            dist = math.hypot(R, cy - ty)
            # scale distance so height matches
            k = hh / wh if wh else 1.0
            k = max(0.6, min(1.7, k))
            R2 = R * k
            ty2 = ty + (wc - c) * dist / FK
            # width check: keep objects inside 84..996 px
            if x0 < 40 or x1 > S.W - 40:
                R2 *= 1.0 + max(0, (max(40 - x0, x1 - (S.W - 40)))) / S.W * 1.2
            print("  t=%.2f h=%4d c=%4d  x[%d,%d]  key%d R %.2f->%.2f ty %.2f->%.2f" %
                  (t, hh, c, x0, x1, ki, R, R2, ty, ty2))
            newkeys[ki][1] = round(R2, 2)
            newkeys[ki][3] = round(ty2, 3)
        keys = newkeys
    print("CAM_KEYS = [")
    for k in keys:
        print("    (%.2f, %.2f, %.2f, %.3f, %.1f)," % tuple(k))
    print("]")

run()
