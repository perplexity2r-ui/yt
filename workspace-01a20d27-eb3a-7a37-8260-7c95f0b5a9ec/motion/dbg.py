import numpy as np
from PIL import Image
from engine import *
import scene as S
r = Renderer(S.W, S.H)
for t in [0.6, 1.8, 3.6, 4.9, 6.6, 9.6]:
    sc = S.build(t)
    sc["camera"]["dof"]["aperture"] = 0.0
    sc["bloom"] = 0.0
    im = r.render(sc)
    im.save("out/dbg_%.2f.png" % t)
    a = np.asarray(im.convert("L")).astype(np.float32)
    print(t, "mean", a.mean().round(2), "objs", len(sc["objects"]))
