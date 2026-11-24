import numpy as np, math
from engine import *
import scene as S
W,H = S.W, S.H
r = Renderer(W,H)
cam = S.camera(3.7)
eye = np.array(cam["pos"]); tgt = np.array(cam["target"])
view = look_at(eye, tgt); proj = perspective(cam["fov"], W/H, 0.05, 200)
for name, p in [("cube_top",(0,1.14,0)), ("cube_bot",(0,-0.03,0)), ("cube_center",(0,0.36,0))]:
    v = proj @ view @ np.array([p[0],p[1],p[2],1.0])
    ndc = v[:3]/v[3]
    px = (ndc[0]*0.5+0.5)*W; py = (1-(ndc[1]*0.5+0.5))*H
    print(name, "ndc", np.round(ndc,3), "px", int(px), int(py))
print("cam", np.round(eye,2), np.round(tgt,2), "dist", round(np.linalg.norm(eye-tgt),2))
sc = S.build(3.7); sc["camera"]["dof"]["aperture"]=0; sc["bloom"]=0
im = r.render(sc); im.save("out/proj.png")
a = np.asarray(im.convert("L")).astype(np.float32)
col = a[:, 480:600]
rows = np.where(col.max(axis=1) > 90)[0]
print("bright rows range", rows.min(), rows.max())
