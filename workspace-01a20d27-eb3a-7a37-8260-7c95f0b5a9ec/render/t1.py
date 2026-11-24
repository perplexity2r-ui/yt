import time, numpy as np
from engine import *
r = Renderer(1080, 1920)
sc = dict(
  camera=dict(pos=(4.2, 3.4, 5.0), target=(0, 0.4, 0), fov=34, dof=dict(focus=7.5, aperture=0.0)),
  bg=(0.015,0.018,0.028),
  ambient=dict(top=(0.09,0.11,0.16), bot=(0.01,0.012,0.02), rim=(0.3,0.45,0.7)),
  objects=[
    dict(mesh="rbox", pos=(0,0,0), size=(1.6,0.2,1.6), color=(0.16,0.18,0.22), metal=0.6, rough=0.25),
    dict(mesh="rbox", pos=(-0.25,0.36,0), size=(0.75,0.06,0.5), rot=(0,0,0.35), color=(0.55,0.6,0.68), metal=0.85, rough=0.18),
    dict(mesh="cyl", pos=(0.45,0.42,0.15), size=(0.28,0.25,0.28), color=(0.9,0.45,0.2), metal=0.2, rough=0.3),
    dict(mesh="box", pos=(0.5,0.75,0.15), size=(0.14,0.14,0.14), color=(0.1,0.9,1.0), emissive=2.5),
    dict(mesh="torus", pos=(-0.6,0.85,-0.35), size=(0.6,0.6,0.6), rot=(1.2,0.4,0), color=(0.9,0.95,1.0), metal=1.0, rough=0.08),
  ], shadows=True, exposure=1.0, bloom=0.5)
t=time.time(); im = r.render(sc); print("render", time.time()-t); im.save("/home/user/render/t1.png")
t=time.time(); im = r.render(sc); print("render2", time.time()-t)
