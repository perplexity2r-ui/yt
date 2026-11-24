import numpy as np
from engine import *
r = Renderer(540, 960)
def test(objs, cam, name):
    sc = dict(camera=cam, bg=(0,0,0), ambient=dict(top=(1,1,1),bot=(1,1,1),rim=(0,0,0)),
              shadows=False, bloom=0.0, exposure=1.0, vignette=0.0, grain=0.0, objects=objs,
              lights=dict(list=[((0,1,0),(1,1,1)),((1,0,0),(1,1,1)),((0,0,1),(1,1,1))], key_int=1, fill_int=1, rim_int=1))
    im = r.render(sc); a=np.array(im.convert("L")); ys,xs = np.where(a>25)
    if len(xs)==0: print(name,"EMPTY"); return
    print(name, "bbox x[%d,%d] y[%d,%d] w=%d h=%d"%(xs.min(),xs.max(),ys.min(),ys.max(),xs.max()-xs.min(),ys.max()-ys.min()))
test([dict(mesh="box", pos=(0,0,0), size=(1,1,1), color=(1,1,1), rough=1.0)], dict(pos=(0,0,5), target=(0,0,0), fov=34), "box1 @5 fov34 (expect h~315)")
test([dict(mesh="box", pos=(0,0,0), size=(1,1,1), color=(1,1,1), rough=1.0)], dict(pos=(0,0,5), target=(0,0,0), fov=60), "box1 @5 fov60 (expect h~554)")
test([dict(mesh="box", pos=(1,0,0), size=(0.5,0.5,0.5), color=(1,1,1), rough=1.0)], dict(pos=(0,0,5), target=(0,0,0), fov=34), "box@x1")
