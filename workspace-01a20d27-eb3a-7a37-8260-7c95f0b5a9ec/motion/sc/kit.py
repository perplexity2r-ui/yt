"""Characters + room kit for the 3D short (built from engine primitives)."""
import math
import numpy as np
from engine import T, RX, RY, RZ, S, _compose, seg, ease

# palette
SKIN = (0.96, 0.72, 0.57)
HAIR = (0.13, 0.10, 0.13)
HOODIE = (0.13, 0.31, 0.22)
HOODIE_D = (0.09, 0.22, 0.16)
PANTS = (0.16, 0.15, 0.20)
SHOE = (0.78, 0.15, 0.16)
SOCK = (0.86, 0.84, 0.80)
ORANGE = (0.98, 0.42, 0.10)
WHITE = (0.88, 0.87, 0.84)
WOOD = (0.26, 0.155, 0.085)
WOOD_L = (0.26, 0.15, 0.09)
WALL = (0.15, 0.14, 0.17)
DARK = (0.07, 0.07, 0.09)
SCREEN = (0.55, 0.72, 0.95)
KEYCAP = (0.20, 0.20, 0.23)


def M(pos=(0, 0, 0), rot=(0, 0, 0), size=(1, 1, 1)):
    return _compose(dict(pos=pos, rot=rot, size=size))


class Kit:
    """Collects objects; helpers build common parts."""

    def __init__(self):
        self.objs = []

    def add(self, mesh, model=None, pos=None, rot=None, size=None, color=(0.8, 0.8, 0.8),
            metal=0.0, rough=0.5, emissive=0.0, tag=None, **kw):
        o = dict(mesh=mesh, color=color, metal=metal, rough=rough, emissive=emissive)
        if model is not None:
            o["model"] = model
        else:
            o["pos"] = pos or (0, 0, 0)
            o["rot"] = rot or (0, 0, 0)
            o["size"] = size or (1, 1, 1)
        if tag:
            o["tag"] = tag
        o.update(kw)
        self.objs.append(o)
        return o

    def part(self, base, mesh, local=(0, 0, 0), rot=(0, 0, 0), size=(1, 1, 1), **kw):
        """object whose model = base @ T(local) @ R @ S"""
        Mx = base @ M(local, rot, size)
        return self.add(mesh, model=Mx, **kw)

    # -------------------------------------------------------------- primitives
    def box(self, base, local, size, color, **kw):
        return self.part(base, "box", local, (0, 0, 0), size, color=color, **kw)

    def rbox(self, base, local, size, color, r=0.05, **kw):
        return self.part(base, "rbox", local, (0, 0, 0), size, color=color, **kw)

    def cyl(self, base, local, size, color, rot=(0, 0, 0), **kw):
        return self.part(base, "cyl", local, rot, size, color=color, **kw)

    def sph(self, base, local, size, color, **kw):
        return self.part(base, "sphere", local, (0, 0, 0), size, color=color, **kw)

    # -------------------------------------------------------------- characters
    def guy(self, base, t, mood="chill", pose="sit"):
        """low-poly coder. local +Y up, faces +Z. pose: sit|stand"""
        K = self
        if pose == "sit":
            # thighs forward, shins down
            for sx in (-1, 1):
                K.rbox(base, (sx * 0.095, 0.52, -0.06), (0.14, 0.15, 0.44), PANTS, r=0.03, rough=0.75)
                K.rbox(base, (sx * 0.095, 0.30, 0.16), (0.13, 0.40, 0.14), PANTS, r=0.03, rough=0.75)
                K.rbox(base, (sx * 0.095, 0.10, 0.22), (0.15, 0.10, 0.28), SHOE, r=0.035, rough=0.35)
                K.box(base, (sx * 0.095, 0.045, 0.24), (0.15, 0.035, 0.28), (0.92, 0.90, 0.86), rough=0.5)
            hipy = 0.60
        else:
            for sx in (-1, 1):
                K.rbox(base, (sx * 0.085, 0.34, 0), (0.13, 0.44, 0.15), PANTS, r=0.03, rough=0.75)
            for sx in (-1, 1):
                K.rbox(base, (sx * 0.085, 0.10, 0.03), (0.15, 0.10, 0.26), SHOE, r=0.035, rough=0.35)
                K.box(base, (sx * 0.085, 0.045, 0.03), (0.15, 0.035, 0.26), (0.92, 0.90, 0.86), rough=0.5)
            hipy = 0.30
        # torso
        ty = hipy + 0.26
        K.rbox(base, (0, ty, 0), (0.39, 0.44, 0.26), HOODIE, r=0.07, rough=0.75)
        K.rbox(base, (0, ty + 0.21, -0.02), (0.34, 0.12, 0.27), HOODIE_D, r=0.05, rough=0.8)
        K.rbox(base, (0, ty - 0.14, 0.135), (0.26, 0.13, 0.04), HOODIE_D, r=0.02, rough=0.8)
        # arms
        for sx in (-1, 1):
            sw = 0.0
            if mood == "shout":
                sw = 1.2 + 0.3 * math.sin(t * 6.0 + sx)
            piv = base @ T(sx * 0.225, ty + 0.19, 0.01) @ RZ(-sw * sx) @ RX(sx * -0.05)
            K.part(piv, "rbox", (0, -0.18, 0), (0, 0, 0), (0.105, 0.36, 0.115), color=HOODIE, rough=0.75)
            K.sph(piv, (0, -0.37, 0.015), (0.105, 0.105, 0.105), SKIN, rough=0.55)
        # neck & head
        neck = ty + 0.28
        K.cyl(base, (0, neck, 0), (0.10, 0.07, 0.10), SKIN, rough=0.6)
        hy_ = neck + 0.16
        K.rbox(base, (0, hy_, 0), (0.31, 0.31, 0.29), SKIN, r=0.07, rough=0.55)
        # hair
        K.rbox(base, (0, hy_ + 0.135, -0.01), (0.335, 0.14, 0.31), HAIR, r=0.06, rough=0.55)
        for i, sx in enumerate((-0.10, -0.034, 0.034, 0.10)):
            K.rbox(base, (sx, hy_ + 0.112 + 0.008 * (i % 2), 0.128), (0.075, 0.062, 0.05), HAIR,
                   r=0.02, rough=0.55)
        K.rbox(base, (0, hy_ + 0.01, -0.135), (0.31, 0.24, 0.06), HAIR, r=0.03, rough=0.6)
        # face
        ez = 0.152
        if mood == "shock":
            for sx in (-1, 1):
                K.sph(base, (sx * 0.07, hy_ + 0.02, ez), (0.075, 0.09, 0.03), (0.05, 0.05, 0.07), rough=0.3)
        else:
            for sx in (-1, 1):
                K.box(base, (sx * 0.072, hy_ + 0.012, ez), (0.068, 0.082, 0.022),
                      (0.97, 0.96, 0.94), rough=0.25)
                K.box(base, (sx * 0.076, hy_ + 0.008, ez + 0.008), (0.036, 0.048, 0.014),
                      (0.06, 0.05, 0.09), rough=0.2)
                K.box(base, (sx * 0.072, hy_ + 0.072, ez), (0.062, 0.014, 0.02), HAIR, rough=0.6)
        if mood == "smile":
            K.box(base, (0, hy_ - 0.045, ez), (0.08, 0.015, 0.02), (0.35, 0.12, 0.12), rough=0.5)
            # blush
            for sx in (-1, 1):
                K.box(base, (sx * 0.115, hy_ - 0.02, ez - 0.01), (0.05, 0.03, 0.01),
                      (0.80, 0.42, 0.38), rough=0.8)
        elif mood == "shock":
            K.sph(base, (0, hy_ - 0.055, ez), (0.05, 0.065, 0.02), (0.30, 0.10, 0.10), rough=0.5)
        else:
            K.box(base, (0, hy_ - 0.048, ez), (0.06, 0.012, 0.02), (0.30, 0.12, 0.12), rough=0.5)
        return base

    def bot(self, base, t, mood="idle"):
        """little orange box robot with antenna + screen eyes"""
        K = self
        bob = math.sin(t * 2.2) * 0.012
        b = base @ T(0, bob, 0)
        # feet
        for sx in (-1, 1):
            K.rbox(b, (sx * 0.075, 0.045, 0.02), (0.09, 0.09, 0.16), (0.62, 0.24, 0.10), r=0.03, rough=0.45)
        # body
        K.rbox(b, (0, 0.24, 0), (0.30, 0.30, 0.22), ORANGE, r=0.05, rough=0.45)
        K.rbox(b, (0, 0.24, 0.115), (0.20, 0.16, 0.02), (0.10, 0.06, 0.04), r=0.02, rough=0.6)
        # arms
        for sx in (-1, 1):
            ang = 0.0
            if mood == "panic":
                ang = -1.4 + 0.4 * math.sin(t * 11 + sx * 1.7)
            elif mood == "proud":
                ang = -1.1
            K.part(b, "rbox", (sx * 0.19, 0.28, 0.0), (0, 0, ang * sx), (0.07, 0.22, 0.07),
                   color=ORANGE, rough=0.4, metal=0.1)
        # head
        K.rbox(b, (0, 0.52, 0), (0.34, 0.26, 0.26), ORANGE, r=0.06, rough=0.4)
        K.rbox(b, (0, 0.52, 0.135), (0.26, 0.17, 0.02), (0.03, 0.04, 0.06), r=0.03, rough=0.2)
        # eyes on screen
        if mood == "proud":
            for sx in (-1, 1):
                K.box(b, (sx * 0.055, 0.535, 0.147), (0.045, 0.012, 0.01), (0.4, 1.0, 0.9), emissive=2.4)
        elif mood == "panic":
            for sx in (-1, 1):
                K.box(b, (sx * 0.055, 0.540, 0.147), (0.05, 0.05, 0.01), (1.0, 0.45, 0.3), emissive=2.4)
        else:
            for sx in (-1, 1):
                K.box(b, (sx * 0.055, 0.530, 0.147), (0.045, 0.038, 0.01), (0.45, 1.0, 0.92), emissive=2.2)
        # mouth
        K.box(b, (0, 0.480, 0.147), (0.06, 0.010, 0.01), (0.4, 1.0, 0.9), emissive=1.8)
        # antenna
        K.cyl(b, (0, 0.70, 0), (0.02, 0.10, 0.02), (0.4, 0.4, 0.45), rough=0.4)
        lit = 2.6 if mood != "panic" else 3.0 + math.sin(t * 12) * 1.5
        col = (1.0, 0.55, 0.25) if mood == "panic" else (1.0, 0.75, 0.35)
        K.sph(b, (0, 0.765, 0), (0.055, 0.055, 0.055), col, emissive=lit, rough=0.3)
        return b

    # -------------------------------------------------------------- desk set
    def desk_scene(self, t, renderer=None, texname="code"):
        """mini characters standing on a desk in front of a giant laptop screen"""
        K = self
        DESK_Y = 0.75
        # floor + wall + bokeh backdrop
        K.add("plane", pos=(0, 0, 0), rot=(-math.pi / 2, 0, 0), size=(10, 10),
              color=(0.055, 0.060, 0.075), rough=0.7)
        K.add("plane", pos=(0, 1.4, -2.60), size=(22.0, 12.0), color=(0.080, 0.075, 0.095), rough=0.9)

        rng = np.random.RandomState(11)
        palette = [(1.0, 0.72, 0.36), (0.45, 0.75, 1.0), (1.0, 0.52, 0.42), (0.60, 0.95, 0.95)]
        for i in range(46):
            x = -2.6 + rng.rand() * 5.2
            y = 0.55 + rng.rand() * 1.95
            z = -2.15 + rng.rand() * 1.0
            r = 0.022 + rng.rand() * 0.040
            c = palette[i % 4]
            K.add("sphere", pos=(x, y, z), size=(r, r, r), color=c,
                  emissive=1.4 + rng.rand() * 1.6, rough=0.4)
        # string lights
        for i in range(11):
            x = -2.3 + i * 0.46
            y = 2.42 - math.sin(i * 0.8) * 0.09
            K.add("sphere", pos=(x, y, -2.24), size=(0.045, 0.045, 0.045),
                  color=(1.0, 0.74, 0.40), emissive=2.4, rough=0.3)
        K.add("box", pos=(0, 2.44, -2.26), size=(5.2, 0.012, 0.012), color=(0.10, 0.09, 0.09), rough=0.8)
        # desk (extends toward camera so the top fills the foreground)
        K.add("rbox", pos=(0.30, DESK_Y, -0.40), size=(4.4, 0.075, 2.00), color=WOOD, r=0.015, rough=0.5)
        K.add("box", pos=(0.30, 0.37, -1.38), size=(4.4, 0.76, 0.06), color=(0.16, 0.11, 0.08), rough=0.75)
        for sx in (-1, 1):
            K.add("box", pos=(0.30 + sx * 2.10, 0.37, -0.40), size=(0.08, 0.75, 1.90),
                  color=(0.16, 0.11, 0.08), rough=0.75)
        # desk mat under the characters
        K.add("box", pos=(-0.05, DESK_Y + 0.041, -0.62), size=(1.75, 0.006, 0.75),
              color=(0.10, 0.12, 0.17), rough=0.9)
        # laptop base + keys
        K.add("rbox", pos=(0.35, DESK_Y + 0.028, -1.02), size=(1.62, 0.05, 0.62), color=(0.62, 0.64, 0.68),
              r=0.012, metal=0.75, rough=0.32)
        K.add("box", pos=(0.35, DESK_Y + 0.056, -1.00), size=(1.40, 0.02, 0.46),
              color=(0.11, 0.12, 0.15), rough=0.7)
        for r in range(3):
            for c in range(12):
                K.add("box", pos=(0.35 - 0.60 + c * 0.109, DESK_Y + 0.070, -1.14 + r * 0.070),
                      size=(0.085, 0.014, 0.055), color=(0.17, 0.17, 0.20), rough=0.55)
        K.add("box", pos=(0.35, DESK_Y + 0.070, -0.90), size=(0.46, 0.013, 0.13),
              color=(0.17, 0.17, 0.20), rough=0.55)
        # laptop lid with the rendered screen (big, tilted back, raised on a stand)
        K.add("rbox", pos=(0.35, DESK_Y + 0.145, -1.36), size=(0.30, 0.28, 0.22), color=(0.14, 0.14, 0.17),
              r=0.02, rough=0.5, metal=0.4)
        lid = M((0.35, DESK_Y + 0.28, -1.38), (0.0, 0.0, 0.0), (1, 1, 1)) @ RX(0.14)
        K.part(lid, "rbox", (0, 0.56, -0.014), (0, 0, 0), (1.95, 1.12, 0.04), color=(0.55, 0.57, 0.62),
               metal=0.75, rough=0.32)
        if renderer is not None and texname:
            import screen as _screen
            K.add("tex", model=lid @ M((0, 0.56, 0.012), (0, 0, 0), (1.78, 0.995, 1)),
                  texture=_screen.texture(renderer, texname), emissive=1.15, tag="screen")
        # mug, plant, headphones, notebook (foreground clutter)
        K.add("cyl", pos=(0.92, DESK_Y + 0.075, -0.18), size=(0.17, 0.21, 0.17), color=(0.78, 0.52, 0.32),
              rough=0.35)
        K.add("torus", pos=(1.02, DESK_Y + 0.09, -0.18), rot=(math.pi / 2, 0, 0),
              size=(0.10, 0.10, 0.45), color=(0.78, 0.52, 0.32), rough=0.35)
        K.add("cyl", pos=(1.42, DESK_Y + 0.10, -0.90), size=(0.26, 0.22, 0.26), color=(0.42, 0.26, 0.19),
              rough=0.7)
        for i in range(7):
            a = i * 0.9
            K.add("rbox", pos=(1.42 + 0.13 * math.cos(a), DESK_Y + 0.34 + 0.05 * (i % 3),
                               -0.90 + 0.13 * math.sin(a)),
                  rot=(0.45 * math.sin(a), a, 0.45 * math.cos(a)), size=(0.06, 0.34, 0.06),
                  color=(0.18, 0.42, 0.25), rough=0.7)
        K.add("torus", pos=(-1.35, DESK_Y + 0.085, 0.10), rot=(math.pi / 2, 0, 0.5),
              size=(0.22, 0.22, 0.38), color=(0.09, 0.09, 0.11), rough=0.45)
        for sx in (-1, 1):
            K.add("cyl", pos=(-1.35 + sx * 0.21, DESK_Y + 0.085, 0.10), rot=(0, 0, math.pi / 2),
                  size=(0.14, 0.07, 0.14), color=(0.09, 0.09, 0.11), rough=0.45)
        K.add("box", pos=(-1.22, DESK_Y + 0.072, -0.62), size=(0.55, 0.02, 0.40),
              color=(0.72, 0.68, 0.58), rough=0.85)
        K.add("box", pos=(-1.22, DESK_Y + 0.085, -0.62), size=(0.18, 0.012, 0.18),
              color=(0.20, 0.20, 0.24), rough=0.6)
        for i in range(3):
            K.add("box", pos=(1.72, DESK_Y + 0.09 + i * 0.055, -0.34), size=(0.34, 0.05, 0.24),
                  color=[(0.45, 0.22, 0.22), (0.24, 0.34, 0.48), (0.44, 0.42, 0.28)][i], rough=0.8)
        # foreground depth layer (close to camera, intentionally large)
        K.add("cyl", pos=(-0.92, DESK_Y + 0.105, 0.16), size=(0.26, 0.30, 0.26), color=(0.82, 0.55, 0.34),
              rough=0.35)
        K.add("torus", pos=(-0.76, DESK_Y + 0.12, 0.16), rot=(math.pi / 2, 0, 0),
              size=(0.14, 0.14, 0.55), color=(0.82, 0.55, 0.34), rough=0.35)
        K.add("box", pos=(0.86, DESK_Y + 0.052, 0.06), rot=(0, 0.30, 0), size=(0.66, 0.045, 0.46),
              color=(0.66, 0.62, 0.52), rough=0.85)
        K.add("box", pos=(0.90, DESK_Y + 0.072, 0.02), rot=(0, 0.30, 0), size=(0.12, 0.02, 0.24),
              color=(0.14, 0.14, 0.18), rough=0.6)
        return DESK_Y + 0.075

