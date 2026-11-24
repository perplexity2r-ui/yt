"""Main short: "Claude, where did my tokens go?" — 3D motion graphics, 1080x1920, 24 fps.

Timeline (10 s):
  0.0 - 1.2   hero cube
  1.0 - 2.9   seven machines
  2.8 - 4.4   haiku  (tiny task)
  4.1 - 5.9   sonnet (medium task)
  5.5 - 7.6   opus   (huge task, x5 tokens)
  7.5 - 9.1   "ДОРОГО ≠ ПЛОХО"
  9.0 - 10.0  CTA
"""
import os, sys, math, time
import numpy as np
from PIL import Image
from engine import *
import overlay as ov

W, H, FPS, DUR = 1080, 1920, 24, 10.0
NF = int(round(DUR * FPS))

C = dict(
    bg_in=(0.045, 0.051, 0.072), bg_out=(0.002, 0.003, 0.007),
    white=(0.95, 0.96, 0.99), dim=(0.52, 0.58, 0.68),
    cyan=(0.24, 0.80, 1.0), teal=(0.15, 0.95, 0.80), orange=(1.0, 0.55, 0.25),
    green=(0.40, 0.95, 0.55), violet=(0.66, 0.48, 1.0), red=(1.0, 0.35, 0.38),
    gold=(1.0, 0.78, 0.32), steel=(0.62, 0.68, 0.78),
)
BODY = (0.135, 0.145, 0.175)
BODY2 = (0.19, 0.205, 0.24)
FLOOR_Y = -1.0
FOG = (0.010, 0.012, 0.020)


# ---------------------------------------------------------------- helpers
def clamp01(x): return 0.0 if x < 0 else (1.0 if x > 1 else x)

def spring(p, k=1.0):
    p = clamp01(p)
    if p >= 1: return 1.0
    return 1 - math.cos(p * math.pi * 0.5 * (1.0 + 0.28 * k)) * math.exp(-3.6 * p * k) * (1 - p) ** 0.4

def pop(t, t0, dur=0.55):
    return spring((t - t0) / dur)

def outa(t, t0, dur=0.4):
    """alpha 0..1 ramp"""
    return clamp01((t - t0) / dur)

def fall(t, t0, dur=0.4):
    """0..1 ease-in for exits"""
    p = clamp01((t - t0) / dur)
    return p * p

def obj(mesh="rbox", pos=(0, 0, 0), size=(1, 1, 1), color=BODY, metal=0.45, rough=0.32,
        emissive=0.0, rot=(0, 0, 0), hidden=False):
    if np.isscalar(size): size = (size, size, size)
    return dict(mesh=mesh, pos=pos, size=size, color=color, metal=metal, rough=rough,
                emissive=emissive, rot=rot, hidden=hidden)


def half_w(R, fov=31.0, W_=None, H_=None):
    """half of the visible world width at distance R"""
    return R * math.tan(math.radians(fov) / 2) * (W_ or W) / (H_ or H)


# ---------------------------------------------------------------- camera timeline
CAM_KEYS = [
    # t,   dist,  ty,    pitch, fov
    (0.00, 6.90, 0.35, 0.395, 31.0),   # hero: pushes in
    (0.80, 6.60, 0.25, 0.400, 31.0),
    (1.35, 6.40, 0.10, 0.430, 31.0),
    (2.25, 4.35, -0.16, 0.470, 30.5),  # seven machines
    (3.10, 4.15, -0.10, 0.480, 30.5),
    (3.85, 4.80, 0.00, 0.470, 31.0),   # haiku
    (4.60, 4.90, -0.05, 0.450, 31.0),
    (5.20, 4.90, -0.20, 0.420, 31.0),  # sonnet
    (5.90, 5.60, -0.15, 0.440, 31.0),
    (7.00, 6.20, -0.10, 0.440, 31.0),  # opus
    (7.90, 5.60, -0.30, 0.410, 31.0),
    (8.45, 4.30, -0.60, 0.340, 31.0),  # conclusion pad
    (9.20, 4.80, -0.30, 0.420, 31.0),
    (9.80, 6.40, 0.00, 0.470, 31.0),   # CTA
    (10.0, 6.20, 0.00, 0.470, 31.5),
]

def _lerp(a, b, p): return a + (b - a) * p

def cam_params(t):
    ks = CAM_KEYS
    if t <= ks[0][0]: return ks[0][1:]
    if t >= ks[-1][0]: return ks[-1][1:]
    for i in range(len(ks) - 1):
        t0, t1 = ks[i][0], ks[i + 1][0]
        if t0 <= t <= t1:
            p = (t - t0) / (t1 - t0)
            p = p * p * (3 - 2 * p)
            return tuple(_lerp(ks[i][j], ks[i + 1][j], p) for j in range(1, 5))
    return ks[-1][1:]

def camera(t):
    dist, ty, pitch, fov = cam_params(t)
    az = 0.52 - 0.035 * t + 0.045 * math.sin(t * 0.85)
    tgt = np.array([0.0, ty - 0.05 * math.sin(t * 0.8), 0.0])
    eye = tgt + np.array([dist * math.cos(pitch) * math.sin(az),
                          dist * math.sin(pitch),
                          dist * math.cos(pitch) * math.cos(az)])
    eye += np.array([0.025 * math.sin(t * 3.1), 0.03 * math.sin(t * 2.6 + 1.0), 0.0])
    return dict(pos=tuple(eye), target=tuple(tgt), fov=fov,
                dof=dict(focus=float(np.linalg.norm(eye - tgt)), aperture=0.11))


# ---------------------------------------------------------------- scene builders
def machine(x, z, p, accent, size=0.46, glow=0.9, y0=FLOOR_Y):
    """little 'AI machine' rising from the floor with overshoot"""
    s = size
    y = y0 + spring(p) * s
    ry = (1 - spring(p)) * 1.3
    o = []
    o.append(obj("rbox", (x, y + s * 0.5, z), (s, s, s), BODY, metal=0.45, rough=0.30, rot=(0, ry, 0)))
    o.append(obj("rbox", (x, y + s * 0.94, z), (s * 0.70, s * 0.11, s * 0.70), accent,
                 metal=0.15, rough=0.20, emissive=glow, rot=(0, ry, 0)))
    o.append(obj("box", (x + s * 0.24, y + s * 0.44, z - s * 0.30), (s * 0.13, s * 0.02, s * 0.08),
                 accent, metal=0.1, rough=0.1, emissive=2.2, rot=(0, ry, 0)))
    return o


def build(t):
    objs = []
    # ---- floor (shadow catcher, melts into the fog)
    objs.append(obj("plane", (0, FLOOR_Y, 0), (60, 60, 1), (0.014, 0.016, 0.022),
                    metal=0.30, rough=0.50, rot=(-math.pi / 2, 0, 0)))

    # ================================================== 1. hero (until 1.35)
    if t < 1.55:
        p = pop(t, 0.02, 0.85)
        sink = fall(t, 1.02, 0.45) * 2.9
        y0 = (1 - p) * -2.9 - sink
        rot = (1 - p) * 0.45 + sink * 0.15
        objs += [
            obj("rbox", (0, y0 + 0.13, 0), (1.9, 0.24, 1.9), BODY, metal=0.72, rough=0.22, rot=(0, rot, 0)),
            obj("rbox", (0, y0 + 0.29, 0), (1.5, 0.08, 1.5), (0.28, 0.31, 0.38), metal=0.88, rough=0.13, rot=(0, rot, 0)),
            obj("sphere", (0, y0 + 0.40, 0), (0.30, 0.30, 0.30), C["cyan"], metal=0.0, rough=0.22, emissive=5.0),
            obj("torus", (0, y0 + 0.40, 0), (0.80, 0.80, 0.80), (0.55, 0.60, 0.70), metal=1.0, rough=0.10,
                rot=(1.15 + 0.35 * math.sin(t * 2.0), t * 1.5, 0.22)),
        ]

    # ================================================== 2. seven machines
    if 1.20 <= t <= 3.05:
        fade = fall(t, 2.68, 0.35)
        acc = [C["violet"], C["teal"], C["cyan"], C["green"], C["gold"], C["orange"], C["red"]]
        spots = [(-0.63, 0.26), (-0.21, 0.26), (0.21, 0.26), (0.63, 0.26),
                 (-0.42, -0.44), (0.0, -0.44), (0.42, -0.44)]
        for i, (x, z) in enumerate(spots):
            t0 = 1.26 + i * 0.078
            p = pop(t, t0, 0.55)
            if p <= 0.001: continue
            ms = machine(x, z, p, acc[i], size=0.365, glow=0.85)
            for o in ms:
                o["pos"] = (o["pos"][0], o["pos"][1] - fade * 2.6, o["pos"][2])
            objs += ms

    # ================================================== 3. haiku
    if 2.78 <= t <= 4.55:
        fin = fall(t, 4.22, 0.30)
        p = pop(t, 2.84, 0.6)
        y = FLOOR_Y + spring(p) * 0.62 - fin * 2.4
        objs += [
            obj("rbox", (0.06, y, 0.02), (0.62, 0.58, 0.60), BODY, metal=0.5, rough=0.26,
                rot=(0, (1 - p) * 1.1 - 0.12, 0)),
            obj("rbox", (0.06, y + 0.32, 0.02), (0.44, 0.07, 0.44), C["teal"], metal=0.15, rough=0.20,
                emissive=1.05, rot=(0, (1 - p) * 1.1 - 0.12, 0)),
            obj("box", (0.30, y + 0.02, 0.34), (0.10, 0.02, 0.05), C["teal"], metal=0.0, rough=0.1,
                emissive=1.6, rot=(0, -0.12, 0)),
        ]
        # request cube flying in and being swallowed
        if t > 3.10:
            q = spring(clamp01((t - 3.10) / 0.45), 1.1)
            sq = 1.0 - 0.66 * clamp01((t - 3.55) / 0.30)
            objs.append(obj("rbox", (-1.15 + q * 1.02, y + 0.18, 0.24), (0.16 * sq, 0.16 * sq, 0.16 * sq),
                            (0.70, 0.74, 0.80), metal=0.35, rough=0.16, rot=(0.25, t * 3.2, 0.2),
                            emissive=0.12))
        # small answer tile
        r = pop(t, 3.62, 0.5)
        if t > 3.62:
            yy = FLOOR_Y + 0.26 + (1 - r) * 1.2 - fin * 2.4
            objs += [
                obj("rbox", (0.52, yy, -0.42), (0.40, 0.26, 0.30), (0.80, 0.86, 0.92), metal=0.25, rough=0.22,
                    rot=(0, -0.45 + (1 - r) * 0.9, 0)),
                obj("box", (0.52, yy + 0.135, -0.42), (0.20, 0.015, 0.12), C["green"], metal=0, rough=0.1,
                    emissive=1.9, rot=(0, -0.45 + (1 - r) * 0.9, 0)),
            ]

    # ================================================== 4. sonnet
    if 4.08 <= t <= 5.95:
        fin = fall(t, 5.60, 0.30)
        base = FLOOR_Y + (1 - pop(t, 4.10, 0.55)) * -2.4 + 0.05 - fin * 2.5
        objs.append(obj("rbox", (0, base, 0), (0.86, 0.10, 0.66), BODY2, metal=0.60, rough=0.26))
        cards = [(C["cyan"], 4.34), (C["violet"], 4.56), (C["teal"], 4.78)]
        for i, (acc, t0) in enumerate(cards):
            p = pop(t, t0, 0.5)
            y = FLOOR_Y + 0.16 + i * 0.155 + (1 - p) * 1.6 - fin * 2.5
            z = 0.10 - i * 0.09
            ry = (1 - p) * 0.8 - 0.16 - i * 0.06
            objs += [
                obj("rbox", (0, y, z), (0.72, 0.10, 0.52), (0.215, 0.235, 0.295), metal=0.55, rough=0.26,
                    rot=(0, ry, 0)),
                obj("box", (-0.20, y + 0.055, z + 0.20), (0.22, 0.015, 0.07), acc, metal=0.0, rough=0.1,
                    emissive=1.5, rot=(0, ry, 0)),
                obj("box", (-0.17, y + 0.055, z + 0.02), (0.28, 0.015, 0.018), (0.50, 0.56, 0.66),
                    metal=0.0, rough=0.45, emissive=0.5, rot=(0, ry, 0)),
            ]

    # ================================================== 5. opus
    if 5.52 <= t <= 7.75:
        fin = fall(t, 7.40, 0.30)
        p = pop(t, 5.58, 0.6)
        lift = (1 - p) * -2.6 - fin * 2.6
        objs += [
            obj("rbox", (0, FLOOR_Y + 0.13 + lift, 0), (1.28, 0.26, 1.02), (0.165, 0.175, 0.215),
                metal=0.68, rough=0.24),
            obj("rbox", (0, FLOOR_Y + 0.285 + lift, 0), (1.02, 0.07, 0.80), (0.30, 0.34, 0.42),
                metal=0.85, rough=0.14),
        ]
        rr = pop(t, 6.00, 0.65)
        objs.append(obj("torus", (0.0, FLOOR_Y + 0.62 + lift, 0.06), (0.78, 0.78, 0.78), C["gold"],
                        metal=1.0, rough=0.16, emissive=0.22,
                        rot=(1.45 + 0.08 * math.sin(t * 1.6), t * 0.9, 0.12 * math.cos(t * 1.2))))
        for i in range(5):
            tc = 6.28 + i * 0.18
            if t < tc: continue
            q = clamp01((t - tc) / 0.48)
            fallp = 1 - (1 - q) ** 2
            st = clamp01((q - 0.66) / 0.34)
            bounce = 0.10 * math.sin(st * math.pi) * (1 - st)
            yy = FLOOR_Y + 0.33 + fallp * (0.92 - i * 0.055) + bounce + lift
            objs.append(obj("cyl", (-0.30, yy, 0.30), (0.20, 0.06, 0.20), C["gold"],
                            metal=1.0, rough=0.18, emissive=0.20, rot=(0, t * 2.2, 0)))

    # ================================================== 6. conclusion platform
    if 7.50 <= t <= 9.20:
        fin = fall(t, 8.98, 0.28)
        p = pop(t, 7.56, 0.6)
        lift = (1 - p) * -2.6 - fin * 2.6
        objs += [
            obj("rbox", (0, FLOOR_Y + 0.05 + lift, 0), (1.15, 0.10, 0.66), (0.075, 0.085, 0.115),
                metal=0.55, rough=0.26),
            obj("box", (0, FLOOR_Y + 0.075 + lift, 0.325), (1.00, 0.012, 0.012), C["cyan"], metal=0.0,
                rough=0.20, emissive=1.6),
            obj("box", (0, FLOOR_Y + 0.075 + lift, -0.325), (1.00, 0.012, 0.012), C["cyan"], metal=0.0,
                rough=0.20, emissive=1.2),
        ]

    # ================================================== 7. CTA
    if t >= 9.00:
        p = pop(t, 9.04, 0.7)
        y = FLOOR_Y + spring(p) * 0.60
        objs += [
            obj("rbox", (0, y, 0), (1.32, 0.20, 1.32), BODY, metal=0.72, rough=0.20),
            obj("rbox", (0, y + 0.145, 0), (1.05, 0.07, 1.05), (0.28, 0.31, 0.38), metal=0.88, rough=0.13),
            obj("sphere", (0, y + 0.235, 0), (0.22, 0.22, 0.22), C["cyan"], metal=0.0, rough=0.22, emissive=3.4),
            obj("torus", (0, y + 0.235, 0), (0.56, 0.56, 0.56), (0.55, 0.60, 0.70), metal=1.0, rough=0.10,
                rot=(1.2, (t - 9.0) * 1.4, 0.2)),
        ]

    sc = dict(camera=camera(t), objects=objs, bg=(0, 0, 0), bg_colors=(C["bg_in"], C["bg_out"]),
              ambient=dict(top=(0.055, 0.070, 0.110), bot=(0.006, 0.008, 0.014), rim=(0.26, 0.40, 0.66)),
              lights=dict(list=[((0.62, 0.92, 0.5), (1.0, 0.97, 0.94)),
                                ((-0.85, 0.35, 0.45), (0.30, 0.42, 0.75)),
                                ((0.1, -0.15, -1.0), (0.28, 0.25, 0.44))],
                          key_int=2.9, fill_int=1.30, rim_int=1.05),
              shadows=True, shadow_center=(0, -0.95, 0), shadow_extent=6.5,
              fog=(FOG, 0.085),
              bg_glow=bg_glow(t),
              exposure=1.00, bloom=0.62, bloom_thresh=0.86, vignette=0.35, grain=0.010, time=t)
    return sc


def bg_glow(t):
    """(colour, screen pos, intensity) for the atmospheric backlight"""
    keys = [
        (0.0, (0.09, 0.22, 0.40), (0.50, 0.62), 0.55),
        (2.4, (0.10, 0.24, 0.42), (0.50, 0.68), 0.62),
        (3.4, (0.05, 0.28, 0.26), (0.55, 0.68), 0.50),
        (4.8, (0.16, 0.13, 0.36), (0.55, 0.68), 0.55),
        (6.4, (0.34, 0.20, 0.06), (0.58, 0.68), 0.55),
        (7.8, (0.06, 0.16, 0.34), (0.50, 0.66), 0.50),
        (9.2, (0.08, 0.20, 0.36), (0.50, 0.64), 0.58),
        (10.0, (0.08, 0.20, 0.36), (0.50, 0.64), 0.58),
    ]
    for i in range(len(keys) - 1):
        t0, c0, p0, i0 = keys[i]
        t1, c1, p1, i1 = keys[i + 1]
        if t0 <= t <= t1:
            q = (t - t0) / (t1 - t0); q = q * q * (3 - 2 * q)
            c = tuple(_lerp(c0[j], c1[j], q) for j in range(3))
            p = (_lerp(p0[0], p1[0], q), _lerp(p0[1], p1[1], q))
            return (c, p, _lerp(i0, i1, q))
    return keys[0][1:]


# ---------------------------------------------------------------- 2D overlay
def overlay_layers(t):
    L = []
    X0, X1 = 84, W - 84
    # ---- intro fade
    intro = clamp01(t / 0.5)

    # ================= hero title
    if t <= 1.42:
        ain = clamp01((t - 0.30) / 0.45) * (1 - clamp01((t - 0.98) / 0.26))
        dy = (1 - clamp01((t - 0.30) / 0.6)) * 24
        L.append(ov.text_layer(W, H, "Куда уходят", ov.HEAD, 88, (X0, 300 + dy), anchor="lm",
                               color=(238, 242, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "токены?", ov.HEAD, 88, (X0, 396 + dy), anchor="lm",
                               color=(238, 242, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "claude  ·  короткий разбор", ov.MONO, 26, (X0, 476),
                               anchor="lm", color=(130, 155, 195), alpha=ain * 0.9, tracking=3))

    # ================= scene 2: seven machines
    if 1.36 <= t <= 2.92:
        ain = clamp01((t - 1.40) / 0.45) * (1 - clamp01((t - 2.62) / 0.28))
        L.append(ov.text_layer(W, H, "СЕМЬ", ov.HEAD, 126, (X0, 296), anchor="lm",
                               color=(235, 240, 250), alpha=ain, tracking=1))
        L.append(ov.text_layer(W, H, "МАШИН", ov.HEAD, 126, (X0, 416), anchor="lm",
                               color=(235, 240, 250), alpha=ain, tracking=1))
        L.append(ov.text_layer(W, H, "и один вопрос: куда ушли токены?", ov.HEAD2, 40, (X0, 510),
                               anchor="lm", color=(150, 168, 195), alpha=ain))

    # ================= scene 3: haiku
    if 2.88 <= t <= 4.45:
        ain = clamp01((t - 2.92) / 0.42) * (1 - clamp01((t - 4.22) / 0.25))
        L.append(ov.plate(W, H, (X0, 276, X0 + 196, 340), 20, (26, 200, 165), 0.14,
                          border=(56, 225, 185), border_alpha=0.45, glow=9, glow_col=(0.06, 0.45, 0.36)))
        L.append(ov.text_layer(W, H, "HAIKU", ov.MONO, 36, (X0 + 98, 308), anchor="mm",
                               color=(96, 245, 208), alpha=ain, tracking=3))
        L.append(ov.text_layer(W, H, "кнопку на 4 пикселя", ov.HEAD, 78, (X0, 430), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "влево", ov.HEAD, 78, (X0, 522), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        a2 = clamp01((t - 3.62) / 0.35) * ain
        L.append(ov.text_layer(W, H, "готово. копейки.", ov.HEAD2, 40, (X0, 616), anchor="lm",
                               color=(96, 245, 208), alpha=a2))
        L.append(ov.meter(W, H, (X1 - 210, 1650), 210, 0.10, 4, C["teal"], ain * 0.95, label="ТОКЕНЫ"))
        L.append(ov.text_layer(W, H, "1  токен", ov.MONO, 28, (X1, 1710), anchor="rm",
                               color=(140, 162, 192), alpha=ain * 0.85, tracking=2))

    # ================= scene 4: sonnet
    if 4.18 <= t <= 5.88:
        ain = clamp01((t - 4.22) / 0.42) * (1 - clamp01((t - 5.55) / 0.25))
        L.append(ov.plate(W, H, (X0, 276, X0 + 220, 340), 20, (108, 88, 255), 0.15,
                          border=(138, 118, 255), border_alpha=0.45, glow=9, glow_col=(0.22, 0.18, 0.55)))
        L.append(ov.text_layer(W, H, "SONNET", ov.MONO, 36, (X0 + 110, 308), anchor="mm",
                               color=(178, 164, 255), alpha=ain, tracking=3))
        L.append(ov.text_layer(W, H, "собери страницу", ov.HEAD, 74, (X0, 424), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "оплаты", ov.HEAD, 74, (X0, 514), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        rows = [("headers", 4.45), ("форма карты", 4.64), ("валидация", 4.83)]
        for i, (name, t0) in enumerate(rows):
            a2 = clamp01((t - t0) / 0.3) * ain
            dy = (1 - clamp01((t - t0) / 0.4)) * 20
            yy = 606 + i * 54 + dy
            L.append(ov.text_layer(W, H, "✓", ov.MONO, 30, (X0 + 2, yy), anchor="lm",
                                   color=(178, 164, 255), alpha=a2))
            L.append(ov.text_layer(W, H, name, ov.HEAD2, 36, (X0 + 44, yy), anchor="lm",
                                   color=(200, 210, 230), alpha=a2, tracking=1))
        L.append(ov.meter(W, H, (X1 - 210, 1650), 210, 0.42, 5, C["violet"], ain * 0.95, label="ТОКЕНЫ"))
        L.append(ov.text_layer(W, H, "~30 000", ov.MONO, 28, (X1, 1710), anchor="rm",
                               color=(150, 140, 200), alpha=ain * 0.9, tracking=2))

    # ================= scene 5: opus
    if 5.62 <= t <= 7.68:
        ain = clamp01((t - 5.66) / 0.42) * (1 - clamp01((t - 7.36) / 0.25))
        L.append(ov.plate(W, H, (X0, 276, X0 + 176, 340), 20, (255, 188, 88), 0.15,
                          border=(255, 205, 120), border_alpha=0.5, glow=9, glow_col=(0.45, 0.33, 0.10)))
        L.append(ov.text_layer(W, H, "OPUS", ov.MONO, 36, (X0 + 88, 308), anchor="mm",
                               color=(255, 205, 120), alpha=ain, tracking=3))
        L.append(ov.text_layer(W, H, "перепиши весь", ov.HEAD, 72, (X0, 422), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "бэкенд", ov.HEAD, 72, (X0, 512), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        n = min(5, max(0, int((t - 6.28) / 0.18) + 1)) if t > 6.28 else 0
        if n > 0:
            a2 = clamp01((t - 6.32) / 0.3) * ain
            L.append(ov.text_layer(W, H, "×%d" % n, ov.HEAD, 168, (X1, 552), anchor="rm",
                                   color=(255, 190, 90), alpha=a2, glow=30, glow_col=(0.55, 0.32, 0.04)))
            L.append(ov.text_layer(W, H, "токенов за тот же запрос", ov.MONO, 26, (X1, 626),
                                   anchor="rm", color=(232, 182, 122), alpha=a2 * 0.95, tracking=1))
        L.append(ov.meter(W, H, (X1 - 210, 1650), 210, 1.0, 8, C["gold"], ain * 0.95, label="ТОКЕНЫ"))
        L.append(ov.text_layer(W, H, "~150 000", ov.MONO, 28, (X1, 1710), anchor="rm",
                               color=(200, 168, 120), alpha=ain * 0.9, tracking=2))

    # ================= scene 6: conclusion
    if 7.62 <= t <= 9.12:
        ain = clamp01((t - 7.66) / 0.42) * (1 - clamp01((t - 8.93) / 0.25))
        L.append(ov.text_layer(W, H, "ДОРОГО", ov.HEAD, 108, (W // 2, 320), anchor="mm",
                               color=(238, 242, 250), alpha=ain, tracking=2))
        L.append(ov.text_layer(W, H, "≠", ov.HEAD, 104, (W // 2, 432), anchor="mm",
                               color=(0, 170, 255), alpha=ain, glow=28, glow_col=(0.1, 0.5, 0.9)))
        L.append(ov.text_layer(W, H, "ПЛОХО", ov.HEAD, 108, (W // 2, 546), anchor="mm",
                               color=(238, 242, 250), alpha=ain, tracking=2))
        a2 = clamp01((t - 8.12) / 0.4) * ain
        L.append(ov.text_layer(W, H, "просто оно стоит своих токенов", ov.HEAD2, 38,
                               (W // 2, 660), anchor="mm", color=(150, 168, 195), alpha=a2))

    # ================= CTA
    if t >= 9.12:
        ain = clamp01((t - 9.16) / 0.5)
        L.append(ov.text_layer(W, H, "ТАКИЕ 3D-РОЛИКИ", ov.HEAD, 66, (W // 2, 470), anchor="mm",
                               color=(238, 242, 250), alpha=ain, tracking=1))
        L.append(ov.text_layer(W, H, "делаю сам — от сценария до рендера", ov.HEAD2, 36,
                               (W // 2, 548), anchor="mm", color=(150, 168, 195), alpha=ain))
        L.append(ov.plate(W, H, (W // 2 - 250, 624, W // 2 + 250, 726), 26, (22, 26, 38), 0.55,
                          border=(78, 98, 138), border_alpha=0.42, glow=12, glow_col=(0.08, 0.22, 0.45)))
        L.append(ov.text_layer(W, H, "@your_handle", ov.MONO, 40, (W // 2, 676), anchor="mm",
                               color=(120, 220, 255), alpha=ain, tracking=2, glow=14,
                               glow_col=(0.08, 0.35, 0.55)))

    # ================= persistent frame furniture
    ab = clamp01((t - 0.2) / 0.7) * (1 - clamp01((t - 9.6) / 0.4))
    L.append(ov.text_layer(W, H, "AI × TOKENS", ov.MONO, 22, (X0, 104), anchor="lm",
                           color=(130, 150, 180), alpha=0.5 * ab, tracking=6))
    L.append(ov.text_layer(W, H, "3D EXPLAINER", ov.MONO, 22, (X1, 104), anchor="rm",
                           color=(130, 150, 180), alpha=0.5 * ab, tracking=6))
    L.append(ov.line(W, H, (X0, 134), (X1, 134), 2, (90, 110, 150), alpha=0.22 * ab))
    # progress bar at the bottom
    L.append(ov.progress(W, H, X0, X1, 1836, t / DUR, 0.35 * ab))
    return L


# ---------------------------------------------------------------- frame
class Short:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.W, self.H = int(W * scale), int(H * scale)
        self.r = Renderer(self.W, self.H)
        self.post = ov.Post(self.W, self.H)

    def frame(self, t, fi=0):
        sc = build(t)
        img = self.r.render(sc)
        canvas = np.asarray(img).astype(np.float32) / 255.0
        for layer in overlay_layers(t):
            if self.scale != 1.0:
                layer = np.asarray(Image.fromarray(layer.astype(np.uint8))
                                   .resize((self.W, self.H), Image.LANCZOS)).astype(np.float32)
            canvas = ov.paste_rgba(canvas, layer)
        canvas = self.post.bloom(canvas, amount=0.20, thresh=0.74, radius=16)
        canvas = self.post.finish(canvas, fi, grain=0.020, vig=0.62)
        return self.post.to_image(canvas)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    scale = float(os.environ.get("SCALE", "1"))
    s = Short(scale)
    if cmd == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            t0 = time.time()
            im = s.frame(t, int(t * FPS))
            im.save("/home/user/motion/out/f_%.2f.png" % t)
            print("t=%.2f  %.2fs" % (t, time.time() - t0), flush=True)
    elif cmd == "sheet":
        # low-res contact sheet across the whole timeline
        step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
        os.makedirs("/home/user/motion/out/sheet", exist_ok=True)
        ts = np.arange(0, DUR - 1e-6, step)
        for t in ts:
            im = s.frame(float(t), int(t * FPS))
            im.save("/home/user/motion/out/sheet/s_%05.2f.png" % t)
        print("sheet done", len(ts))
    elif cmd == "chunk":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        outdir = os.environ.get("OUTDIR", "/tmp/frames")
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        for fi in range(a, min(b, NF)):
            t = fi / FPS
            im = s.frame(t, fi)
            im.save(os.path.join(outdir, "%04d.png" % fi))
            if fi % 10 == 0:
                print("frame %d/%d  %.1fs" % (fi, NF, time.time() - t0), flush=True)
        print("chunk done %.1fs" % (time.time() - t0), flush=True)
    elif cmd == "video":
        os.makedirs("/home/user/motion/out/frames", exist_ok=True)
        t0 = time.time()
        for fi in range(NF):
            t = fi / FPS
            im = s.frame(t, fi)
            im.save("/home/user/motion/out/frames/%04d.png" % fi)
            if fi % 24 == 0:
                print("frame %d/%d  %.1fs" % (fi, NF, time.time() - t0), flush=True)
        print("done %.1fs" % (time.time() - t0))
