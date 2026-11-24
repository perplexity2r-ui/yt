"""Short #2 — "Claude API: цены в цифрах" — 12 s, 1080x1920, 24 fps.
Data (Nov 2026, $ per 1M tokens, input/output):
  Haiku 5.5  0.10 / 0.50
  Sonnet 5.5 2    / 10
  Opus 5.5   4    / 20
  Fable 5.2  8    / 40
  cache read = 5% of input  ·  Batch API = -50%
"""
import os, sys, math, time
import numpy as np
from PIL import Image
from engine import *
import overlay as ov

W, H, FPS, DUR = 1080, 1920, 24, 12.0
NF = int(round(DUR * FPS))

C = dict(
    bg_in=(0.042, 0.048, 0.068), bg_out=(0.002, 0.003, 0.007),
    cyan=(0.24, 0.80, 1.0), teal=(0.15, 0.95, 0.80), violet=(0.66, 0.48, 1.0),
    gold=(1.0, 0.78, 0.32), orange=(1.0, 0.55, 0.25), green=(0.40, 0.95, 0.55),
    red=(1.0, 0.38, 0.40),
)
BODY = (0.135, 0.145, 0.175)
BODY2 = (0.19, 0.205, 0.24)
FLOOR_Y = -1.0
FOG = (0.010, 0.012, 0.020)

MODELS = [
    # name,         price_out, accent,             tag
    ("HAIKU 5.5", 0.50, C["teal"]),
    ("SONNET 5.5", 10.0, C["violet"]),
    ("OPUS 5.5", 20.0, C["gold"]),
    ("FABLE 5.2", 40.0, C["orange"]),
]


def clamp01(x): return 0.0 if x < 0 else (1.0 if x > 1 else x)

def spring(p, k=1.0):
    p = clamp01(p)
    if p >= 1: return 1.0
    return 1 - math.cos(p * math.pi * 0.5 * (1.0 + 0.28 * k)) * math.exp(-3.6 * p * k) * (1 - p) ** 0.4

def pop(t, t0, dur=0.55):
    return spring((t - t0) / dur)

def fall(t, t0, dur=0.4):
    p = clamp01((t - t0) / dur)
    return p * p

def obj(mesh="rbox", pos=(0, 0, 0), size=(1, 1, 1), color=BODY, metal=0.45, rough=0.32,
        emissive=0.0, rot=(0, 0, 0), hidden=False):
    if np.isscalar(size): size = (size, size, size)
    return dict(mesh=mesh, pos=pos, size=size, color=color, metal=metal, rough=rough,
                emissive=emissive, rot=rot, hidden=hidden)


def tok_size(price_out, max_size=0.44):
    """token cube size with VOLUME proportional to price (cube root) — honest scaling"""
    return max_size * (price_out / 40.0) ** (1.0 / 3.0)


# ---------------------------------------------------------------- camera
def _lerp(a, b, p): return a + (b - a) * p

CAM_KEYS = [
    # t,    dist,  ty,    pitch, fov
    (0.00, 6.40, 0.30, 0.400, 31.0),
    (1.20, 6.00, 0.35, 0.410, 31.0),
    (1.90, 5.20, 0.10, 0.400, 31.0),
    (3.10, 5.60, 0.55, 0.380, 31.0),
    (4.40, 6.00, 0.95, 0.370, 31.0),
    (5.30, 4.90, 0.20, 0.420, 30.5),
    (6.60, 4.70, 0.10, 0.430, 30.5),
    (7.60, 4.60, 0.10, 0.430, 30.5),
    (8.80, 4.90, 0.20, 0.420, 31.0),
    (9.70, 5.20, 0.35, 0.410, 31.0),
    (10.6, 6.20, 0.40, 0.400, 31.0),
    (12.0, 6.00, 0.40, 0.400, 31.5),
]

def cam_params(t):
    ks = CAM_KEYS
    if t <= ks[0][0]: return ks[0][1:]
    if t >= ks[-1][0]: return ks[-1][1:]
    for i in range(len(ks) - 1):
        if ks[i][0] <= t <= ks[i + 1][0]:
            p = (t - ks[i][0]) / (ks[i + 1][0] - ks[i][0])
            p = p * p * (3 - 2 * p)
            return tuple(_lerp(ks[i][j], ks[i + 1][j], p) for j in range(1, 5))
    return ks[-1][1:]

def camera(t):
    dist, ty, pitch, fov = cam_params(t)
    az = 0.44 - 0.02 * t + 0.04 * math.sin(t * 0.8)
    tgt = np.array([0.0, ty - 0.04 * math.sin(t * 0.8), 0.0])
    eye = tgt + np.array([dist * math.cos(pitch) * math.sin(az), dist * math.sin(pitch),
                          dist * math.cos(pitch) * math.cos(az)])
    eye += np.array([0.02 * math.sin(t * 3.1), 0.025 * math.sin(t * 2.6 + 1.0), 0.0])
    return dict(pos=tuple(eye), target=tuple(tgt), fov=fov,
                dof=dict(focus=float(np.linalg.norm(eye - tgt)), aperture=0.11))


# ---------------------------------------------------------------- scene
def build(t):
    objs = []
    objs.append(obj("plane", (0, FLOOR_Y, 0), (60, 60, 1), (0.014, 0.016, 0.022),
                    metal=0.30, rough=0.50, rot=(-math.pi / 2, 0, 0)))

    # ======================= A. hero token (0 – 1.5)
    if t < 1.75:
        p = pop(t, 0.02, 0.85)
        sink = fall(t, 1.10, 0.5) * 3.0
        y = (1 - p) * -2.9 - sink
        objs += [
            obj("rbox", (0, y + 0.12, 0), (1.75, 0.22, 1.75), BODY, metal=0.72, rough=0.22),
            obj("rbox", (0, y + 0.27, 0), (1.38, 0.07, 1.38), (0.28, 0.31, 0.38), metal=0.88, rough=0.13),
            obj("rbox", (0, y + 0.46, 0), (0.34, 0.34, 0.34), C["cyan"], metal=0.15, rough=0.18,
                emissive=2.6, rot=(0, t * 0.9, 0)),
            obj("torus", (0, y + 0.46, 0), (0.72, 0.72, 0.72), (0.55, 0.60, 0.70), metal=1.0, rough=0.10,
                rot=(1.2, t * 1.4, 0.2)),
        ]

    # ======================= B. price staircase (1.4 – 5.2)
    if 1.35 <= t <= 5.35:
        fade = fall(t, 5.05, 0.30)
        for i, (name, price, acc) in enumerate(MODELS):
            t0 = 1.45 + i * 0.20
            p = pop(t, t0, 0.6)
            if p <= 0.001: continue
            y_step = -1.0 + i * 0.60 + (1 - p) * 1.4 - fade * 3.2
            z_step = -i * 0.52
            sz = tok_size(price)
            # platform
            objs += [
                obj("rbox", (0, y_step + 0.05, z_step), (1.15, 0.10, 0.78), (0.10, 0.11, 0.145),
                    metal=0.62, rough=0.22, rot=(0, 0, 0)),
                obj("box", (0, y_step + 0.10, z_step + 0.39), (1.02, 0.010, 0.010), acc,
                    metal=0.0, rough=0.20, emissive=1.5),
                # token cube, volume ∝ price
                obj("rbox", (0, y_step + 0.10 + sz / 2, z_step), (sz, sz, sz), acc,
                    metal=0.35, rough=0.20, emissive=0.55, rot=(0, 0.5 + 0.25 * i, 0)),
            ]
        # connecting light beams between steps
        for i in range(3):
            t0 = 2.35 + i * 0.16
            a = clamp01((t - t0) / 0.35) * (1 - fall(t, 5.0, 0.3))
            if a <= 0.01: continue
            y0 = -1.0 + i * 0.60 + 0.12; y1 = -1.0 + (i + 1) * 0.60 + 0.12
            z0 = -i * 0.52; z1 = -(i + 1) * 0.52
            objs.append(obj("box", (0, (y0 + y1) / 2, (z0 + z1) / 2),
                            (0.012, 0.012, 0.52 * 1.0), MODELS[i][2], metal=0.0, rough=0.2,
                            emissive=1.1 * a,
                            rot=(math.atan2(y1 - y0, abs(z1 - z0)) * -1.0, 0, 0)))

    # ======================= C. output = 5x input (5.1 – 7.0)
    if 5.05 <= t <= 7.05:
        p = pop(t, 5.10, 0.6)
        fin = fall(t, 6.80, 0.28)
        y0 = FLOOR_Y + 0.05 + (1 - p) * -2.2 - fin * 2.6
        objs += [
            obj("rbox", (0, y0, 0), (1.5, 0.10, 0.9), (0.10, 0.11, 0.145), metal=0.62, rough=0.22),
            obj("box", (0, y0 + 0.05, 0.45), (1.35, 0.010, 0.010), C["cyan"], metal=0.0, rough=0.2,
                emissive=1.4),
        ]
        # input bar (height 0.26) and output bar (height 1.30) -> exactly 5x
        for j, (h, col, x, t0) in enumerate([(0.26, C["cyan"], -0.24, 5.22),
                                             (1.30, C["teal"], 0.24, 5.62)]):
            pp = pop(t, t0, 0.55)
            hh = h * pp
            if hh <= 0.005: continue
            objs.append(obj("box", (x, y0 + 0.05 + hh / 2, 0), (0.26, hh, 0.26), col,
                            metal=0.25, rough=0.22, emissive=0.75))

    # ======================= D. prompt caching (6.9 – 8.9)
    if 6.90 <= t <= 8.95:
        p = pop(t, 6.94, 0.6)
        fin = fall(t, 8.72, 0.26)
        y0 = FLOOR_Y + 0.05 + (1 - p) * -2.2 - fin * 2.6
        shrink = clamp01((t - 7.85) / 0.55)
        sh = 1.0 - 0.63 * (shrink * shrink * (3 - 2 * shrink))     # volume -95% -> linear -63%
        h = 1.30 * sh
        objs += [
            obj("rbox", (0, y0, 0), (1.45, 0.10, 0.85), (0.10, 0.11, 0.145), metal=0.62, rough=0.22),
            obj("box", (0, y0 + 0.05, 0.425), (1.30, 0.010, 0.010), C["violet"], metal=0.0, rough=0.2,
                emissive=1.3),
            obj("box", (0, y0 + 0.05 + h / 2, 0), (0.34, h, 0.34), C["violet"], metal=0.25, rough=0.20,
                emissive=0.85),
            obj("torus", (0, y0 + 0.60, 0.0), (1.05, 1.05, 1.05), (0.55, 0.60, 0.70), metal=1.0,
                rough=0.10, rot=(1.5, t * 0.9, 0.1), emissive=0.15),
        ]
        # context tokens looping through the cache ring
        for i in range(6):
            ph = (t * 0.55 + i / 6.0) % 1.0
            ang = ph * 2 * math.pi
            objs.append(obj("rbox", (0.52 * math.cos(ang), y0 + 0.60 + 0.52 * math.sin(ang), 0.0),
                            (0.10, 0.10, 0.10), C["violet"], metal=0.2, rough=0.2, emissive=1.6,
                            rot=(0.3, ang, 0.2)))

    # ======================= E. batch -50% (8.8 – 10.3)
    if 8.80 <= t <= 10.4:
        p = pop(t, 8.84, 0.6)
        fin = fall(t, 10.15, 0.26)
        y0 = FLOOR_Y + 0.05 + (1 - p) * -2.2 - fin * 2.4
        cut = clamp01((t - 9.30) / 0.55)
        h = 1.10 * (1 - 0.5 * (cut * cut * (3 - 2 * cut)))
        objs += [
            obj("rbox", (0, y0, 0), (1.35, 0.10, 0.85), (0.10, 0.11, 0.145), metal=0.62, rough=0.22),
            obj("box", (0, y0 + 0.05, 0.425), (1.20, 0.010, 0.010), C["gold"], metal=0.0, rough=0.2,
                emissive=1.4),
            obj("box", (0, y0 + 0.05 + h / 2, 0), (0.34, h, 0.34), C["gold"], metal=0.25, rough=0.20,
                emissive=0.8),
        ]
        # the slice that got cut away floats off
        if cut > 0.02:
            hh = 1.10 - h
            objs.append(obj("box", (0.95 * cut, y0 + 0.05 + h + hh / 2 + 0.35 * cut, 0.0),
                            (0.30, hh, 0.30), C["gold"], metal=0.25, rough=0.22, emissive=0.35,
                            rot=(0, 0.6 * cut, 0.2 * cut)))

    # ======================= F. kicker + CTA (10.2 – 12)
    if t >= 10.2:
        p = pop(t, 10.24, 0.7)
        y = FLOOR_Y + spring(p) * 0.55
        objs += [
            obj("rbox", (0, y, 0), (1.55, 0.20, 1.55), BODY, metal=0.72, rough=0.20),
            obj("rbox", (0, y + 0.145, 0), (1.22, 0.07, 1.22), (0.28, 0.31, 0.38), metal=0.88, rough=0.13),
            obj("rbox", (0, y + 0.30, 0), (0.30, 0.30, 0.30), C["cyan"], metal=0.15, rough=0.18,
                emissive=2.2, rot=(0, (t - 10.2) * 1.1, 0)),
            obj("torus", (0, y + 0.30, 0), (0.62, 0.62, 0.62), (0.55, 0.60, 0.70), metal=1.0, rough=0.10,
                rot=(1.2, (t - 10.2) * 1.3, 0.2)),
        ]

    return dict(camera=camera(t), objects=objs, bg=(0, 0, 0), bg_colors=(C["bg_in"], C["bg_out"]),
                ambient=dict(top=(0.055, 0.070, 0.110), bot=(0.006, 0.008, 0.014),
                             rim=(0.26, 0.40, 0.66)),
                lights=dict(list=[((0.62, 0.92, 0.5), (1.0, 0.97, 0.94)),
                                  ((-0.85, 0.35, 0.45), (0.30, 0.42, 0.75)),
                                  ((0.1, -0.15, -1.0), (0.28, 0.25, 0.44))],
                            key_int=2.9, fill_int=1.30, rim_int=1.05),
                shadows=True, shadow_center=(0, -0.9, 0), shadow_extent=8.0,
                fog=(FOG, 0.085), bg_glow=bg_glow(t),
                exposure=1.00, bloom=0.62, bloom_thresh=0.86, vignette=0.35, grain=0.010, time=t)


def bg_glow(t):
    keys = [
        (0.0, (0.08, 0.20, 0.36), (0.50, 0.64), 0.55),
        (1.6, (0.06, 0.22, 0.30), (0.50, 0.66), 0.55),
        (3.2, (0.18, 0.12, 0.32), (0.50, 0.60), 0.50),
        (4.6, (0.32, 0.20, 0.05), (0.52, 0.58), 0.50),
        (6.0, (0.05, 0.26, 0.22), (0.50, 0.66), 0.50),
        (7.4, (0.16, 0.10, 0.34), (0.50, 0.66), 0.50),
        (9.0, (0.30, 0.20, 0.06), (0.50, 0.66), 0.50),
        (10.4, (0.08, 0.20, 0.34), (0.50, 0.64), 0.55),
        (12.0, (0.08, 0.20, 0.34), (0.50, 0.64), 0.55),
    ]
    for i in range(len(keys) - 1):
        t0, c0, p0, i0 = keys[i]
        t1, c1, p1, i1 = keys[i + 1]
        if t0 <= t <= t1:
            q = (t - t0) / (t1 - t0); q = q * q * (3 - 2 * q)
            return (tuple(_lerp(c0[j], c1[j], q) for j in range(3)),
                    (_lerp(p0[0], p1[0], q), _lerp(p0[1], p1[1], q)), _lerp(i0, i1, q))
    return keys[0][1:]


# ---------------------------------------------------------------- overlay
PRICE_ROWS = [  # click-in times for the staircase labels
    ("HAIKU 5.5", "$0.10", "$0.50", "×1", C["teal"], 1.60),
    ("SONNET 5.5", "$2", "$10", "×20", C["violet"], 1.80),
    ("OPUS 5.5", "$4", "$20", "×40", C["gold"], 2.00),
    ("FABLE 5.2", "$8", "$40", "×80", C["orange"], 2.20),
]

def overlay_layers(t):
    L = []
    X0, X1 = 84, W - 84

    # ---------- A: hook
    if t <= 1.6:
        ain = clamp01((t - 0.28) / 0.45) * (1 - clamp01((t - 1.05) / 0.28))
        L.append(ov.text_layer(W, H, "Сколько стоит", ov.HEAD, 84, (X0, 300), anchor="lm",
                               color=(238, 242, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "1 000 000", ov.HEAD, 84, (X0, 396), anchor="lm",
                               color=(238, 242, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "токенов?", ov.HEAD, 84, (X0, 492), anchor="lm",
                               color=(238, 242, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "claude api  ·  ноябрь 2026", ov.MONO, 26, (X0, 566),
                               anchor="lm", color=(130, 155, 195), alpha=ain * 0.9, tracking=3))

    # ---------- B: staircase labels
    if 1.55 <= t <= 5.25:
        ain_all = clamp01((t - 1.55) / 0.4) * (1 - clamp01((t - 4.95) / 0.28))
        L.append(ov.text_layer(W, H, "ЦЕНА 1М ТОКЕНОВ", ov.MONO, 24, (X0, 268), anchor="lm",
                               color=(140, 158, 190), alpha=ain_all * 0.85, tracking=5))
        L.append(ov.text_layer(W, H, "вход / выход", ov.HEAD2, 30, (X1, 268), anchor="rm",
                               color=(140, 158, 190), alpha=ain_all * 0.85))
        for i, (name, pin, pout, mult, acc, t0) in enumerate(PRICE_ROWS):
            a = clamp01((t - t0) / 0.32) * ain_all
            if a <= 0.003: continue
            yy = 352 + i * 104
            dy = (1 - clamp01((t - t0) / 0.4)) * 18
            L.append(ov.plate(W, H, (X0, yy - 40 + dy, X0 + 10, yy + 40 + dy), 5, acc, 0.95))
            L.append(ov.text_layer(W, H, name, ov.MONO, 32, (X0 + 28, yy - 20 + dy), anchor="lm",
                                   color=tuple(int(c * 255) for c in acc), alpha=a, tracking=2))
            L.append(ov.text_layer(W, H, mult, ov.HEAD2, 32, (X0 + 250, yy - 20 + dy), anchor="lm",
                                   color=(150, 168, 195), alpha=a * 0.95))
            L.append(ov.text_layer(W, H, pin, ov.MONO, 36, (X1, yy - 20 + dy), anchor="rm",
                                   color=(238, 242, 250), alpha=a, tracking=1))
            L.append(ov.text_layer(W, H, pout, ov.MONO, 36, (X1, yy + 24 + dy), anchor="rm",
                                   color=tuple(int(c * 255) for c in acc), alpha=a, tracking=1))
        a2 = clamp01((t - 2.55) / 0.4) * ain_all
        L.append(ov.text_layer(W, H, "выход всегда дороже входа в 5 раз", ov.HEAD2, 34,
                               (X0, 830), anchor="lm", color=(150, 168, 195), alpha=a2))

    # ---------- C: output = 5x input
    if 5.12 <= t <= 6.95:
        ain = clamp01((t - 5.16) / 0.42) * (1 - clamp01((t - 6.70) / 0.25))
        L.append(ov.text_layer(W, H, "ВЫХОД", ov.MONO, 44, (X0, 300), anchor="lm",
                               color=(96, 245, 208), alpha=ain * clamp01((t - 5.66) / 0.3), tracking=4))
        L.append(ov.text_layer(W, H, "×5", ov.HEAD, 150, (X1, 306), anchor="rm",
                               color=(96, 245, 208), alpha=ain * clamp01((t - 5.66) / 0.3),
                               glow=26, glow_col=(0.06, 0.4, 0.32)))
        L.append(ov.text_layer(W, H, "от входа", ov.HEAD2, 34, (X1, 386), anchor="rm",
                               color=(150, 168, 195), alpha=ain * clamp01((t - 5.66) / 0.3)))
        L.append(ov.text_layer(W, H, "1 запрос  ·  читаешь мало,", ov.HEAD2, 36, (X0, 470), anchor="lm",
                               color=(200, 210, 230), alpha=ain))
        L.append(ov.text_layer(W, H, "генеришь много", ov.HEAD2, 36, (X0, 524), anchor="lm",
                               color=(200, 210, 230), alpha=ain))

    # ---------- D: caching
    if 6.96 <= t <= 8.90:
        ain = clamp01((t - 7.00) / 0.42) * (1 - clamp01((t - 8.66) / 0.24))
        L.append(ov.plate(W, H, (X0, 268, X0 + 236, 334), 20, (108, 88, 255), 0.16,
                          border=(138, 118, 255), border_alpha=0.5, glow=9, glow_col=(0.22, 0.18, 0.55)))
        L.append(ov.text_layer(W, H, "PROMPT CACHE", ov.MONO, 32, (X0 + 118, 301), anchor="mm",
                               color=(178, 164, 255), alpha=ain, tracking=2))
        L.append(ov.text_layer(W, H, "повторный контекст", ov.HEAD, 66, (X0, 410), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "читается за 5%", ov.HEAD, 66, (X0, 492), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        a2 = clamp01((t - 7.90) / 0.4) * ain
        L.append(ov.text_layer(W, H, "−95%", ov.HEAD, 148, (X1, 700), anchor="rm",
                               color=(178, 164, 255), alpha=a2, glow=28, glow_col=(0.25, 0.18, 0.6)))
        L.append(ov.text_layer(W, H, "на каждом следующем запросе", ov.MONO, 26, (X1, 776),
                               anchor="rm", color=(160, 150, 210), alpha=a2 * 0.95, tracking=1))

    # ---------- E: batch
    if 8.90 <= t <= 10.35:
        ain = clamp01((t - 8.94) / 0.4) * (1 - clamp01((t - 10.12) / 0.24))
        L.append(ov.plate(W, H, (X0, 268, X0 + 172, 334), 20, (255, 188, 88), 0.16,
                          border=(255, 205, 120), border_alpha=0.5, glow=9, glow_col=(0.45, 0.33, 0.10)))
        L.append(ov.text_layer(W, H, "BATCH", ov.MONO, 34, (X0 + 86, 301), anchor="mm",
                               color=(255, 205, 120), alpha=ain, tracking=3))
        L.append(ov.text_layer(W, H, "асинхронные", ov.HEAD, 68, (X0, 410), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        L.append(ov.text_layer(W, H, "задачи", ov.HEAD, 68, (X0, 494), anchor="lm",
                               color=(235, 240, 250), alpha=ain))
        a2 = clamp01((t - 9.30) / 0.4) * ain
        L.append(ov.text_layer(W, H, "−50%", ov.HEAD, 148, (X1, 700), anchor="rm",
                               color=(255, 190, 90), alpha=a2, glow=28, glow_col=(0.55, 0.34, 0.05)))
        L.append(ov.text_layer(W, H, "на всё: вход, выход, кэш", ov.MONO, 26, (X1, 776),
                               anchor="rm", color=(200, 168, 120), alpha=a2 * 0.95, tracking=1))

    # ---------- F: kicker + CTA
    if t >= 10.25:
        ain = clamp01((t - 10.30) / 0.45)
        L.append(ov.text_layer(W, H, "ДОРОГОЙ НЕ ТОКЕН", ov.HEAD, 64, (W // 2, 420), anchor="mm",
                               color=(238, 242, 250), alpha=ain, tracking=1))
        L.append(ov.text_layer(W, H, "дорогой контекст", ov.HEAD, 64, (W // 2, 502), anchor="mm",
                               color=(0, 170, 255), alpha=ain, glow=22, glow_col=(0.1, 0.5, 0.9)))
        a2 = clamp01((t - 10.85) / 0.45)
        L.append(ov.text_layer(W, H, "ТАКИЕ 3D-РОЛИКИ", ov.HEAD, 56, (W // 2, 1450), anchor="mm",
                               color=(238, 242, 250), alpha=a2, tracking=1))
        L.append(ov.text_layer(W, H, "делаю сам — от сценария до рендера", ov.HEAD2, 34,
                               (W // 2, 1516), anchor="mm", color=(150, 168, 195), alpha=a2))
        L.append(ov.plate(W, H, (W // 2 - 240, 1580, W // 2 + 240, 1680), 26, (22, 26, 38), 0.55,
                          border=(78, 98, 138), border_alpha=0.42, glow=12, glow_col=(0.08, 0.22, 0.45)))
        L.append(ov.text_layer(W, H, "@your_handle", ov.MONO, 40, (W // 2, 1630), anchor="mm",
                               color=(120, 220, 255), alpha=a2, tracking=2, glow=14,
                               glow_col=(0.08, 0.35, 0.55)))

    # ---------- furniture
    ab = clamp01((t - 0.2) / 0.7) * (1 - clamp01((t - 11.5) / 0.4))
    L.append(ov.text_layer(W, H, "CLAUDE API", ov.MONO, 22, (X0, 104), anchor="lm",
                           color=(130, 150, 180), alpha=0.5 * ab, tracking=6))
    L.append(ov.text_layer(W, H, "USD / 1M TOKENS", ov.MONO, 22, (X1, 104), anchor="rm",
                           color=(130, 150, 180), alpha=0.5 * ab, tracking=4))
    L.append(ov.line(W, H, (X0, 134), (X1, 134), 2, (90, 110, 150), alpha=0.22 * ab))
    L.append(ov.progress(W, H, X0, X1, 1836, t / DUR, 0.35 * ab))
    # source note
    L.append(ov.text_layer(W, H, "цены anthropic api · ноябрь 2026", ov.MONO, 20, (X0, 1790),
                           anchor="lm", color=(110, 126, 155), alpha=0.45 * ab, tracking=2))
    return L


class Short:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.W, self.H = int(W * scale), int(H * scale)
        self.r = Renderer(self.W, self.H)
        self.post = ov.Post(self.W, self.H)

    def frame(self, t, fi=0):
        img = self.r.render(build(t))
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
    s = Short(float(os.environ.get("SCALE", "1")))
    if cmd == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            t0 = time.time()
            s.frame(t, int(t * FPS)).save("/home/user/motion/out/two_%.2f.png" % t)
            print("t=%.2f  %.2fs" % (t, time.time() - t0), flush=True)
    elif cmd == "chunk":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        outdir = os.environ.get("OUTDIR", "/tmp/frames2")
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        for fi in range(a, min(b, NF)):
            s.frame(fi / FPS, fi).save(os.path.join(outdir, "%04d.png" % fi))
            if fi % 10 == 0:
                print("frame %d/%d  %.1fs" % (fi, NF, time.time() - t0), flush=True)
        print("chunk done %.1fs" % (time.time() - t0), flush=True)
