"""3D character short — real geometry, room, characters, camera moves, captions.
13 s · 1080x1920 · 24 fps. CLI: test t,t | chunk a b | still t
"""
import os, sys, math, time
import numpy as np
from PIL import Image
import overlay as ov
from engine import Renderer, T, RX, RY, RZ, S
from engine import _compose, seg, ease, norm
from kit import Kit, M, ORANGE, HOODIE, SKIN, HAIR

W, H, FPS = 1080, 1920, 24
DUR = 13.0
NF = int(round(DUR * FPS))
OUT = os.path.dirname(os.path.abspath(__file__))

CAM_KEYS = [
    # t,    eye,                   target,               fov
    (0.0,  (0.04, 1.058, 1.92), (0.02, 1.000, -0.76), 54.0),
    (3.2,  (0.02, 1.050, 1.72), (0.01, 0.995, -0.76), 54.0),
    (5.6,  (0.05, 1.052, 1.50), (0.04, 1.010, -0.72), 52.0),
    (8.3,  (0.03, 1.050, 1.74), (0.02, 0.995, -0.76), 54.0),
    (10.9, (0.02, 1.052, 1.54), (0.00, 1.000, -0.70), 53.0),
    (13.0, (0.01, 1.054, 1.42), (-0.01, 1.005, -0.68), 52.0),
]


def camera(t):
    ks = CAM_KEYS
    if t <= ks[0][0]:
        e, g, f = ks[0][1], ks[0][2], ks[0][3]
    else:
        e = g = f = None
        for i in range(len(ks) - 1):
            t0, t1 = ks[i][0], ks[i + 1][0]
            if t0 <= t <= t1:
                p = ease((t - t0) / (t1 - t0), "inout")
                e = tuple(ks[i][1][k] + (ks[i + 1][1][k] - ks[i][1][k]) * p for k in range(3))
                g = tuple(ks[i][2][k] + (ks[i + 1][2][k] - ks[i][2][k]) * p for k in range(3))
                f = ks[i][3] + (ks[i + 1][3] - ks[i][3]) * p
                break
        if e is None:
            e, g, f = ks[-1][1], ks[-1][2], ks[-1][3]
    # beat accents: quick push-in
    for tb, amp in ((5.62, 0.055), (8.45, 0.050), (3.24, 0.030)):
        k = math.exp(-((t - tb) / 0.13) ** 2)
        e = (e[0], e[1], e[2] - amp * k)
        g = (g[0], g[1] + 0.012 * k, g[2])
    # hand-held wobble
    wob = 0.012
    e = (e[0] + wob * math.sin(t * 1.7), e[1] + wob * math.sin(t * 2.3 + 1.1), e[2] + wob * math.sin(t * 1.1 + 2.0))
    return dict(pos=e, target=g, fov=f, roll=0.004 * math.sin(t * 1.9), dof=dict(focus=1.7, aperture=0.0))


# ------------------------------------------------------------------ action
def action(t):
    """returns (guy_pos, guy_rot, guy_mood, bot_pos, bot_rot, bot_mood, phone_on)"""
    # phase A 0–3.2: sitting chill, bot typing at laptop
    # phase B 3.2–5.6: bot panics (smoke), guy still smug
    # phase C 5.6–8.3: everything shakes, red screen
    # phase D 8.3–10.9: bot proud, guy claps/celebrates
    # phase E 10.9–13: guy stares at phone, bot dead
    gp = list(GP)
    grot = [0.0, 0.62, 0.0]
    gmood = "chill"
    bp = list(BP)
    brot = [0.0, 0.42, 0.0]
    bmood = "idle"
    phone = 0.0

    if t >= 3.2:
        bmood = "panic"
    if t >= 3.2:
        groove = seg(t, 3.2, 3.9) * (1 - seg(t, 8.2, 8.5))
        grot[1] = 0.62 + 0.40 * groove
    if t >= 8.3:
        grot[1] = 1.02 - 0.40 * seg(t, 8.3, 8.7)
    if t >= 8.3:
        grot[1] = max(grot[1], 0.62)
    if t >= 5.6:
        shake_k = seg(t, 5.6, 5.9) * (1 - seg(t, 7.9, 8.2))
        gp[0] += 0.012 * shake_k * math.sin(t * 47)
        grot[1] += 0.05 * shake_k * math.sin(t * 41)
    if t >= 8.3:
        bmood = "proud"
        gmood = "smile"
        k = seg(t, 8.3, 8.7, "back")
        gmood = "smile"
    if t >= 10.9:
        gmood = "smile"
        bmood = "idle"
        k = seg(t, 10.9, 11.4)
        grot[1] = 0.62 - 0.25 * k
        phone = seg(t, 11.0, 11.3)
        bp[0] += 0.0
    return tuple(gp), tuple(grot), gmood, tuple(bp), tuple(brot), bmood, phone


# ------------------------------------------------------------------ scene build
def build(t, renderer=None):
    K = Kit()
    desk_y = K.desk_scene(t, renderer, texname=("error" if 5.6 <= t < 8.4 else ("success" if t >= 8.4 else "code")))
    gp, grot, gmood, bp, brot, bmood, phone = action(t)
    guy_base = M((GP[0], desk_y + 0.002, GP[2]), grot, (0.66, 0.66, 0.66))
    bot_base = M((BP[0], desk_y + 0.002, BP[2]), brot, (0.80, 0.80, 0.80))
    K.guy(guy_base, t, gmood, pose="stand")
    K.bot(bot_base, t, bmood)
    if phone > 0.01:
        hand = guy_base @ T(0.085, 0.735, 0.255) @ RY(0.26) @ RX(-0.18)
        K.part(hand, "rbox", (0, 0, 0), (0, 0, 0), (0.125, 0.225, 0.020), color=(0.05, 0.05, 0.07),
               rough=0.3, metal=0.4)
        K.part(hand, "plane", (0, 0, 0.016), (0, 0, 0), (0.112, 0.208, 1), color=(0.66, 0.83, 1.0),
               emissive=1.9 * phone, rough=0.2)
        K.part(hand, "rbox", (0.0, -0.135, 0.005), (0, 0, 0), (0.095, 0.070, 0.085), color=(0.96, 0.72, 0.57),
               rough=0.6)
    # smoke
    if t >= 3.2:
        k_sm = seg(t, 3.2, 3.7) * (1 - seg(t, 10.9, 11.3))
        if k_sm > 0:
            for i in range(13):
                age = (t * 0.42 + i / 13.0) % 1.0
                y = desk_y + 0.52 + age * 0.80
                x = BP[0] + 0.075 * math.sin(age * 4.2 + i * 0.9) - age * 0.06
                z = BP[2] + 0.02 + age * 0.22
                sz = 0.026 + age * 0.075
                a = (1 - age) ** 1.3 * k_sm
                if a < 0.02:
                    continue
                K.add("sphere", pos=(x, y, z), size=(sz, sz, sz), color=(0.82, 0.83, 0.87),
                      rough=1.0, emissive=0.42 * a)
    # speed lines in the error phase
    if 6.2 <= t < 8.2:
        for i in range(5):
            ph = (t * 1.6 + i * 0.37) % 1.0
            K.add("box", pos=(-1.3 + ph * 2.8, desk_y + 0.9 + 0.28 * i, -0.30 + 0.1 * i),
                  size=(0.30, 0.012, 0.012), color=(1.0, 0.45, 0.35), emissive=2.0 * (1 - ph), rough=0.5)
    return dict(
        objects=K.objs,
        camera=camera(t),
        lights=dict(list=[((0.30, 0.85, 0.45), (1.0, 0.88, 0.70)),
                          ((-0.25, 0.35, 1.0), (0.45, 0.60, 0.90)),
                          ((0.20, -0.35, -1.0), (0.45, 0.55, 0.85))],
                    key_int=2.5, fill_int=1.75, rim_int=1.9),
        ambient=dict(top=(0.10, 0.11, 0.15), bot=(0.022, 0.024, 0.034), rim=(0.30, 0.55, 0.90)),
        fog=((0.024, 0.026, 0.038), 0.115),
        bg=(0.02, 0.02, 0.03),
        bg_colors=((0.040, 0.050, 0.080), (0.006, 0.007, 0.014)),
        bg_glow=((0.18, 0.28, 0.52), (0.5, 0.40), 0.50),
        shadow_center=(0.2, 0.85, -0.7), shadow_extent=2.2,
        bloom=0.75, bloom_thresh=0.80, exposure=1.02, vignette=0.62, grain=0.013,
        time=t,
    )


GP = [-0.30, 0.0, -0.52]
BP = [0.34, 0.0, -0.66]

# ------------------------------------------------------------------ captions
CAPS = [
    (0.2, 3.2, "Клод, сделай мне сайт."),
    (3.3, 5.5, "Только без ошибок."),
    (5.6, 8.3, "Это не ошибки.\nЭто катастрофа."),
    (8.4, 10.8, "Сайт готов. Из дыма,\nно готов."),
    (10.9, 13.2, "«Ого, а я, похоже,\nреально хорош»."),
]


def caption(t):
    for t0, t1, text in CAPS:
        if t0 - 0.001 <= t <= t1:
            a = min(1.0, (t - t0) / 0.18) * min(1.0, max(0.0, (t1 - t) / 0.18))
            return text, max(0.0, min(1.0, a))
    return None, 0.0


class Short:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.W, self.H = int(W * scale), int(H * scale)
        self.r = Renderer(self.W, self.H, shadow=1024 if scale < 1 else 2048)
        self.post = ov.Post(self.W, self.H)

    def frame(self, t, fi=0):
        sc = build(t, self.r)
        img = self.r.render(sc)
        canvas = np.asarray(img).astype(np.float32) / 255.0
        canvas = self.post.bloom(canvas, amount=0.18, thresh=0.78, radius=14)
        canvas = self.post.finish(canvas, fi, grain=0.014, vig=0.60)
        s = self.scale
        text, a = caption(t)
        if text and a > 0.004:
            for i, line in enumerate(text.split("\n")):
                lay = ov.text_layer(self.W, self.H, line, ov.HEAD2, int(60 * s),
                                    (self.W / 2, (1690 + i * 74) * s), anchor="mm",
                                    color=(255, 252, 246), alpha=a, shadow=0.9)
                canvas = ov.paste_rgba(canvas, lay)
        return self.post.to_image(canvas)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    scale = float(os.environ.get("SCALE", "1"))
    if cmd == "test":
        sh = Short(scale)
        os.makedirs(os.path.join(OUT, "out"), exist_ok=True)
        for t in [float(x) for x in sys.argv[2].split(",")]:
            t0 = time.time()
            sh.frame(t, int(t * FPS)).save(os.path.join(OUT, "out", "s_%.2f.png" % t))
            print("t=%.2f %.2fs" % (t, time.time() - t0), flush=True)
    elif cmd in ("chunk", "still"):
        sh = Short(scale)
        if cmd == "chunk":
            a, b = int(sys.argv[2]), int(sys.argv[3])
            outdir = os.environ.get("OUTDIR", "/tmp/frames3")
            os.makedirs(outdir, exist_ok=True)
            t0 = time.time()
            for fi in range(a, min(b, NF)):
                sh.frame(fi / FPS, fi).save(os.path.join(outdir, "%04d.png" % fi))
                if fi % 8 == 0:
                    print("frame %d/%d %.1fs" % (fi, NF, time.time() - t0), flush=True)
            print("done %.1fs" % (time.time() - t0), flush=True)
