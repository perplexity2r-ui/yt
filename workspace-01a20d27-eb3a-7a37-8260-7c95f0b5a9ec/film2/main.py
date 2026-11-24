"""Character short #1 — «Клод, сделай мне сайт. Только без ошибок.»
Style: illustrated key art + ken-burns camera + comic speech bubbles (like the reference).
13 s, 1080x1920, 24 fps.
"""
import os, sys, math, json, time
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import overlay as ov

W, H, FPS, DUR = 1080, 1920, 24, 13.0
NF = int(round(DUR * FPS))
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "art")

F = dict(
    head="Mont-ExtraBold.ttf", head2="Inter-Bold.ttf", mono="JetBrainsMono-Bold.ttf",
    semi="Mont-SemiBold.ttf",
)
BUBBLE_RGB = (16, 18, 26)


# ------------------------------------------------------------------ utils
def clamp01(x): return 0.0 if x < 0 else (1.0 if x > 1 else x)

def ease_out(p):
    p = clamp01(p); return 1 - (1 - p) ** 3

def ease_io(p):
    p = clamp01(p); return p * p * (3 - 2 * p)

def ease_back(p, s=1.7):
    p = clamp01(p) - 1.0
    return 1 + (s + 1) * p ** 3 + s * p ** 2

def span(t, t0, t1, kind="out"):
    if t < t0: return 0.0
    if t > t1: return 1.0
    p = (t - t0) / max(t1 - t0, 1e-6)
    return ease_io(p) if kind == "io" else ease_out(p)

def alpha_span(t, t0, t1, fin=0.25, fout=0.25):
    return clamp01((t - t0) / fin) * (1 - clamp01((t - (t1 - fout)) / fout))


# ------------------------------------------------------------------ art loading
class Art:
    def __init__(self, name, W, H):
        im = Image.open(os.path.join(ART, name)).convert("RGB")
        self.src = im
        self.W, self.H = W, H
        self.cover_scale = max(W / im.width, H / im.height)
        self._cache = {}

    def frame(self, zoom, cx, cy, shake=(0.0, 0.0), blur=0.0):
        """zoom>=1, cx/cy in 0..1 of the source image (focus point)"""
        key = (round(zoom, 3), round(cx, 3), round(cy, 3), round(shake[0], 1), round(shake[1], 1))
        if key in self._cache:
            return self._cache[key]
        base_w = self.src.width * self.cover_scale * zoom
        base_h = self.src.height * self.cover_scale * zoom
        im = self.src.resize((max(self.W, int(base_w)), max(self.H, int(base_h))), Image.LANCZOS)
        # crop around focus
        fx = im.width * cx - self.W / 2 + shake[0]
        fy = im.height * cy - self.H / 2 + shake[1]
        fx = min(max(fx, 0), im.width - self.W)
        fy = min(max(fy, 0), im.height - self.H)
        out = im.crop((int(fx), int(fy), int(fx) + self.W, int(fy) + self.H))
        if blur > 0:
            out = out.filter(ImageFilter.GaussianBlur(blur))
        if len(self._cache) > 90:
            self._cache.clear()
        self._cache[key] = out
        return out


# ------------------------------------------------------------------ bubbles
def bubble(w, h, text, size=40, tail="bl", accent=(255, 168, 92), name=None, name_col=None,
           alpha=1.0, fill=BUBBLE_RGB, border=None, font=None, glow_col=(1.0, 0.62, 0.25),
           pad=(38, 30), radius=34, align="center", shadow=True):
    """rounded speech bubble as RGBA float layer; tail at bl/br/left/right"""
    font = font or F["semi"]
    txt_col = (250, 248, 244)
    # measure text
    tmp = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(tmp)
    lines = text.split("\n")
    fnt = ov.font if False else None
    from overlay import font as getfont
    ff = getfont(font, size)
    tw = max(d.textlength(l, font=ff) for l in lines)
    lh = int(size * 1.28)
    th = lh * len(lines)
    nw = 0
    if name:
        nf = getfont(F["head"], int(size * 0.92))
        nw = d.textlength(name, font=nf) + int(size * 0.8)
    cw = int(max(tw, nw if name else 0) + pad[0] * 2)
    ch = int(th + pad[1] * 2 + (int(size * 1.15) if name else 0))
    tail_h = 26
    img = Image.new("RGBA", (cw + 8, ch + tail_h + 8), (0, 0, 0, 0))
    dd = ImageDraw.Draw(img)
    r = radius
    dd.rounded_rectangle([4, 4, 4 + cw, 4 + ch], radius=r, fill=fill + (255,),
                         outline=(border or accent) + (255,), width=3)
    # tail
    if tail == "bl":
        pts = [(int(cw * 0.26), 4 + ch - 2), (int(cw * 0.26) + 34, 4 + ch - 2), (int(cw * 0.20), 4 + ch + tail_h)]
    elif tail == "br":
        pts = [(int(cw * 0.66), 4 + ch - 2), (int(cw * 0.66) + 34, 4 + ch - 2), (int(cw * 0.74), 4 + ch + tail_h)]
    elif tail == "left":
        pts = [(4, int(ch * 0.5)), (4, int(ch * 0.5) + 34), (4 - tail_h, int(ch * 0.5) + 8)]
    else:
        return None
    dd.polygon(pts, fill=fill + (255,), outline=(border or accent) + (255,))
    # name row
    y0 = 4 + pad[1]
    if name:
        nf = getfont(F["head"], int(size * 0.92))
        dd.text((4 + pad[0], y0), name, font=nf, fill=(name_col or accent) + (255,))
        y0 += int(size * 1.15)
    for i, line in enumerate(lines):
        if align == "center":
            x = 4 + (cw - d.textlength(line, font=ff)) / 2
        else:
            x = 4 + pad[0]
        dd.text((x, y0 + i * lh), line, font=ff, fill=txt_col + (255,))
    arr = np.asarray(img).astype(np.float32)
    if shadow:
        sh = np.asarray(img.filter(ImageFilter.GaussianBlur(14))).astype(np.float32)
        sh[..., :3] = 0
        sh[..., 3] *= 0.5
        out = sh
        a = np.clip(arr[..., 3:] / 255.0, 0, 1)
        out[..., :3] = arr[..., :3] * a + out[..., :3] * (1 - a)
        out[..., 3:] = np.clip(arr[..., 3:] + out[..., 3:] * (1 - a), 0, 255)
        arr = out
    if glow_col is not None:
        g = np.asarray(img.filter(ImageFilter.GaussianBlur(26))).astype(np.float32)
        inside = np.clip(arr[..., 3:] / 255.0, 0, 1)
        ga = np.clip(g[..., 3:] / 255.0, 0, 1) * (1.0 - inside) * 0.85
        arr[..., :3] = np.clip(arr[..., :3] + np.array(glow_col, np.float32)[None, None] * 255 * ga, 0, 255)
        arr[..., 3:] = np.clip(arr[..., 3:] + 255 * ga, 0, 255)
    if alpha < 1.0:
        arr[..., 3] *= alpha
    return arr


def place(canvas, layer, x, y, scale=1.0, alpha=1.0, anchor="tl"):
    """composite RGBA float layer onto float canvas (0..1) at pixel pos"""
    h, w = layer.shape[:2]
    if scale != 1.0:
        im = Image.fromarray(layer.astype(np.uint8)).resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        layer = np.asarray(im).astype(np.float32)
        h, w = layer.shape[:2]
    if anchor[0] == "c": x -= w // 2
    elif anchor[0] == "r": x -= w
    if anchor[1] == "c": y -= h // 2
    elif anchor[1] == "b": y -= h
    x, y = int(x), int(y)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0: return canvas
    sub = layer[y0 - y:y1 - y, x0 - x:x1 - x]
    a = np.clip(sub[..., 3:] / 255.0, 0, 1) * alpha
    dst = canvas[y0:y1, x0:x1]
    dst *= (1 - a)
    dst += (sub[..., :3] / 255.0) * a
    return canvas


def txt(canvas, text, x, y, size, fname, color=(245, 245, 250), alpha=1.0, anchor="lt",
        glow=0.0, glow_col=None, tracking=0.0, shadow=1.0):
    lay = ov.text_layer(W, H, text, fname, size, (x, y), anchor=anchor, color=color, alpha=alpha,
                        tracking=tracking, glow=glow, glow_col=glow_col, shadow=shadow)
    return ov.paste_rgba(canvas, lay)


def lower_third(t, text, icon=True, t0=0.0):
    """rounded caption card at the bottom, like the reference"""
    a = alpha_span(t, t0, t0 + 99, 0.35, 0.01)
    if a <= 0.003: return None
    pad = 30
    from overlay import font as getfont
    ff = getfont(F["semi"], 31)
    lines = text.split("\n")
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tw = max(tmp.textlength(l, font=ff) for l in lines)
    lh = 42
    cw, ch = int(tw + pad * 2 + (58 if icon else 0)), int(lh * len(lines) + pad * 2)
    img = Image.new("RGBA", (cw + 10, ch + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([5, 5, 5 + cw, 5 + ch], radius=30, fill=(14, 16, 24, 235),
                        outline=(255, 176, 96, 190), width=2)
    x0 = 5 + pad
    if icon:
        d.rounded_rectangle([x0 - 6, 5 + pad + 2, x0 + 40, 5 + pad + 48], radius=10,
                            fill=(255, 176, 96, 235))
        d.polygon([(x0 + 12, 5 + pad + 14), (x0 + 12, 5 + pad + 36), (x0 + 30, 5 + pad + 25)],
                  fill=(20, 20, 26, 255))
        x0 += 58
    for i, line in enumerate(lines):
        d.text((x0, 5 + pad + i * lh), line, font=ff, fill=(250, 248, 244, 255))
    arr = np.asarray(img).astype(np.float32)
    arr[..., 3] *= a
    return arr


# ------------------------------------------------------------------ timeline
SCENES = [
    # name, start, end, zoom_from, zoom_to, focus path
    ("scene1_base.png", 0.0, 3.3, 1.02, 1.13, (0.50, 0.50), (0.46, 0.44)),
    ("scene2_robot.png", 3.2, 5.6, 1.06, 1.16, (0.50, 0.45), (0.50, 0.40)),
    ("scene3_errors.png", 5.5, 8.4, 1.05, 1.18, (0.50, 0.45), (0.50, 0.38)),
    ("scene4_success.png", 8.3, 11.0, 1.30, 1.02, (0.62, 0.42), (0.55, 0.45)),
    ("scene5_phone.png", 10.9, 13.0, 1.06, 1.15, (0.50, 0.45), (0.52, 0.42)),
]


class Short:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.W, self.H = int(W * scale), int(H * scale)
        self.arts = {n: Art(n, self.W, self.H) for n, *_ in SCENES}
        self.post = ov.Post(self.W, self.H)

    def base(self, t):
        # find active scene (crossfade 0.4s)
        for i, (name, t0, t1, z0, z1, f0, f1) in enumerate(SCENES):
            if t0 <= t < t1:
                p = (t - t0) / (t1 - t0)
                # shake on the error scene
                shake = (0, 0)
                if name == "scene3_errors.png":
                    amp = 5.0 * clamp01((t - 6.6) / 0.4) * (1 - clamp01((t - 7.9) / 0.3))
                    shake = (amp * math.sin(t * 51.0), amp * math.sin(t * 43.0 + 1.3))
                z = z0 + (z1 - z0) * ease_io(p)
                cx = f0[0] + (f1[0] - f0[0]) * ease_io(p)
                cy = f0[1] + (f1[1] - f0[1]) * ease_io(p)
                cur = self.arts[name].frame(z, cx, cy, shake)
                # crossfade from previous
                if i > 0 and t < t0 + 0.42:
                    k = clamp01((t - t0) / 0.42)
                    pn, p0, p1, pz0, pz1, pf0, pf1 = SCENES[i - 1]
                    pp = (t - p0) / (p1 - p0)
                    pz = pz0 + (pz1 - pz0) * ease_io(pp)
                    pcx = pf0[0] + (pf1[0] - pf0[0]) * ease_io(pp)
                    pcy = pf0[1] + (pf1[1] - pf0[1]) * ease_io(pp)
                    prev = self.arts[pn].frame(pz, pcx, pcy)
                    # directional slide+fade
                    arr = np.asarray(cur).astype(np.float32) * ease_io(k)
                    pa = np.asarray(prev).astype(np.float32) * (1 - ease_io(k))
                    return (pa + arr).astype(np.uint8)
                return cur
        return None

    def frame(self, t, fi=0):
        img = self.base(t)
        canvas = np.asarray(img).astype(np.float32) / 255.0
        L = self.overlays(t)
        for (layer, x, y, sc, a, anchor) in L:
            canvas = place(canvas, layer, x * self.scale, y * self.scale, sc * self.scale, a, anchor)
        canvas = self.post.bloom(canvas, amount=0.16, thresh=0.80, radius=14)
        canvas = self.post.finish(canvas, fi, grain=0.016, vig=0.55)
        return self.post.to_image(canvas)

    # -------------------------------------------------------------- overlays
    def overlays(self, t):
        L = []
        s = self.scale

        # ---- opening hook
        a = alpha_span(t, 0.15, 1.15, 0.3, 0.25)
        if a > 0.003:
            L.append((bubble(0, 0, "Клод, сделай мне сайт.\nТолько без ошибок.", size=44, name="ТЫ",
                             accent=(255, 176, 96), glow_col=None), 540, 210, 1.0, a, "ct"))
        # ---- claude accepted
        a = alpha_span(t, 1.35, 3.35, 0.28, 0.22)
        if a > 0.003:
            L.append((bubble(0, 0, "Принято.\nДелаю сайт…", size=42, name="Claude",
                             accent=(255, 150, 70), name_col=(255, 176, 96)), 620, 300, 1.0, a, "ct"))
        # thinking dots
        if 2.0 <= t <= 3.4:
            dots = int((t - 2.0) / 0.22) % 3 + 1
            L.append((bubble(0, 0, "• " * dots, size=44, accent=(255, 176, 96)), 600, 520, 1.0,
                      alpha_span(t, 2.0, 3.4, 0.2, 0.2), "ct"))

        # ---- checklist (scene 2)
        items = [("Структура", 3.55), ("Дизайн", 3.80), ("Код", 4.05), ("Тесты", 4.30), ("Деплой", 4.55)]
        if 3.50 <= t <= 5.6:
            box_a = alpha_span(t, 3.50, 5.6, 0.3, 0.2)
            L.append((checklist(items, t), 96, 470, 1.0, box_a, "lt"))
        # ---- claude sweats
        a = alpha_span(t, 3.9, 5.55, 0.3, 0.25)
        if a > 0.003:
            L.append((bubble(0, 0, "Тут чуть сложнее…", size=38, name="Claude",
                             accent=(255, 150, 70), name_col=(255, 176, 96)), 640, 215, 1.0, a, "ct"))

        # ---- errors
        a = alpha_span(t, 5.95, 8.3, 0.3, 0.25)
        if a > 0.003:
            L.append((bubble(0, 0, "Ещё немного поправь.", size=40, name="ТЫ",
                             accent=(255, 176, 96)), 660, 200, 1.0, a, "ct"))
            L.append((bubble(0, 0, "…", size=40, name="Claude", accent=(255, 120, 90),
                             name_col=(255, 140, 100), border=(255, 120, 90),
                             glow_col=(1.0, 0.3, 0.2)), 600, 400, 1.0,
                      a * clamp01((t - 6.25) / 0.3), "ct"))

        # ---- success
        a = alpha_span(t, 8.55, 11.0, 0.3, 0.25)
        if a > 0.003:
            L.append((bubble(0, 0, "Сайт готов!", size=54, name=None, accent=(90, 240, 170),
                             border=(90, 240, 170), glow_col=(0.2, 1.0, 0.6)), 540, 175, 1.0, a, "ct"))
            a2 = a * clamp01((t - 9.1) / 0.3)
            L.append((bubble(0, 0, "Вы абсолютно правы.", size=38, name="Claude",
                             accent=(255, 150, 70), name_col=(255, 176, 96)), 560, 400, 1.0, a2, "ct"))

        # ---- phone / smug
        a = alpha_span(t, 11.05, 13.05, 0.3, 0.01)
        if a > 0.003:
            L.append((bubble(0, 0, "«Ещё немного поправь.»", size=40, name="ТЫ",
                             accent=(255, 176, 96)), 600, 205, 1.0, a, "ct"))
            L.append((bubble(0, 0, "Ого, а я, похоже,\nреально хорош.", size=44, accent=(120, 220, 255),
                             border=(120, 220, 255), glow_col=(0.2, 0.7, 1.0)), 540, 1500, 1.0,
                      a * clamp01((t - 11.6) / 0.35), "ct"))

        # ---- captions
        cap = None
        if 0.4 <= t <= 3.3:
            cap = ("Попросил Клода сделать сайт,\nзакинул ноги на стол и залип в телефоне.", 0.4)
        elif 3.4 <= t <= 5.6:
            cap = ("Потом пару раз сказал «сделай получше».\nКлод начал дымиться.", 3.4)
        elif 5.7 <= t <= 8.4:
            cap = ("Сайт заработал. Из дыма, но заработал.", 5.7)
        elif 8.5 <= t <= 11.0:
            cap = ("И я подумал: «Ого, а я, похоже,\nреально хорош».", 8.5)
        elif 11.0 <= t:
            cap = ("Клод: «Вы абсолютно правы».", 11.0)
        if cap:
            lay = lower_third(t, cap[0], True, cap[1])
            if lay is not None:
                L.append((lay, 540, 1690, 1.0, 1.0, "cb"))

        return L


def checklist(items, t):
    """checkbox list card with a pop-in tick per item"""
    from overlay import font as getfont
    pad, lh = 26, 62
    ff = getfont(F["semi"], 34)
    cw = 300
    ch = pad * 2 + lh * len(items)
    img = Image.new("RGBA", (cw + 10, ch + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([5, 5, 5 + cw, 5 + ch], radius=24, fill=(14, 16, 24, 225),
                        outline=(255, 176, 96, 150), width=2)
    for i, (name, t0) in enumerate(items):
        y = 5 + pad + i * lh
        d.rounded_rectangle([5 + pad - 4, y + 8, 5 + pad + 32, y + 44], radius=8,
                            outline=(120, 130, 150, 255), width=3)
        if t >= t0:
            k = clamp01((t - t0) / 0.22)
            col = (90, 240, 170, int(255 * k))
            d.line([(5 + pad + 4, y + 26), (5 + pad + 13, y + 36), (5 + pad + 28, y + 14)],
                   fill=col, width=6)
        a = clamp01((t - (t0 - 0.5)) / 0.3)
        d.text((5 + pad + 48, y + 12), name, font=ff, fill=(250, 248, 244, int(255 * a)))
    return np.asarray(img).astype(np.float32)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    sh = Short(float(os.environ.get("SCALE", "1")))
    if cmd == "test":
        for t in [float(x) for x in sys.argv[2].split(",")]:
            t0 = time.time()
            sh.frame(t, int(t * FPS)).save("/home/user/film2/out/t_%.2f.png" % t)
            print("t=%.2f %.2fs" % (t, time.time() - t0), flush=True)
    elif cmd == "chunk":
        a, b = int(sys.argv[2]), int(sys.argv[3])
        outdir = os.environ.get("OUTDIR", "/tmp/f2")
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        for fi in range(a, min(b, NF)):
            sh.frame(fi / FPS, fi).save(os.path.join(outdir, "%04d.png" % fi))
            if fi % 12 == 0:
                print("frame %d/%d %.1fs" % (fi, NF, time.time() - t0), flush=True)
        print("done %.1fs" % (time.time() - t0), flush=True)
