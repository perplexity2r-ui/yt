"""2D overlay layer: typography, plates, text masks, bloom, vignette, grain."""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = "/home/user/assets/fonts"

_FONT_CACHE = {}

def font(name, size):
    key = (name, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(os.path.join(FONTS, name), size)
    return _FONT_CACHE[key]

WHITE = (255, 255, 255)
MONO = "JetBrainsMono-Bold.ttf"
HEAD = "Mont-ExtraBold.ttf"
HEAD2 = "Inter-Bold.ttf"


# ------------------------------------------------------------------ primitives
def text_layer(W, H, text, fname, size, pos, anchor="mm", color=WHITE, alpha=1.0,
               tracking=0.0, glow=0.0, glow_col=None, shadow=0.0, blur=0.0, scale=1.0):
    """Returns an RGBA numpy layer with rendered text."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if alpha <= 0.003 or size * scale < 1:
        return np.asarray(img)
    f = font(fname, max(1, int(round(size * scale))))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if tracking and tracking != 0:
        widths = [d.textlength(ch, font=f) for ch in text]
        total = sum(widths) + tracking * max(len(text) - 1, 0)
        x = pos[0]
        if anchor[0] == "m": x -= total / 2
        elif anchor[0] == "r": x -= total
        for ch, w in zip(text, widths):
            d.text((x, pos[1]), ch, font=f, fill=color + (255,), anchor="l" + anchor[1])
            x += w + tracking
    else:
        d.text(pos, text, font=f, fill=color + (255,), anchor=anchor)
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    if shadow > 0:
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sh.paste(layer.filter(ImageFilter.GaussianBlur(shadow * 0.55)), (0, int(shadow * 0.55)),
                 layer.filter(ImageFilter.GaussianBlur(shadow * 0.55)))
        arr = np.asarray(sh).astype(np.float32) * 0.55
        img = Image.fromarray(arr.astype(np.uint8))
    if glow > 0:
        g = layer.filter(ImageFilter.GaussianBlur(glow))
        arr = np.asarray(g).astype(np.float32)
        gc = np.array((glow_col or color), np.float32)
        aa = arr[..., 3:4] / 255.0
        g2 = np.concatenate([aa * gc[None, None, :], aa * 255.0], axis=2)
        base = np.asarray(img).astype(np.float32) / 255.0
        gl = g2 / 255.0
        comp = base + gl * 0.85 * (1 - base[..., 3:4])
        base = np.concatenate([np.clip(comp[..., :3], 0, 1), np.clip(comp[..., 3:4], 0, 1)], axis=2)
        img = Image.fromarray((base * 255).astype(np.uint8))
    out = np.asarray(layer).astype(np.float32)
    base = np.asarray(img).astype(np.float32)
    # composite text over glow/shadow
    sa = out[..., 3:4] / 255.0
    ba = base[..., 3:4] / 255.0
    rgb = out[..., :3] * sa + base[..., :3] * ba * (1 - sa)
    a = sa + ba * (1 - sa)
    rgb = np.where(a > 1e-4, rgb / np.maximum(a, 1e-4), 0)
    o = np.concatenate([rgb, a * 255.0], axis=2)
    if alpha < 1.0:
        o[..., 3] *= alpha
    return o

def mask_from_layer(arr, W, H):
    a = np.clip(arr[..., 3:4] / 255.0, 0, 1)
    b = a.reshape(H // 4, 4, W // 4, 4).mean(axis=(1, 3))
    b = np.asarray(Image.fromarray((np.clip(b, 0, 1)[..., 0] * 255).astype(np.uint8))
                   .resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.2))
                   ).astype(np.float32) / 255.0
    return b

def paste_rgba(canvas, arr):
    """canvas: float32 HxWx3 in 0..1 ; arr: float RGBA layer"""
    a = np.clip(arr[..., 3:4] / 255.0, 0, 1)
    rgb = arr[..., :3] / 255.0
    canvas *= (1 - a)
    canvas += rgb * a
    return canvas

def paste_glow(canvas, arr, strength=1.0):
    """Additive glow from a layer's alpha."""
    a = np.clip(arr[..., 3:4] / 255.0, 0, 1)
    rgb = arr[..., :3] / 255.0
    canvas += rgb * a * strength
    return canvas

def plate(W, H, box, radius, color, alpha, glow=0.0, glow_col=None, border=None,
          border_alpha=0.35, border_w=2, grad=None):
    """Rounded rectangle plate with optional soft glow and border.
    box = (x0, y0, x1, y1)"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if x1 <= x0 or y1 <= y0: return np.zeros((H, W, 4), np.float32)
    pad = int(max(glow * 3, 8))
    w = x1 - x0 + pad * 2; h = y1 - y0 + pad * 2
    plate_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(plate_img)
    r = int(radius)
    color = tuple(int(c) for c in color)
    d.rounded_rectangle([pad, pad, pad + (x1 - x0) - 1, pad + (y1 - y0) - 1], radius=r,
                        fill=color + (int(255 * alpha),))
    arr = np.asarray(plate_img).astype(np.float32)
    if grad is not None:
        # vertical gradient multiply
        gcols = np.linspace(0, 1, h)[:, None, None]
        top = np.array(grad[0], np.float32) * 255
        bot = np.array(grad[1], np.float32) * 255
        gimg = top[None, None, :] * (1 - gcols) + bot[None, None, :] * gcols
        rgb = arr[..., :3] / 255.0
        arr[..., :3] = np.clip(gimg * 1.0, 0, 255)
        arr[..., 3] *= 1.0
    out = np.zeros((H, W, 4), np.float32)
    cx0 = max(0, x0 - pad); cy0 = max(0, y0 - pad)
    sx0 = max(0, -(x0 - pad)); sy0 = max(0, -(y0 - pad))
    cw = min(W - cx0, w - sx0); ch = min(H - cy0, h - sy0)
    if cw <= 0 or ch <= 0: return out
    out[cy0:cy0 + ch, cx0:cx0 + cw] = arr[sy0:sy0 + ch, sx0:sx0 + cw]
    if glow > 0:
        ga = Image.fromarray((out[..., 3]).astype(np.uint8)).filter(ImageFilter.GaussianBlur(glow))
        a = np.asarray(ga).astype(np.float32) / 255.0
        gc = np.array(glow_col or color, np.float32)
        out[..., :3] = np.maximum(out[..., :3], gc[None, None, :] * 255 * a[..., None] * 0.9)
        out[..., 3] = np.maximum(out[..., 3], a * 255 * 0.9)
    if border is not None:
        bimg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        db = ImageDraw.Draw(bimg)
        db.rounded_rectangle([pad, pad, pad + (x1 - x0) - 1, pad + (y1 - y0) - 1], radius=r,
                             outline=tuple(int(c) for c in border) + (int(255 * border_alpha),),
                             width=border_w)
        barr = np.asarray(bimg).astype(np.float32)
        sub = out[cy0:cy0 + ch, cx0:cx0 + cw]
        bs = barr[sy0:sy0 + ch, sx0:sx0 + cw]
        a = np.clip(bs[..., 3:4] / 255.0, 0, 1)
        sub[..., :3] = sub[..., :3] * (1 - a) + bs[..., :3] * a
        sub[..., 3:4] = np.clip(sub[..., 3:4] + bs[..., 3:4] * (1 - sub[..., 3:4] / 255.0), 0, 255)
    return out

def line(W, H, p0, p1, width, color, alpha=1.0, glow=0.0):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line([p0, p1], fill=tuple(int(c) for c in color) + (int(255 * alpha),), width=int(width))
    if glow > 0:
        g = img.filter(ImageFilter.GaussianBlur(glow))
        arr = np.asarray(g).astype(np.float32)
        a = arr[..., 3]
        img2 = Image.fromarray(np.dstack([np.where(a[..., None] > 0, np.array(color, np.float32)[None, None, :] * 255, 0), a]).astype(np.uint8))
        arr1 = np.asarray(img).astype(np.float32)
        arr2 = np.asarray(img2).astype(np.float32)
        out = np.zeros_like(arr1)
        out[..., :3] = arr1[..., :3] + arr2[..., :3] * 0.8
        out[..., 3] = np.clip(arr1[..., 3] + arr2[..., 3] * 0.8, 0, 255)
        return out
    return np.asarray(img).astype(np.float32)


class Post:
    """Cheap numpy/PIL post: bloom, vignette, grain, grade."""
    def __init__(self, W, H, rng_seed=7):
        self.W, self.H = W, H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        u = (xx / W - 0.5) * 2; v = (yy / H - 0.5) * 2
        self.vignette = np.clip(1.0 - 0.42 * (u ** 2 * 0.75 + v ** 2 * 0.55), 0.0, 1.0)
        self.vignette = self.vignette[..., None]
        rng = np.random.default_rng(rng_seed)
        self.grain = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
        self.grain = [np.asarray(Image.fromarray(((g * 30) + 128).astype(np.uint8))
                                 .resize((W, H), Image.BILINEAR)).astype(np.float32) for g in self.grain]
        # subtle chromatic + scanline pattern
        self.scan = (0.985 + 0.015 * (((yy.astype(np.int32) % 3) == 0).astype(np.float32)))[..., None]

    def bloom(self, canvas, amount=0.55, thresh=0.75, radius=26):
        if amount <= 0: return canvas
        small = Image.fromarray((np.clip(canvas, 0, 40) * 63.75).astype(np.uint8)).resize(
            (self.W // 4, self.H // 4), Image.BILINEAR)
        arr = np.asarray(small).astype(np.float32) / 255.0 * 4.0
        lum = arr.max(axis=2, keepdims=True)
        k = np.clip((lum - thresh) / np.maximum(lum, 1e-3), 0, 1)
        bright = arr * k
        b = Image.fromarray((np.clip(bright / 4.0, 0, 1) * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius))
        bb = np.asarray(b.resize((self.W, self.H), Image.BILINEAR)).astype(np.float32) / 255.0 * 4.0
        return canvas + bb * amount

    def finish(self, canvas, frame_idx=0, grain=0.018, vig=1.0, scan=0.0):
        c = canvas * self.vignette if vig >= 1.0 else canvas * (1.0 + (self.vignette - 1.0) * vig)
        if scan > 0:
            c = c * (1.0 + (self.scan - 1.0) * scan)
        g = self.grain[frame_idx % len(self.grain)]
        c = c + ((g - 128.0) / 255.0)[..., None] * grain * 2.2
        return c

    def to_image(self, canvas):
        c = np.clip(canvas, 0, 1)
        c = np.power(c, 1 / 1.06)
        return Image.fromarray((c * 255).astype(np.uint8))


def meter(W, H, pos, width, frac, ticks, color, alpha=1.0, label=None, height=14):
    """token meter: segmented bar + label. pos = (x_right, y_baseline)"""
    if alpha <= 0.003: return np.zeros((H, W, 4), np.float32)
    x1, y = pos
    x0 = x1 - width
    seg = width / ticks
    out = np.zeros((H, W, 4), np.float32)
    for i in range(ticks):
        a = clampf((frac * ticks) - i, 0, 1)
        if a <= 0:
            col = np.array((70, 78, 96), np.float32)
            aa = 0.35
        else:
            col = np.array(color, np.float32) * 255
            aa = 0.35 + 0.65 * a
        bx0 = int(x0 + i * seg + 2); bx1 = int(x0 + (i + 1) * seg - 2)
        by0 = int(y - height); by1 = int(y)
        region = out[by0:by1, bx0:bx1]
        g = np.linspace(1.0, 0.55, region.shape[0])[:, None, None]
        region[..., :3] = col[None, None, :] * g
        region[..., 3] = 255 * aa * alpha
    if label:
        lab = text_layer(W, H, label, MONO, 22, (x0, y - height - 16), anchor="lm",
                         color=(130, 150, 180), alpha=0.8 * alpha, tracking=4)
        out = _over(out, lab)
    return out


def progress(W, H, x0, x1, y, frac, alpha=1.0):
    if alpha <= 0.003: return np.zeros((H, W, 4), np.float32)
    out = np.zeros((H, W, 4), np.float32)
    out[int(y):int(y) + 3, int(x0):int(x1)] = np.array([70, 82, 108, 60], np.float32) * np.array([1, 1, 1, alpha])
    xe = x0 + (x1 - x0) * max(0.0, min(1.0, frac))
    out[int(y):int(y) + 3, int(x0):int(xe)] = np.array([120, 220, 255, 235], np.float32) * np.array([1, 1, 1, alpha])
    return out


def clampf(x, a, b):
    return a if x < a else (b if x > b else x)


def _over(base, top):
    a = np.clip(top[..., 3:4] / 255.0, 0, 1)
    b = np.clip(base[..., 3:4] / 255.0, 0, 1)
    rgb = top[..., :3] * a + base[..., :3] * b * (1 - a)
    al = a + b * (1 - a)
    rgb = np.where(al > 1e-4, rgb / np.maximum(al, 1e-4), 0)
    return np.concatenate([rgb, al * 255.0], axis=2)
