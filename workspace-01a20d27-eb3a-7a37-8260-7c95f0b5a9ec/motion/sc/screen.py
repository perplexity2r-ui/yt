"""PIL textures for the laptop screen (code editor / error / success)."""
import numpy as np
from PIL import Image, ImageDraw
import overlay as ov

W, H = 900, 520
BG = (17, 20, 27)
BAR = (26, 30, 39)
FG = (215, 222, 235)
DIM = (110, 122, 142)
CYAN = (110, 215, 250)
ORANGE = (245, 165, 80)
GREEN = (120, 225, 160)
RED = (255, 110, 95)
VIOLET = (185, 160, 255)


def _frame(title="main.tsx", tint=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 46], fill=BAR)
    for i, c in enumerate(((255, 96, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([16 + i * 22, 16, 30 + i * 22, 30], fill=c)
    f = ov.font(ov.MONO, 19)
    d.text((W / 2, 23), title, font=f, fill=DIM, anchor="mm")
    if tint:
        img = Image.blend(img, Image.new("RGB", (W, H), tint), 0.10)
        d = ImageDraw.Draw(img)
    return img, d


def _lines(d, rows, y0=70, lh=30, x=36):
    """rows: list of (indent, [(text, color), ...])"""
    f = ov.font(ov.MONO, 20)
    for i, (ind, parts) in enumerate(rows):
        x_ = x + ind * 26
        y = y0 + i * lh
        for txt, col in parts:
            d.text((x_, y), txt, font=f, fill=col)
            x_ += d.textlength(txt, font=f)


def code():
    img, d = _frame("workspace.tsx")
    fh = ov.font(ov.HEAD2, 26)
    fm = ov.font(ov.MONO, 18)
    d.text((36, 62), "Работаем над проектом", font=fh, fill=FG)
    d.text((36, 96), "src / app / workspace", font=fm, fill=DIM)
    d.rectangle([36, 122, 40, 122 + 300], fill=CYAN)
    rows = [
        (0, [("export function ", VIOLET), ("Workspace", CYAN), ("() {", FG)]),
        (1, [("return (", FG)]),
        (2, [("<ProjectBoard", CYAN)]),
        (3, [("tasks", ORANGE), ("={currentTasks}", FG)]),
        (4, [("suggest", ORANGE), ("={suggestions}", FG)]),
        (2, [("/>", CYAN)]),
        (1, [(")", FG)]),
        (0, [("}", FG)]),
        (0, [("", FG)]),
        (0, [("// проверка сборки", DIM)]),
        (0, [("npm run build", GREEN)]),
    ]
    f = ov.font(ov.MONO, 20)
    for i, (ind, parts) in enumerate(rows):
        x_ = 62 + ind * 26
        y = 132 + i * 29
        for txt, col in parts:
            d.text((x_, y), txt, font=f, fill=col)
            x_ += d.textlength(txt, font=f)
    # cursor
    d.rectangle([62, 132 + 10 * 29, 72, 132 + 10 * 29 + 22], fill=CYAN)
    return img


def error():
    img, d = _frame("terminal — build", tint=(255, 90, 70))
    fh = ov.font(ov.HEAD2, 24)
    fm = ov.font(ov.MONO, 19)
    d.rectangle([36, 62, W - 36, 108], fill=(60, 22, 20))
    d.text((48, 74), "!!  СБОРКА НЕ УДАЛАСЬ", font=fh, fill=(255, 150, 135))
    rows = ["TypeError: undefined is not a function", "  at render (workspace.tsx:41:9)",
            "  at mount (react-dom.js:1120:5)", "!! 12 ошибок в 4 файлах", "",
            "error TS2339: Property 'tasks' does not exist", "error TS2554: Expected 2 arguments",
            "error: Cannot read properties of null", "!! build failed in 3.42s"]
    for i, r in enumerate(rows):
        col = RED if r.startswith("!") or r.startswith("error") else (255, 190, 180)
        d.text((40, 126 + i * 32), r, font=fm, fill=col)
    for i in range(14):
        d.rectangle([36, 126 + i * 32 + 24, 36 + (60 + (i * 73) % 400), 126 + i * 32 + 27],
                    fill=(90, 30, 26))
    return img


def success():
    img, d = _frame("preview — localhost:3000", tint=(70, 200, 140))
    fh = ov.font(ov.HEAD, 34)
    fm = ov.font(ov.MONO, 18)
    fb = ov.font(ov.HEAD2, 22)
    # fake landing page
    d.rectangle([0, 46, W, 96], fill=(24, 28, 36))
    d.text((36, 60), "сайт", font=fb, fill=FG)
    xnav = W - 36
    for t in reversed(["возможности", "цены", "контакты"]):
        w_ = d.textlength(t, font=fm)
        d.text((xnav - w_, 62), t, font=fm, fill=DIM)
        xnav -= w_ + 26
    d.rectangle([0, 96, W, H], fill=(30, 36, 48))
    d.rectangle([36, 130, W - 36, 330], fill=(46, 62, 88))
    d.polygon([(60, 330), (250, 170), (420, 330)], fill=(72, 96, 128))
    d.polygon([(330, 330), (520, 200), (740, 330)], fill=(86, 112, 146))
    d.ellipse([640, 150, 700, 210], fill=(255, 214, 150))
    d.text((72, 250), "САЙТ ГОТОВ", font=fh, fill=(240, 250, 245))
    d.text((72, 300), "собрано за 3 итерации", font=fm, fill=(170, 200, 185))
    d.rectangle([72, 350, 300, 404], fill=GREEN)
    d.text((186, 377), "Открыть", font=fb, fill=(16, 32, 24), anchor="mm")
    d.rectangle([320, 350, 560, 404], outline=(120, 140, 160), width=2)
    d.text((440, 377), "Деплой", font=fb, fill=(200, 215, 230), anchor="mm")
    d.text((36, H - 46), "✓ сборка успешна · 12 файлов · 0 ошибок", font=fm, fill=GREEN)
    return img


CACHE = {}


def texture(renderer, name):
    key = (id(renderer), name)
    if key not in CACHE:
        img = {"code": code, "error": error, "success": success}[name]()
        CACHE[key] = renderer.texture_from_image(img)
    return CACHE[key]
