"""
Paint the home-page thumbnail for the Messenger integration: a still life of a pile of parcels on a
wooden floor against a dark wall, rendered as thousands of bristled brush strokes with impasto
lighting (same technique as eeg/tools/paint_thumb.py).

    python integrace-messenger/tools/paint_thumb.py     # writes integrace-messenger/thumb.jpg (640x360)

Deterministic (fixed seed). Needs numpy, scipy and Pillow.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parents[1] / "thumb.jpg"
W, H = 1280, 720  # painted at 2x the published size
FLOOR_Y = 0.58 * H
LIGHT = (0.22 * W, 0.18 * H)  # a soft pool of light on the wall, upper left
rng = np.random.default_rng(5)


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def hexrgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


# Palette: a dark teal wall and warm floorboards, so the cardboard glows.
WALL_DARK = hexrgb("#1b2e33")
WALL = hexrgb("#35504f")
WALL_LIT = hexrgb("#7d8a73")
FLOOR_DARK = hexrgb("#3a2a20")
FLOOR = hexrgb("#7a5538")
FLOOR_LIT = hexrgb("#a8794c")
SHADOW = hexrgb("#1e1a18")
UMBER = hexrgb("#3a251b")
CREAM = hexrgb("#f7ead0")
TEAL = hexrgb("#2f6f7a")
GREEN = hexrgb("#2f9e6b")
TWINE = hexrgb("#e6cf9a")

# Each parcel: front-bottom corner x, y, half-width, height, (lid, lit face, shadow face), tape.
KRAFT = (hexrgb("#f0d6a2"), hexrgb("#d29a5c"), hexrgb("#7e5232"))
PALE = (hexrgb("#f6e5bd"), hexrgb("#e2bb80"), hexrgb("#8f6a43"))
DARK = (hexrgb("#d9a86c"), hexrgb("#b47844"), hexrgb("#62402a"))
BOXES = [  # back to front, bottom to top
    (0.82 * W, 0.84 * H, 118, 176, PALE, CREAM),
    (0.30 * W, 0.95 * H, 150, 150, KRAFT, TEAL),
    (0.58 * W, 0.99 * H, 170, 128, DARK, "twine"),
    (0.30 * W, 0.95 * H - 150 - 75 + 48, 96, 84, PALE, GREEN),
    (0.58 * W, 0.99 * H - 128 - 85 + 62, 124, 112, KRAFT, CREAM),
    (0.58 * W, 0.99 * H - 128 - 85 + 62 - 112 - 62 + 40, 80, 64, DARK, TEAL),
]

# Regions: strokes never cross from one to another, which keeps edges crisp.
WALLR, FLOORR, CAST, B_TOP, B_LEFT, B_RIGHT = range(6)
REGION = Image.new("L", (W, H), WALLR)
OWNER = Image.new("L", (W, H), 255)  # which parcel is visible at each pixel


def box_faces(px, py, a, h):
    left = [(px - a, py - a / 2), (px, py), (px, py - h), (px - a, py - a / 2 - h)]
    right = [(px, py), (px + a, py - a / 2), (px + a, py - a / 2 - h), (px, py - h)]
    top = [(px, py - h), (px + a, py - a / 2 - h), (px, py - a - h), (px - a, py - a / 2 - h)]
    return top, left, right


def build_regions():
    d, o = ImageDraw.Draw(REGION), ImageDraw.Draw(OWNER)
    d.rectangle((0, FLOOR_Y, W, H), fill=FLOORR)
    # Shadows fall to the right, across the floor and up the wall.
    for px, py, a, h, _, _ in BOXES[:3]:
        d.polygon([(px - a, py - a / 2), (px + a, py - a / 2), (px + a + 1.1 * h, py - a / 2 - 0.15 * h),
                   (px + 1.1 * h, py - 0.15 * h), (px, py)], fill=CAST)
        d.polygon([(px + a, py - a / 2 - h), (px + a + 0.9 * h, py - a / 2 - 0.9 * h),
                   (px + a + 1.1 * h, py - a / 2 - 0.15 * h), (px + a, py - a / 2)], fill=CAST)
    for k, (px, py, a, h, _, _) in enumerate(BOXES):
        for face, rid in zip(box_faces(px, py, a, h), (B_TOP, B_LEFT, B_RIGHT)):
            d.polygon(face, fill=rid)
            o.polygon(face, fill=k)


def at(img, x, y):
    return img.getpixel((min(W - 1, max(0, int(x))), min(H - 1, max(0, int(y)))))


def base_colour(x, y):
    """Underpainting: what colour the canvas wants to be at (x, y)."""
    r = at(REGION, x, y)
    if r in (B_TOP, B_LEFT, B_RIGHT):
        c = BOXES[at(OWNER, x, y)][4][r - B_TOP]
        return lerp(c, UMBER, 0.25 * max(0, (y - 0.75 * H) / (0.25 * H)))  # a little darker near the floor
    glow = math.exp(-((math.hypot(x - LIGHT[0], 1.3 * (y - LIGHT[1])) / (0.45 * W)) ** 2))
    if r == WALLR:
        c = lerp(WALL_DARK, WALL, 0.3 + 0.7 * y / FLOOR_Y)
        return lerp(c, WALL_LIT, 0.75 * glow)
    t = (y - FLOOR_Y) / (H - FLOOR_Y)
    u = (x + 260 * t) / 150  # floorboards, slightly in perspective
    c = lerp(FLOOR_DARK, (FLOOR, FLOOR_LIT, FLOOR)[int(u) % 3], 0.5 + 0.5 * t)
    if math.sin(u * math.pi) ** 2 < 0.02:
        c = lerp(c, FLOOR_DARK, 0.6)  # the seams
    c = lerp(c, FLOOR_LIT, 0.35 * glow)
    if r == CAST:
        c = lerp(c, SHADOW, 0.6)
    return c


def flow_angle(x, y):
    """Stroke direction per region: loose diagonal scumbles on the wall, along boards and box edges."""
    r = at(REGION, x, y)
    if r == WALLR:
        return 0.9 * math.sin(x / 230 + y / 170) - 0.3
    if r in (FLOORR, CAST):
        return math.atan2(-0.06, 1) - 0.05 * math.sin(y / 30)
    if r == B_TOP:
        return math.atan2(-0.5, 1) if int(x / 40) % 2 else math.atan2(0.5, 1)
    if r == B_LEFT:
        return math.atan2(0.5, 1)
    return math.pi / 2  # vertical drags on the shadowed faces


def jitter(c, s):
    return tuple(int(max(0, min(255, v + rng.normal(0, s)))) for v in c)


def stroke(draw, hdraw, x, y, width, length, cjit, stepsize=4.0):
    colour = jitter(base_colour(x, y), cjit)
    pts = [(x, y)]
    home = (at(REGION, x, y), at(OWNER, x, y))
    for _ in range(int(length / stepsize)):
        a = flow_angle(x, y)
        x += math.cos(a) * stepsize
        y += math.sin(a) * stepsize
        if (at(REGION, x, y), at(OWNER, x, y)) != home:  # strokes stop at an edge
            break
        pts.append((x, y))
    if len(pts) < 2:
        return
    nb = max(3, int(width / 2.2))
    for b in range(nb):
        off = (b / (nb - 1) - 0.5) * width
        bp = []
        for i, (px, py) in enumerate(pts):
            j = min(i, len(pts) - 2)
            a = math.atan2(pts[j + 1][1] - pts[j][1], pts[j + 1][0] - pts[j][0])
            bp.append((px - math.sin(a) * off, py + math.cos(a) * off))
        lw = max(2, int(width / nb * 1.6))
        draw.line(bp, fill=jitter(colour, 6), width=lw, joint="curve")
        hdraw.line(bp, fill=int(rng.uniform(90, 255)), width=lw, joint="curve")


def brush_line(draw, hdraw, p, q, colour, w, wobble=1.5, owner=None):
    """A hand-drawn line of a few bristles, for box edges and tape; only where `owner` parcel shows."""
    n = max(2, int(math.dist(p, q) / 6))
    for b in range(3):
        off = (b - 1) * w * 0.35
        pts = [(p[0] + (q[0] - p[0]) * i / n + rng.normal(0, wobble), p[1] + (q[1] - p[1]) * i / n + off + rng.normal(0, wobble))
               for i in range(n + 1)]
        lw = max(1, int(w * 0.5))
        runs, run = [], []
        for pt in pts:
            if owner is None or at(OWNER, *pt) == owner:
                run.append(pt)
            elif run:
                runs.append(run)
                run = []
        runs.append(run)
        fill, hfill = jitter(colour, 8), int(rng.uniform(170, 255))
        for run in runs:
            if len(run) > 1:
                draw.line(run, fill=fill, width=lw, joint="curve")
                hdraw.line(run, fill=hfill, width=lw, joint="curve")


def mid(p, q, t=0.5):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def dress_parcel(draw, hdraw, k):
    """Tape or twine, a shipping label, then umber edges."""
    px, py, a, h, (lid, lit, dark), tape = BOXES[k]
    top, left, right = box_faces(px, py, a, h)
    if tape == "twine":  # crossed string over the lid and down both faces
        for s, e in ((mid(top[0], top[3]), mid(top[1], top[2])), (mid(top[0], top[1]), mid(top[3], top[2]))):
            brush_line(draw, hdraw, s, e, TWINE, 5, 0.6, owner=k)
        brush_line(draw, hdraw, mid(top[0], top[3]), mid(left[0], left[1]), TWINE, 5, 0.6, owner=k)
        brush_line(draw, hdraw, mid(top[0], top[1]), mid(right[0], right[1]), lerp(TWINE, dark, 0.5), 5, 0.6, owner=k)
    else:  # a strip of tape along the lid seam and down the lit face
        brush_line(draw, hdraw, mid(top[0], top[3]), mid(top[1], top[2]), tape, 0.2 * a, 0.8, owner=k)
        brush_line(draw, hdraw, mid(top[0], top[3]), mid(left[0], left[1]), lerp(tape, lit, 0.15), 0.2 * a, 0.8, owner=k)
    if a > 100:  # a shipping label on the lit face, with a few scribbled lines of address
        lx, ly = left[0][0] + 0.12 * a, left[0][1] + 0.2 * h
        for i in range(5):
            brush_line(draw, hdraw, (lx, ly + i * 7), (lx + 0.32 * a, ly + 0.16 * a + i * 7), CREAM, 8, 0.6, owner=k)
        for i in range(3):
            s = (lx + 6, ly + 6 + i * 9)
            brush_line(draw, hdraw, s, (s[0] + 0.2 * a * rng.uniform(0.6, 1), s[1] + 0.1 * a), UMBER, 2, 0.6, owner=k)
    for face in (top, left, right):
        for i in range(4):
            brush_line(draw, hdraw, face[i], face[(i + 1) % 4], UMBER, 4, 1.0, owner=k)


def main():
    build_regions()
    canvas = Image.new("RGB", (W, H))
    height = Image.new("L", (W, H), 0)
    px = canvas.load()
    for yy in range(0, H, 2):  # flat underpainting
        for xx in range(0, W, 2):
            c = tuple(int(v) for v in base_colour(xx, yy))
            for dy in range(2):
                for dx in range(2):
                    px[xx + dx, yy + dy] = c
    draw, hdraw = ImageDraw.Draw(canvas), ImageDraw.Draw(height)

    # Big loose strokes first, then finer ones on top (coarse-to-fine, like a painter).
    for width, length, n, cjit in ((30, 120, 1400, 10), (18, 80, 3200, 9), (10, 50, 6000, 8), (6, 30, 6000, 7)):
        for _ in range(n):
            x, y = rng.uniform(-40, W + 40), rng.uniform(-20, H + 20)
            stroke(draw, hdraw, x, y, width * rng.uniform(0.7, 1.3), length * rng.uniform(0.6, 1.4), cjit)

    for k in range(len(BOXES)):
        dress_parcel(draw, hdraw, k)

    # Impasto: light the stroke height map from the upper left.
    hm = gaussian_filter(np.asarray(height, dtype=np.float32) / 255, 1.2)
    gy, gx = np.gradient(hm)
    shade = 1 + 1.5 * (-gx * 0.7 - gy * 0.7)
    weave = 1 + 0.02 * (np.sin(np.arange(W) / 1.6)[None, :] * np.sin(np.arange(H) / 1.6)[:, None])
    img = np.asarray(canvas, dtype=np.float32) * (shade * weave)[..., None]
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=40, threshold=2))
    img.resize((W // 2, H // 2), Image.LANCZOS).save(OUT, quality=86, optimize=True, progressive=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
