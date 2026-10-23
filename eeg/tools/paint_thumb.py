"""
Paint the home-page thumbnail for the EEG project: an impressionist dusk seascape whose waves are
EEG traces, rendered as thousands of bristled brush strokes with impasto lighting.

    python eeg/tools/paint_thumb.py          # writes eeg/thumb.jpg (640x360)

Deterministic (fixed seed). Needs numpy, scipy and Pillow.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parents[1] / "thumb.jpg"
W, H = 1280, 720  # painted at 2x the published size
HORIZON = 0.47 * H
SUN = (0.70 * W, HORIZON - 0.02 * H)
rng = np.random.default_rng(11)


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def hexrgb(h):
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


# Palette: the site's teal ink and green, pushed toward dusk.
DEEP = hexrgb("#16303d")
TEAL = hexrgb("#1f5a63")
GREEN = hexrgb("#2f9e6b")
ROSE = hexrgb("#d98f73")
OCHRE = hexrgb("#efc27a")
CREAM = hexrgb("#f7ead0")
SEA_DARK = hexrgb("#12313a")
SEA = hexrgb("#24606a")
VIOLET = hexrgb("#6b5a7e")
FOAM = hexrgb("#6fa596")


def eeg_wave(x, phase, amp):
    """An alpha-dominated EEG-like trace: ~10 Hz carrier with waxing/waning spindles and beta."""
    u = x / W
    spindle = 0.55 + 0.45 * math.sin(2 * math.pi * (1.3 * u + phase * 0.37))
    return amp * (
        spindle * math.sin(2 * math.pi * (10.5 * u + phase))
        + 0.28 * math.sin(2 * math.pi * (23 * u + 2.1 * phase))
        + 0.18 * math.sin(2 * math.pi * (3.1 * u + 0.7 * phase))
    )


def base_colour(x, y):
    """Underpainting: what colour the canvas wants to be at (x, y)."""
    dx, dy = (x - SUN[0]) / W, (y - SUN[1]) / H
    glow = math.exp(-(dx * dx * 6 + dy * dy * 14))
    if y < HORIZON:
        t = y / HORIZON  # 0 at top, 1 at horizon
        if t < 0.55:
            c = lerp(DEEP, TEAL, t / 0.55)
        else:
            c = lerp(TEAL, ROSE, (t - 0.55) / 0.45)
        cloud = math.sin(x / 210 + 3 * math.sin(y / 61)) * math.sin(y / 23 + x / 400)
        if 0.25 < t < 0.9 and cloud > 0.35:
            c = lerp(c, VIOLET if t < 0.6 else ROSE, min(0.7, (cloud - 0.35) * 1.6))
        c = lerp(c, OCHRE, min(1, glow * 1.4))
        if glow > 0.75:
            c = lerp(c, CREAM, (glow - 0.75) * 4)
        return c
    t = (y - HORIZON) / (H - HORIZON)  # 0 at horizon, 1 at bottom
    c = lerp(SEA, SEA_DARK, t ** 0.7)
    c = lerp(c, GREEN, 0.25 * (1 - t) * (0.5 + 0.5 * math.sin(x / 37)))
    # The sun's reflection: a broken vertical column of light on the water.
    col = math.exp(-((x - SUN[0]) / (0.05 * W + 0.12 * W * t)) ** 2)
    c = lerp(c, OCHRE, col * (0.85 - 0.5 * t))
    return c


def flow_angle(x, y):
    """Stroke direction: gentle bands in the sky, wave tangents (EEG slope) on the sea."""
    if y < HORIZON:
        dx, dy = x - SUN[0], y - SUN[1]
        r = math.hypot(dx, dy) + 1e-6
        swirl = 0.6 * math.exp(-r / (0.12 * W))  # strokes curl a little around the sun
        band = 0.08 * math.sin(x / 140 + y / 50)
        return band * (1 - swirl) + (math.atan2(dy, dx) + math.pi / 2) * swirl
    t = (y - HORIZON) / (H - HORIZON)
    amp = 4 + 26 * t ** 1.3
    phase = y / 23
    slope = (eeg_wave(x + 1, phase, amp) - eeg_wave(x - 1, phase, amp)) / 2
    return math.atan(slope)


def jitter(c, s):
    return tuple(int(max(0, min(255, v + rng.normal(0, s)))) for v in c)


def stroke(draw, hdraw, x, y, width, length, cjit, stepsize=4.0):
    colour = jitter(base_colour(x, y), cjit)
    pts = [(x, y)]
    sky = y < HORIZON
    for _ in range(int(length / stepsize)):
        a = flow_angle(x, y)
        x += math.cos(a) * stepsize
        y += math.sin(a) * stepsize
        if (y < HORIZON) != sky:  # keep a clean horizon: strokes stop at it
            break
        pts.append((x, y))
    if len(pts) < 2:
        return
    # Bristles: parallel offset lines with their own tint and height.
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


def main():
    canvas = Image.new("RGB", (W, H))
    height = Image.new("L", (W, H), 0)
    px = canvas.load()
    for yy in range(0, H, 4):  # flat underpainting
        for xx in range(0, W, 4):
            c = tuple(int(v) for v in base_colour(xx, yy))
            for dy in range(4):
                for dx in range(4):
                    px[xx + dx, yy + dy] = c
    draw, hdraw = ImageDraw.Draw(canvas), ImageDraw.Draw(height)

    # Big loose strokes first, then finer ones on top (coarse-to-fine, like a painter).
    for width, length, n, cjit in ((30, 120, 1400, 10), (18, 80, 3200, 9), (10, 50, 6000, 8), (6, 30, 6000, 7)):
        for _ in range(n):
            x, y = rng.uniform(-40, W + 40), rng.uniform(-20, H + 20)
            stroke(draw, hdraw, x, y, width * rng.uniform(0.7, 1.3), length * rng.uniform(0.6, 1.4), cjit)

    # Crests: short broken dabs along the EEG traces, bright only near the sun's column.
    for k in range(11):
        t = (k + 0.5) / 11
        y0 = HORIZON + 8 + (H - HORIZON) * t ** 1.4
        amp = 4 + 26 * t ** 1.3
        phase = y0 / 23
        x = rng.uniform(-60, 0)
        while x < W:
            seg = rng.uniform(30, 110) * (0.6 + t)
            near = math.exp(-((x + seg / 2 - SUN[0]) / (0.16 * W)) ** 2)
            c = lerp(lerp(SEA, FOAM, 0.55 + 0.3 * t), CREAM, near * 0.9)
            w = 2 + 4 * t
            nb = 3
            for b in range(nb):
                off = (b - 1) * w * 0.45
                pts = [(xx, y0 + 0.55 * eeg_wave(xx, phase, amp) - 3 + off) for xx in np.arange(x, x + seg, 3)]
                lw = max(1, int(w * 0.6))
                draw.line(pts, fill=jitter(c, 10), width=lw, joint="curve")
                hdraw.line(pts, fill=int(rng.uniform(160, 255)), width=lw, joint="curve")
            x += seg + rng.uniform(60, 220) * (1.4 - near)

    # Impasto: light the stroke height map from the upper left.
    h = gaussian_filter(np.asarray(height, dtype=np.float32) / 255, 1.2)
    gy, gx = np.gradient(h)
    shade = 1 + 1.5 * (-gx * 0.7 - gy * 0.7)
    weave = 1 + 0.02 * (np.sin(np.arange(W) / 1.6)[None, :] * np.sin(np.arange(H) / 1.6)[:, None])
    img = np.asarray(canvas, dtype=np.float32) * (shade * weave)[..., None]
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=40, threshold=2))
    img.resize((W // 2, H // 2), Image.LANCZOS).save(OUT, quality=86, optimize=True, progressive=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
