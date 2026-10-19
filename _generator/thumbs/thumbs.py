import random, math, re, sys

INK = "#1a3846"; CREAM = "#f4efe3"

def grain(id):
    return (f'<filter id="{id}"><feTurbulence type="fractalNoise" baseFrequency=".9" numOctaves="2" stitchTiles="stitch"/>'
            f'<feColorMatrix values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .1 0"/><feComposite in2="SourceGraphic" operator="in"/></filter>')

def f(v): return f"{v:.1f}".rstrip("0").rstrip(".")

def blade(x, y, w, h, lean):
    tx, ty = x + lean, y - h
    return (f"M{f(x-w/2)} {f(y)}Q{f(x+lean*.25)} {f(y-h*.55)} {f(tx)} {f(ty)}"
            f"Q{f(x+lean*.25+w*.35)} {f(y-h*.5)} {f(x+w/2)} {f(y)}z")

def football():
    rnd = random.Random(7)
    hz, n = 64, 10
    # mown stripes parallel to the goal line, widening toward the viewer
    ys = [hz + 116 * (k / n) ** 1.7 for k in range(n + 1)]
    light, dark = ("#8cc47a", "#6fb36a"), ("#5fa866", "#4a9459")
    stripes, layers = [], {}
    for k in range(n):
        lite = k % 2 == 0
        stripes.append(f'<rect y="{f(ys[k])}" width="320" height="{f(ys[k+1]-ys[k]+.5)}" fill="{(light if lite else dark)[0]}"/>')
    rows = 18
    for r in range(rows):
        t = r / (rows - 1)
        y = hz + 3 + (t ** 1.7) * 118
        k = next((i for i in range(n) if ys[i + 1] >= y), n - 1)
        col = (light if k % 2 == 0 else dark)[1]
        h, w, step = 2 + t ** 1.8 * 13, 1 + t * 2.6, 3 + t * 6
        x = -5 + rnd.random() * step
        while x < 325:
            layers.setdefault(col, []).append(blade(x, y + rnd.random() * 2, w * (.7 + rnd.random() * .6),
                                                    h * (.6 + rnd.random() * .7), (rnd.random() - .5) * h * .6))
            x += step * (.6 + rnd.random() * .8)
    grass = "".join(f'<path fill="{c}" d="{"".join(p)}"/>' for c, p in layers.items())

    # pitch corner: goal line along y=gl, touchline receding to the horizon
    cx, gl = 34, 150
    lines = (f'<path d="M{cx} {gl}H330M{cx} {gl}L{cx+70} {hz+2}M{cx+22} {gl}A22 9 0 0 0 {cx+6} {gl-9}"'
             f' fill="none" stroke="#fbf8ef" stroke-width="3" stroke-linecap="round"/>')
    flag = (f'<line x1="{cx}" y1="{gl}" x2="{cx}" y2="{gl-62}" stroke="#fbf8ef" stroke-width="2.5" stroke-linecap="round"/>'
            f'<path d="M{cx+1} {gl-62}q12 4 22 1q-5 7 -1 14q-10 3 -21 -1z" fill="#e8743b" stroke="{INK}" stroke-width="1.5" stroke-linejoin="round"/>'
            f'<ellipse cx="{cx}" cy="{gl+1}" rx="7" ry="2.2" fill="#123c28" opacity=".35"/>')

    # gaussian prediction: histogram bars + density curve
    mu, sd, base, peak, x0, x1 = 190, 34, gl, 32, 76, 304
    g = lambda x: base - (base - peak) * math.exp(-((x - mu) ** 2) / (2 * sd * sd))
    bars, bw = [], 14
    for bx in range(x0 + 4, x1 - 4, bw):
        top = g(bx + bw / 2) + rnd.uniform(-4, 4)
        if base - top < 3: continue
        bars.append(f'<rect x="{bx+1.5}" y="{f(top)}" width="{bw-3}" height="{f(base-top)}" rx="2"/>')
    curve = "M" + " L".join(f"{f(x)} {f(g(x))}" for x in range(x0, x1 + 1, 3))

    front = [blade(rnd.uniform(-5, 325), 186, rnd.uniform(2.5, 4.5), rnd.uniform(8, 18), rnd.uniform(-7, 7)) for _ in range(40)]

    return f'''<svg viewBox="0 0 320 180" role="img" aria-label="Prediction distribution standing in the corner of a football pitch">
              <defs>
                <linearGradient id="sky-a" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f6e2bd"/><stop offset="1" stop-color="#f1ead6"/></linearGradient>
                <linearGradient id="shade-a" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".25"/><stop offset=".45" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#0d3a24" stop-opacity=".4"/></linearGradient>
                <linearGradient id="bar-a" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{CREAM}"/><stop offset="1" stop-color="#e3dcc8"/></linearGradient>
                {grain("grain-a")}
              </defs>
              <rect width="320" height="180" fill="url(#sky-a)"/>
              <circle cx="272" cy="28" r="13" fill="#f0b45e"/>
              <path d="M0 66C40 57 80 61 120 56s80 2 120-3 60 4 80 3V68H0z" fill="#a6c98f" opacity=".7"/>
              {"".join(stripes)}
              {grass}
              {lines}
              <rect y="{hz}" width="320" height="{180-hz}" fill="url(#shade-a)"/>
              {flag}
              <ellipse cx="{mu}" cy="{base+2}" rx="100" ry="5" fill="#123c28" opacity=".3"/>
              <g fill="url(#bar-a)" stroke="{INK}" stroke-opacity=".5" stroke-width="1.2">{"".join(bars)}</g>
              <path d="{curve}" fill="none" stroke="{INK}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="{mu}" cy="{peak}" r="7" fill="{INK}" stroke="{CREAM}" stroke-width="2.5"/>
              <path fill="#2f7a4a" d="{"".join(front)}"/>
              <rect width="320" height="180" fill="#000" filter="url(#grain-a)"/>
            </svg>'''

def iso_box(cx, cy, s, h, tape="#2f6f7a", label=True, op=1):
    # top face centred on (cx,cy); s = half width, h = height
    q = s / 2
    top = f"M{cx} {cy-q}L{cx+s} {cy}L{cx} {cy+q}L{cx-s} {cy}z"
    left = f"M{cx-s} {cy}L{cx} {cy+q}V{cy+q+h}L{cx-s} {cy+h}z"
    right = f"M{cx} {cy+q}L{cx+s} {cy}V{cy+h}L{cx} {cy+q+h}z"
    out = [f'<g opacity="{op}" stroke="{INK}" stroke-width="2.5" stroke-linejoin="round">',
           f'<path d="{left}" fill="#d9c9a6"/><path d="{right}" fill="#c4b088"/><path d="{top}" fill="{CREAM}"/>',
           f'<path d="M{cx-s/2} {cy-q/2}L{cx+s/2} {cy+q/2}V{cy+q/2+h}" fill="none" stroke="{tape}" stroke-width="{s*.16:.1f}" stroke-linejoin="miter"/>']
    if label:
        lx, ly = cx - s * .78, cy + h * .3
        out.append(f'<path d="M{lx} {ly}l{s*.5:.1f} {s*.25:.1f}v{h*.38:.1f}l-{s*.5:.1f} -{s*.25:.1f}z" fill="#fff" stroke-width="1.5"/>')
        for i in range(3):
            yy = ly + h * .1 + i * h * .09
            out.append(f'<path d="M{lx+s*.08:.1f} {yy+s*.04:.1f}l{s*.3:.1f} {s*.15:.1f}" stroke-width="1.2"/>')
    out.append("</g>")
    return "".join(out)

def messenger():
    rnd = random.Random(3)
    speed = "".join(f'<path d="M{f(x)} {f(y)}h{f(l)}" />' for x, y, l in
                    [(rnd.uniform(10, 120), rnd.uniform(40, 150), rnd.uniform(18, 60)) for _ in range(9)])
    dots = "".join(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r)}"/>' for x, y, r in
                   [(rnd.uniform(0, 320), rnd.uniform(0, 180), rnd.uniform(.6, 1.8)) for _ in range(40)])
    return f'''<svg viewBox="0 0 320 180" role="img" aria-label="A stack of parcels on its way from a shop">
              <defs>
                <linearGradient id="bg-b" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f4b77a"/><stop offset="1" stop-color="#d9673f"/></linearGradient>
                {grain("grain-b")}
              </defs>
              <rect width="320" height="180" fill="url(#bg-b)"/>
              <g fill="{CREAM}" opacity=".35">{dots}</g>
              <circle cx="70" cy="58" r="64" fill="#fff" opacity=".07"/><circle cx="70" cy="58" r="40" fill="#fff" opacity=".07"/>
              <g stroke="{CREAM}" stroke-opacity=".45" stroke-width="3" stroke-linecap="round">{speed}</g>
              <path d="M-6 160C50 160 70 98 128 112" fill="none" stroke="{CREAM}" stroke-width="4" stroke-dasharray="1 11" stroke-linecap="round"/>
              {iso_box(64, 48, 18, 20, op=.55, label=False)}
              {iso_box(118, 128, 14, 15, op=.7, label=False)}
              <ellipse cx="214" cy="164" rx="80" ry="10" fill="#7a2e14" opacity=".3"/>
              {iso_box(186, 104, 34, 38)}
              {iso_box(244, 112, 30, 34)}
              {iso_box(186, 70, 30, 32)}
              <rect width="320" height="180" fill="#000" filter="url(#grain-b)"/>
            </svg>'''

p = sys.argv[1]
s = open(p).read()
svgs = re.findall(r"<svg viewBox.*?</svg>", s, flags=re.S)
s = s.replace(svgs[0], football()).replace(svgs[1], messenger())
open(p, "w").write(s)
