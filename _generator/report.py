"""Render the forecast as two Czech pages for the personal site: predictions and model explanation."""
from datetime import datetime
from html import escape
from math import exp

import numpy as np

import site_shell as shell

# Held-out check over earlier seasons (leagues 4-7, seasons 2024/25-2026/27, every match predicted
# only from earlier days). Computed once by research/; log-loss, lower is better.
HISTORY_CHECK = {"n": 913, "old": 0.897, "new": 0.874, "base": 0.967}

# Chance that the 5th-placed team wins its promotion playoff, and that the 11th-placed team stays up.
# These are assumptions (no playoff history in the data), stated on both pages.
P_WIN_PROMO_PLAYOFF = 0.40
P_STAY_UP_PLAYOFF = 0.55

NBSP = " "
WEEKDAYS = ["po", "út", "st", "čt", "pá", "so", "ne"]


def cz(x, digits=1, signed=False):
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace("-", "−").replace(".", ",")


def pct(x, digits=0):
    return f"{cz(100 * x, digits)}{NBSP}%"


def when(ts):
    d = datetime.fromtimestamp(ts)
    return f"{WEEKDAYS[d.weekday()]} {d.day}. {d.month}. · {d:%H:%M}"


def page(title, description, body, root):
    """Wrap `body` in the site's shared shell (templates/base.html). `root` is the path to the site root."""
    return shell.env.get_template("maly-fotbal.html").render(
        title=title, description=description, body=body, root=root, fonts=shell.FONTS)


def hero(kicker, h1, sub, tab, root):
    links = "".join(
        f'<a href="{href}" class="{"btn-primary" if key == tab else "btn-ghost"}">{label}</a>'
        for key, href, label in (("forecast", root + "maly-fotbal/", "Předpověď"),
                                 ("model", root + "maly-fotbal/model/", "Jak to funguje")))
    return f"""    <header class="hero"><div class="wrap">
      <p class="eyebrow">{kicker}</p>
      <h1>{h1}</h1>
      <p class="hero-sub">{sub}</p>
      <div class="hero-actions">{links}</div>
    </div></header>"""


def top(h1, meta, back=''):
    return (f'    <header class="mf-top"><div class="wide">{back}<h1>{h1}</h1>'
            f'<p class="meta">{meta}</p></div></header>')


def section(title, content):
    head = f'<div class="mf-head"><h2 class="title">{title}</h2></div>' if title else ""
    return f'<section><div class="wide">{head}{content}</div></section>'


def fixture_cards(fixtures):
    out = []
    for f in fixtures:
        m = f["match"]
        ph, pd, pa = f["probs"]
        out.append(
            f"<article class=fx><time>{when(m['ts'])}</time>"
            f"<div class=teams><span class=h>{escape(m['home']['name'])}</span>"
            f"<span class=sc title='očekávané góly'>{cz(f['xg_h'])} : {cz(f['xg_a'])}</span>"
            f"<span class=a>{escape(m['away']['name'])}</span></div>"
            f"<div class=stub><div class=bar><i class=c-h style='width:{100 * ph:.1f}%'></i>"
            f"<i class=c-d style='width:{100 * pd:.1f}%'></i><i class=c-a style='width:{100 * pa:.1f}%'></i></div>"
            f"<div class=pr><span><b>{pct(ph)}</b> domácí</span><span>remíza {pct(pd)}</span>"
            f"<span>hosté <b>{pct(pa)}</b></span></div></div></article>"
        )
    return "<div class=fx-grid>" + "".join(out) + "</div>"


def standings_table(fc, ids, names, order):
    T = len(ids)
    table = fc["table"]
    lo, hi = fc["q10"].min() - 2, fc["q90"].max() + 2

    def pos(x):
        return 100 * (x - lo) / (hi - lo)

    def zone(rank):
        return "up" if rank < 4 else "po" if rank in (4, T - 2) else "down" if rank == T - 1 else ""

    rows = ""
    for rank, k in enumerate(order):
        d = fc["rank_dist"][k]
        promoted = d[:4].sum() + P_WIN_PROMO_PLAYOFF * d[4]
        relegated = d[T - 1] + (1 - P_STAY_UP_PLAYOFF) * d[T - 2]
        rows += (
            f"<tr class='{zone(rank)}'><td class=rk>{rank + 1}<td class='l team'>{escape(names[ids[k]])}"
            f"<td class='hide-s dim'>{table[k, 0]}<td class=hide-s>{table[k, 6]}"
            f"<td class='l pts'><div class=pts-in><span class=range title='80 % simulací skončí mezi {fc['q10'][k]:.0f} a {fc['q90'][k]:.0f} body'>"
            f"<span class=track></span><span class=span style='left:{pos(fc['q10'][k]):.1f}%;"
            f"width:{pos(fc['q90'][k]) - pos(fc['q10'][k]):.1f}%'></span>"
            f"<span class=dot style='left:{pos(fc['exp_points'][k]):.1f}%'></span></span>"
            f"<span class=strong>{fc['exp_points'][k]:.0f}</span></div>"
            f"<td class=hide-s>{pct(d[0])}<td class=strong>{pct(promoted)}<td class=strong>{pct(relegated)}</tr>"
        )
    return (
        "<div class=mf-card><div class=mf-scroll><table class=mf>"
        "<tr><th>#<th class=l>Tým<th class=hide-s>Z<th class=hide-s>B<th class='l pts'>Konečné body<th class=hide-s>Vítěz ligy<th>Postup<th>Sestup</tr>"
        f"{rows}</table></div></div>"
        "<div class=legend>"
        "<span><i style='background:#bfe0c8'></i>přímý postup do 5. ligy (1.–4.)</span>"
        "<span><i style='background:#f3e2a0'></i>baráž (5. a 11. místo)</span>"
        "<span><i style='background:#f2bdb5'></i>přímý sestup do 7. ligy (12.)</span></div>"
        f"<p class=note>Tečka = očekávané body, pruh = rozsah, do kterého spadne 80 % simulací. "
        f"Postup a sestup zahrnují i baráž; šance na její zisk ({pct(P_WIN_PROMO_PLAYOFF)} u postupové, "
        f"{pct(P_STAY_UP_PLAYOFF)} u záchranné) jsou jen odhad. Z = odehrané zápasy, B = body.</p>"
    )


def heatmap(fc, ids, names, order):
    T = len(ids)
    zones = ["up"] * 4 + ["po"] + [""] * (T - 7) + ["po", "down"]
    rows = "".join(
        f"<tr><td class='l team'>{escape(names[ids[k]])}" + "".join(
            f"<td style='background:color-mix(in srgb,var(--blue) {min(90, 120 * p):.0f}%,transparent)"
            f"{';color:#fff' if p > 0.35 else ''}'>{'' if p < 0.005 else f'{100 * p:.0f}'}"
            for p in fc["rank_dist"][k]) + "</tr>"
        for k in order)
    head = "".join(f"<th class='{zones[r]}'>{r + 1}" for r in range(T))
    return (f"<div class=mf-card><div class=mf-scroll><table class='mf hm'><tr><th class=l>Tým{head}</tr>{rows}"
            "</table></div></div><p class=note>Pravděpodobnost (v %), že tým skončí na dané pozici.</p>")


def ratings_table(fit, prior, ids, names):
    strength = fit.goal.strength
    rows = "".join(
        f"<tr><td class='l team'>{escape(names[ids[k]])}"
        f"<td>{cz(exp(fit.goal.att[k]), 2)}<td>{cz(exp(-fit.goal.dfs[k]), 2)}<td class=strong>{cz(strength[k], 2, True)}"
        f"<td class=dim>{cz(prior[k], 2, True)}</tr>"
        for k in np.argsort(-strength))
    return (
        "<div class=mf-card><div class=mf-scroll><table class=mf>"
        "<tr><th class=l>Tým<th>Útok<th>Obdržené góly<th>Celkově<th>Výchozí odhad</tr>"
        f"{rows}</table></div></div>"
        "<p class=note>Násobky proti průměrnému týmu, bez domácího prostředí. "
        "Útok nad 1 a obdržené góly pod 1 jsou dobré. Výchozí odhad je síla, kterou týmu přisoudily minulé sezóny "
        "a historie jeho hráčů; k ní se odhad stahuje, dokud o týmu nemáme dost letošních zápasů.</p>")


def render_forecast(matches, ids, names, fc):
    order = np.argsort(-fc["exp_points"])
    total, played = len(matches), sum(m["hg"] is not None for m in matches)
    now = datetime.now()
    upcoming = sorted(fc["fixtures"], key=lambda f: (f["match"]["ts"], f["match"]["id"]))
    near, later = upcoming[:9], upcoming[9:]

    by_round = {}
    for f in later:
        by_round.setdefault(f["match"]["round"], []).append(f)
    later_html = "".join(f"<h3 class=round-title>{r}. kolo</h3>{fixture_cards(fs)}" for r, fs in sorted(by_round.items()))

    legend = ("<div class=legend><span><i class=c-h></i>výhra domácích</span><span><i class=c-d></i>remíza</span>"
              "<span><i class=c-a></i>výhra hostů</span></div>")
    body = top(
        "Predikce 6. ligy malého fotbalu",
        f"Odehráno {played} ze {total} zápasů · aktualizováno {now.day}. {now.month}. {now.year} · "
        '<a href="model/index.html">Jak model funguje</a>')
    body += section(None, standings_table(fc, ids, names, order))
    body += section("Pravděpodobnost jednotlivých pozic", heatmap(fc, ids, names, order))
    body += section(
        "Nejbližší zápasy", fixture_cards(near) + legend +
        (f"<details class='mf-d'><summary>Všechny zbývající zápasy ({len(later)})</summary>"
         f"<div class=inner>{later_html}</div></details>" if later else ""))
    body += section(
        "Podrobnosti",
        f"<details class='mf-d'><summary>Síla týmů</summary><div class=inner>{ratings_table(fc['fit'], fc['prior'], ids, names)}</div></details>"
        '<p class=note style="margin-top:2rem"><a href="model/index.html">Jak model funguje →</a></p>')
    return page("Predikce malého fotbalu v Brně · Milan Tichavský",
                "Předpověď konečného pořadí 6. ligy malého fotbalu v Brně z letošních výsledků a minulých sezón.",
                body, "../")


def render_model(matches, ids, names, fc, effects, bt):
    e, h = effects, HISTORY_CHECK
    sims = f"{fc['sims']:,}".replace(",", NBSP)
    n_bt = next(iter(bt.values()))["n"]

    def bt_row(label, key):
        s = bt[key]
        return f"<tr><td class=l>{label}<td>{cz(s['logloss'], 3)}<td>{cz(s['brier'], 3)}<td>{pct(s['accuracy'])}</tr>"

    gain = bt["no_history"]["logloss"] - bt["model"]["logloss"]
    verdict = (
        ("Model je výrazně lepší než prosté sazby výhra/remíza/prohra. "
         if bt["model"]["logloss"] < bt["base_rates"]["logloss"] - 0.02 else "")
        + (f"Znalost minulých sezón letos zlepšuje log-loss o {cz(gain, 3)}, hlavně v prvních kolech."
           if gain > 0.002 else
           "Letos zatím minulé sezóny měřitelně nepomáhají, ale letošních zápasů je na takové srovnání málo; "
           "v minulých sezónách pomáhaly hlavně v prvních kolech."))

    prose = f"""<div class="prose">
<h2 class="title">Data</h2>
<p>Výsledky a soupisky (kdo v kterém zápase nastoupil) se stahují z webu <strong>malyfotbalbrno.cz</strong>,
a to nejen pro letošní 6. ligu, ale i pro {e['seasons']} předchozích sezón všech mužských lig od 1. do 7.
Stránka se jednou týdně vygeneruje znovu a odhady se přepočítají.</p>

<h2 class="title">Výchozí odhad z minulých sezón</h2>
<p>Na začátku sezóny o týmech letos nic nevíme, ale víme, jak hrály loni. Z týmů, které postoupily nebo sestoupily,
model odhadl, že sousední ligy se liší asi o <strong>{cz(100 * e['per_league'], 0)}{NBSP}%</strong> gólů.
Loňská síla se navíc přenáší jen částečně (asi z {cz(100 * e['carry_over'], 0)}{NBSP}%): týmy se mezi sezónami
mění. Nové týmy dostanou průměr, jakého nové týmy v minulosti dosahovaly.</p>
<p>Jak sezóna běží, výchozí odhad bere čím dál víc v úvahu i to, <strong>kdo za tým skutečně hraje</strong>:
u každého hráče, jak silný byl loni jeho tým, i ve vyšší lize. Tým poskládaný z hráčů první ligy tak začne výš,
i když je v soutěži nový.</p>

<h2 class="title">Model gólů</h2>
<div class="formula">očekávané góly = základ × útok týmu × obrana soupeře × domácí prostředí</div>
<div class="facts">
<div class="fact"><b>{cz(100 * e['home_adv'], 0, True)}{NBSP}%</b><span>góly za domácí prostředí</span></div>
<div class="fact"><b>{cz(100 * e['per_league'], 0)}{NBSP}%</b><span>rozdíl gólů mezi sousedními ligami</span></div>
<div class="fact"><b>{e['transitions']}</b><span>přechodů týmů mezi sezónami v datech</span></div></div>
<p>Síla týmů se odhaduje z letošních zápasů, ale <strong>stahuje se k výchozímu odhadu</strong> (ridge regrese).
Po pár kolech tak rozhoduje hlavně historie, ke konci sezóny hlavně letošní výsledky.</p>
<p>Góly v malém fotbale kolísají zhruba dvakrát víc, než předpokládá obvyklé Poissonovo rozdělení
(malé kádry, střídání sestav). Model proto používá <strong>negativně binomické rozdělení</strong> se stejným průměrem.
S Poissonem byl model příliš sebejistý: když favoritovi dával 90{NBSP}%, vyhrál jen asi v 80{NBSP}% případů.</p>

<h2 class="title">Simulace sezóny</h2>
<p>Zbývající zápasy podle skutečného rozpisu odsimuluji <strong>{sims}{NBSP}krát</strong>.
Aby se do pořadí přenesla i nejistota v odhadu síly týmů, model se 150krát přepočítá na náhodně vybraných
odehraných zápasech (bootstrap). Z výsledků plynou pravděpodobnosti postupu a sestupu.</p>
<p>Do postupu a sestupu počítám i baráž. Protože nemám data z minulých baráží, jsou její šance jen
odhad: {pct(P_WIN_PROMO_PLAYOFF)} pro postupující tým a {pct(P_STAY_UP_PLAYOFF)} pro záchranu.</p>

<h2 class="title">Funguje to?</h2>
<p>Model jsem vyzkoušel na minulých sezónách 4. až 7. ligy: každý zápas předpověděl jen z toho, co se odehrálo
dříve. Nastavení jsem ladil na sezónách 2022/23 a 2023/24 a ověřil na {h['n']} zápasech let 2024/25 až 2026/27,
které ladění nikdy neviděly. Log-loss (nižší je lepší): tento model {cz(h['new'], 3)}, původní verze
{cz(h['old'], 3)}, prosté sazby výhra/remíza/prohra {cz(h['base'], 3)}.</p>
<p>Letošní sezóna stejným způsobem ({n_bt} zápasů):</p>
<div class="mf-card mini"><table class="mf">
<tr><th class=l>Model<th>Log-loss<th>Brier<th>Trefený výsledek</tr>
{bt_row("Tento model", "model")}
{bt_row("Totéž bez minulých sezón", "no_history")}
{bt_row("Základ: průměrné sazby lig", "base_rates")}</table></div>
<div class="callout">{verdict} Vzorek je malý, takže pořadí berte jako kvalifikovaný odhad, ne jistotu.</div>

<h2 class="title">Co model neumí</h2>
<ul class="limits">
<li>Kdo na zápas přijde. Počet hráčů a účast opor jsem zkoušel, ale předpovědi nezlepšily: dopředu se odhadnout
nedají a vliv je malý (asi +0,2 gólu za každého hráče navíc).</li>
<li>Kvalitu jednotlivých hráčů. Hráč je ohodnocen jen podle toho, jak silný byl loni jeho tým.</li>
<li>Zranění, tresty a dovolené.</li>
<li>Vývoj během sezóny. Rozsahy bodů jsou proto spíš užší, než by měly být: v minulých sezónách se do
80{NBSP}% rozsahu vešly jen asi dvě třetiny týmů.</li></ul>

<p class="note" style="margin-top:2.5rem"><a href="../index.html">← Zpět na předpověď</a></p>
</div>"""

    body = top("Jak model funguje",
               "Z letošních výsledků a minulých sezón odhadne sílu týmů a odsimuluje zbytek sezóny.",
               '<a class=back href="../index.html">← Předpověď</a>')
    body += f'<section><div class="wide">{prose}</div></section>'
    return page("Jak funguje model · Predikce malého fotbalu · Milan Tichavský",
                "Stručné vysvětlení modelu pro předpověď malého fotbalu v Brně: minulé sezóny, gólový model a simulace.",
                body, "../../")
