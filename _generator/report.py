"""Render the forecast as two Czech pages for the personal site: predictions and model explanation."""
from datetime import datetime
from html import escape
from math import exp

import numpy as np

# Chance that the 5th-placed team wins its promotion playoff, and that the 11th-placed team stays up.
# These are assumptions (no playoff history in the data), stated on both pages.
P_WIN_PROMO_PLAYOFF = 0.40
P_STAY_UP_PLAYOFF = 0.55

NBSP = " "
WEEKDAYS = ["po", "út", "st", "čt", "pá", "so", "ne"]
FONTS = "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap"


def cz(x, digits=1, signed=False):
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace("-", "−").replace(".", ",")


def pct(x, digits=0):
    return f"{cz(100 * x, digits)}{NBSP}%"


def when(ts):
    d = datetime.fromtimestamp(ts)
    return f"{WEEKDAYS[d.weekday()]} {d.day}. {d.month}. · {d:%H:%M}"


def page(title, description, body, root):
    """Shared shell using the site's style.css. `root` is the relative path to the site root."""
    return f"""<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(title)}</title>
  <meta name="description" content="{escape(description)}" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="{FONTS}" rel="stylesheet" />
  <link rel="stylesheet" href="{root}style.css" />
  <link rel="stylesheet" href="{root}maly-fotbal/maly-fotbal.css" />
</head>
<body>
  <nav><div class="wrap"><a href="{root}index.html" class="nav-logo">Milan Tichavský</a></div></nav>
  <main>
{body}
  </main>
  <footer><div class="wrap"><p>&copy; 2026 Milan Tichavský</p></div></footer>
</body>
</html>
"""


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
    return f'<section><div class="wide"><div class="mf-head"><h2 class="title">{title}</h2></div>{content}</div></section>'


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
            f"<span>hosté <b>{pct(pa)}</b></span></div>"
            f"<div class=meta>očekávaně {f['n_h']:.0f} v {f['n_a']:.0f} hráčů · "
            f"{cz(6 * f['c_h'])} v {cz(6 * f['c_a'])} opor</div></div></article>"
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
            f"<td class=l><span class=range title='80 % simulací skončí mezi {fc['q10'][k]:.0f} a {fc['q90'][k]:.0f} body'>"
            f"<span class=track></span><span class=span style='left:{pos(fc['q10'][k]):.1f}%;"
            f"width:{pos(fc['q90'][k]) - pos(fc['q10'][k]):.1f}%'></span>"
            f"<span class=dot style='left:{pos(fc['exp_points'][k]):.1f}%'></span></span>"
            f"<span class=strong>{fc['exp_points'][k]:.0f}</span>"
            f"<td class=hide-s>{pct(d[0])}<td class=strong>{pct(promoted)}<td class=strong>{pct(relegated)}</tr>"
        )
    return (
        "<div class=mf-card><div class=mf-scroll><table class=mf>"
        "<tr><th>#<th class=l>Tým<th class=hide-s>Z<th class=hide-s>B<th class=l>Konečné body<th class=hide-s>Vítěz ligy<th>Postup<th>Sestup</tr>"
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


def ratings_table(fit, ids, names):
    strength = fit.goal.strength
    rows = "".join(
        f"<tr><td class='l team'>{escape(names[ids[k]])}"
        f"<td>{cz(exp(fit.goal.att[k]), 2)}<td>{cz(exp(-fit.goal.dfs[k]), 2)}<td class=strong>{cz(strength[k], 2, True)}"
        f"<td>{cz(fit.mean_n[k])}<td>{cz(6 * fit.mean_c[k])}</tr>"
        for k in np.argsort(-strength))
    return (
        "<div class=mf-card><div class=mf-scroll><table class=mf>"
        "<tr><th class=l>Tým<th>Útok<th>Obdržené góly<th>Celkově<th>Běžně hráčů<th>Běžně opor</tr>"
        f"{rows}</table></div></div>"
        "<p class=note>Násobky proti průměrnému týmu při jeho obvyklé sestavě, bez domácího prostředí. "
        "Útok nad 1 a obdržené góly pod 1 jsou dobré. Opory = šest hráčů týmu s nejvíc starty.</p>")


def regulars_block(fc, ids, names, order):
    core_ids, apps, games, pnames = fc["core"]
    out = ""
    for k in order:
        tid = ids[k]
        players = ", ".join(f"{escape(pnames[p])} <span>{apps[tid][p]}/{games[tid]}</span>" for p in core_ids[tid])
        out += f"<p><b>{escape(names[tid])}</b>{players}</p>"
    return f"<div class=regulars>{out}</div><p class=note>V závorce: starty / odehrané zápasy týmu.</p>"


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
        '<a href="model/">Jak model funguje</a>')
    body += section("Předpokládané konečné pořadí", standings_table(fc, ids, names, order))
    body += section("Pravděpodobnost jednotlivých pozic", heatmap(fc, ids, names, order))
    body += section(
        "Nejbližší zápasy", fixture_cards(near) + legend +
        (f"<details class='mf-d'><summary>Všechny zbývající zápasy ({len(later)})</summary>"
         f"<div class=inner>{later_html}</div></details>" if later else ""))
    body += section(
        "Podrobnosti",
        f"<details class='mf-d'><summary>Síla týmů</summary><div class=inner>{ratings_table(fc['fit'], ids, names)}</div></details>"
        '<p class=note style="margin-top:2rem"><a href="model/">Jak model funguje →</a></p>')
    return page("Predikce malého fotbalu v Brně · Milan Tichavský",
                "Předpověď konečného pořadí 6. ligy malého fotbalu v Brně s ohledem na sestavy týmů.",
                body, "../")


def render_model(matches, ids, names, fc, effects, bt):
    e = effects
    sims = f"{fc['sims']:,}".replace(",", NBSP)
    n_bt = next(iter(bt.values()))["n"]
    gain = bt["no_lineup"]["logloss"] - bt["full"]["logloss"]
    beats_base = bt["full"]["logloss"] < bt["base_rates"]["logloss"]
    small_squad_effect = abs(e["opp_c"]) * 6 < 0.3 and abs(e["close_c"]) * 6 < 0.3

    squad_note = (
        f"Proti soupeři o jednu směrodatnou odchylku silnějšímu týmy nasazují {cz(e['opp_n'], 1, True)} hráče "
        f"({cz(6 * e['opp_c'], 1, True)} opor) oproti obvyklému stavu. "
        + ("Rozdíly jsou zatím malé, takže jasný zvyk šetřit nejlepší hráče proti slabším soupeřům data neukazují."
           if small_squad_effect else "Směr odpovídá představě, že týmy sestavu volí podle soupeře."))

    def bt_row(label, key):
        s = bt[key]
        return f"<tr><td class=l>{label}<td>{cz(s['logloss'], 3)}<td>{cz(s['brier'], 3)}<td>{pct(s['accuracy'])}</tr>"

    verdict = (
        ("Model je výrazně lepší než prosté sazby výhra/remíza/prohra. " if beats_base else "")
        + (f"Informace o sestavách zlepšují log-loss o {cz(gain, 3)}."
           if gain > 0.005 else
           "Informace o sestavách ale zatím model měřitelně nezlepšují: hlavně pomáhají správně číst minulé "
           "výsledky (kdo chyběl) a na naučení výběru sestav je zatím málo dat. S postupem sezóny by se to mělo zlepšit."))

    prose = f"""<div class="prose">
<h2 class="title">Data</h2>
<p>Výsledky a soupisky (kdo v kterém zápase nastoupil) se stahují z webu <strong>malyfotbalbrno.cz</strong>.
Stránka se jednou týdně vygeneruje znovu a odhady se přepočítají.</p>

<h2 class="title">Kdo hrál, ne jen kdo vyhrál</h2>
<p>U každého týmu určím šest <strong>opor</strong>, tedy hráče s nejvíc starty. U každého zápasu pak sleduji dvě čísla:
kolik lidí tým měl a kolik z šesti opor přišlo. Výhra 1:0 v šesti lidech tak váží jinak než výhra 1:0 v deseti.</p>
<p>Sestavy porovnávám vždy s tím, co tým běžně nasazuje. Model tak nezaměňuje tým, který chodí v plném počtu,
za tým, který je dobrý.</p>

<h2 class="title">Model gólů</h2>
<p>Počet gólů je modelovaný Poissonovým rozdělením, jehož střední hodnota se skládá z několika částí:</p>
<div class="formula">očekávané góly = základ × útok týmu × obrana soupeře × domácí prostředí × vliv sestav</div>
<div class="facts">
<div class="fact"><b>{cz(100 * e['home_adv'], 0, True)}{NBSP}%</b><span>góly za domácí prostředí</span></div>
<div class="fact"><b>{cz(100 * e['per_player'], 1, True)}{NBSP}%</b><span>góly za každého hráče navíc</span></div>
<div class="fact"><b>{cz(100 * e['full_core'], 0, True)}{NBSP}%</b><span>góly za 3 opory navíc</span></div></div>
<p>Týmy mají odehráno jen 6–7 zápasů, takže odhady síly <strong>stahuji k průměru</strong> (ridge regrese).
Jinak by jedno vítězství 7:0 rozhodlo o všem.</p>

<h2 class="title">Kdo nastoupí příště</h2>
<p>Týmy si sestavu vybírají podle soupeře: na favorita přijdou všichni, na outsidera nemusí. Z dosavadních zápasů proto
odhaduji, o kolik se sestava týmu liší od obvyklé podle síly soupeře a vyrovnanosti zápasu. {squad_note}</p>

<h2 class="title">Simulace sezóny</h2>
<p>Zbývající zápasy podle skutečného rozpisu odsimuluji <strong>{sims}{NBSP}krát</strong>.
Aby se do pořadí přenesla i nejistota v odhadu síly týmů, model se 150krát přepočítá na náhodně vybraných
odehraných zápasech (bootstrap). Proto jsou rozsahy bodů tak široké. Z výsledků plynou pravděpodobnosti postupu a sestupu.</p>
<p>Do postupu a sestupu počítám i baráž. Protože nemám data z minulých baráží, jsou její šance jen
odhad: {pct(P_WIN_PROMO_PLAYOFF)} pro postupující tým a {pct(P_STAY_UP_PLAYOFF)} pro záchranu.</p>

<h2 class="title">Funguje to?</h2>
<p>Každé kolo od třetího jsem předpověděl jen z kol předchozích, bez znalosti sestav ({n_bt} zápasů).
Nižší hodnoty jsou lepší.</p>
<div class="mf-card mini"><table class="mf">
<tr><th class=l>Model<th>Log-loss<th>Brier<th>Trefený výsledek</tr>
{bt_row("Tento model", "full")}
{bt_row("Totéž bez sestav", "no_lineup")}
{bt_row("Základ: průměrné sazby ligy", "base_rates")}</table></div>
<div class="callout">{verdict} Vzorek je malý, takže pořadí berte jako kvalifikovaný odhad, ne jistotu.</div>

<h2 class="title">Co model neumí</h2>
<ul class="limits">
<li>Kvalitu jednotlivých hráčů. Opory jsou určené jen počtem startů.</li>
<li>Zranění, tresty a dovolené, tedy kdo bude k dispozici konkrétní den.</li>
<li>Dlouhodobý vývoj. Po několika kolech je dat zatím velmi málo.</li></ul>

<h2 class="title">Opory týmů</h2>
{regulars_block(fc, ids, names, sorted(range(len(ids)), key=lambda k: -fc["exp_points"][k]))}
<p class="note" style="margin-top:2.5rem"><a href="../">← Zpět na předpověď</a></p>
</div>"""

    body = top("Jak model funguje",
               "Z výsledků a sestav odhadne sílu týmů, tipne, kdo za koho nastoupí, a odsimuluje zbytek sezóny.",
               '<a class=back href="../">← Předpověď</a>')
    body += f'<section><div class="wide">{prose}</div></section>'
    return page("Jak funguje model · Predikce malého fotbalu · Milan Tichavský",
                "Stručné vysvětlení modelu pro předpověď malého fotbalu v Brně: sestavy, gólový model a simulace.",
                body, "../../")
