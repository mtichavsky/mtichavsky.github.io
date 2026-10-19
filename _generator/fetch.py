"""Download fixtures, results and match line-ups from malyfotbalbrno.cz.

Fixtures come from the site's public JSON endpoint (/api/match/). Line-ups are not in that
list, but every match page (/zapas/<id>/) embeds the full match, rosters included, in its
public __NEXT_DATA__ block. Finished matches are cached in data/lineups/ so reruns only
download what is new or changed.
"""
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.malyfotbalbrno.cz"
LEAGUE_ID = "a23bf801-2281-4158-9ec2-8bb9a46e1f08"  # 6. Barber Moliére liga, 2026/27
DATA = Path(__file__).parent / "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (small-football-prediction-model)"}


def _get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _fetch_fixtures(league_id=LEAGUE_ID):
    seen, offset = {}, 0
    while True:
        url = f"{BASE}/api/match/?league={league_id}&offset={offset}&sortBy=round&orderDesc=0"
        page = json.loads(_get(url))
        new = [m for m in page if m["id"] not in seen]
        seen.update((m["id"], m) for m in page)
        if not new or len(page) < 40:
            return list(seen.values())
        offset += 40


def _parse_score(raw):
    m = re.fullmatch(r"(\d+):(\d+)", raw or "")
    return (int(m[1]), int(m[2])) if m else (None, None)


def _fetch_lineups(match_id):
    html = _get(f"{BASE}/zapas/{match_id}/")
    blob = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    match = json.loads(blob[1])["props"]["pageProps"]["match"]
    out = {}
    for side in ("home", "visiting"):
        out[side] = [
            {"id": p["player"]["id"], "name": p["player"]["fullName"],
             "gk": p["goalKeeper"], "goals": p["goals"]}
            for p in match[side]["players"]
        ]
    return out


def _lineups_cached(fx):
    """Line-ups for a finished fixture; re-downloaded if the score changed since caching."""
    path = DATA / "lineups" / f"{fx['id']}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["score"] == fx["score"] and cached["home"] and cached["visiting"]:
            return cached
    try:
        lineups = _fetch_lineups(fx["id"])
    except Exception as exc:  # keep going; the match is then left out of the fit
        print(f"  warning: no line-ups for match {fx['id']}: {exc}")
        return None
    lineups["score"] = fx["score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lineups, ensure_ascii=False))
    return lineups


def load_matches():
    """All league matches as plain dicts; played ones carry goals and line-ups."""
    fixtures = _fetch_fixtures()
    played = [f for f in fixtures if _parse_score(f["score"])[0] is not None]
    with ThreadPoolExecutor(max_workers=4) as pool:
        lineups = dict(zip((f["id"] for f in played), pool.map(_lineups_cached, played)))
    return [to_match(f, lineups.get(f["id"])) for f in sorted(fixtures, key=lambda f: (f["dateStart"], f["id"]))]


def to_match(f, lu):
    """One fixture as a plain dict; played ones carry goals and (if available) line-ups."""
    hg, ag = _parse_score(f["score"])
    return {
        "id": f["id"],
        "round": f["round"],
        "ts": f["dateStart"],
        "league": f["leagueName"],
        "home": {"id": f["home"]["team"]["id"], "name": f["home"]["team"]["name"]},
        "away": {"id": f["visiting"]["team"]["id"], "name": f["visiting"]["team"]["name"]},
        "hg": hg,
        "ag": ag,
        "forfeit": bool(f.get("contumacy")),  # awarded, not played: counts in the table, not in the fit
        "home_players": lu["home"] if lu else None,
        "away_players": lu["visiting"] if lu else None,
    }
