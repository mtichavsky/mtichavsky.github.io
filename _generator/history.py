"""Earlier seasons of the men's leagues 1-7: where each team and player played and how strong it was.

Used to give every team a starting estimate before it has played this season (see model.py).
Everything comes from seasons before the current one, so it never peeks at this season's results.

Cache: data/history/index.json (season/league index), data/history/fixtures/<league>.json and the
shared line-up cache data/lineups/. Finished seasons are downloaded once.
"""
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import least_squares

import fetch
import model

HIST = fetch.DATA / "history"
SEASONS_BACK = 5
LEVEL = re.compile(r"^([1-7])\. ")  # "6. Barber Moliére liga" -> 6; cup, women's, youth leagues have no number
RIDGE_FULL_SEASON = 3.0  # a full season is ~20 matches per team: little shrinkage needed
MIN_TRANSITIONS = 30


def _season_index():
    path = HIST / "index.json"
    try:
        html = fetch._get(f"{fetch.BASE}/tabulka/")
        blob = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
        seasons = json.loads(blob[1])["props"]["pageProps"]["seasons"]
    except Exception as exc:
        if not path.exists():
            raise
        print(f"  warning: season index not refreshed ({exc}); using cache")
        return json.loads(path.read_text())
    index = [{"season": s["name"], "leagues": [
        {"id": lg["id"], "name": lg["name"], "level": int(LEVEL.match(lg["name"])[1])}
        for lg in s["leagues"]
        if LEVEL.match(lg["name"]) and "turnaj" not in lg["name"] and "Univerzitní" not in lg["name"]]}
        for s in seasons]  # newest first
    HIST.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return index


def _league_fixtures(league_id):
    path = HIST / "fixtures" / f"{league_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    fx = fetch._fetch_fixtures(league_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fx, ensure_ascii=False))
    return fx


def load(league_id=fetch.LEAGUE_ID):
    """History for the season of `league_id`: the SEASONS_BACK seasons before it."""
    index = _season_index()
    k = next(n for n, s in enumerate(index) if any(lg["id"] == league_id for lg in s["leagues"]))
    current = index[k]
    level = next(lg["level"] for lg in current["leagues"] if lg["id"] == league_id)
    past = index[k + 1:k + 1 + SEASONS_BACK]

    data, todo = {}, []
    for s in past:
        for lg in s["leagues"]:
            fx = [f for f in _league_fixtures(lg["id"]) if f.get("phaseName") == "Tabulka"]
            data[(s["season"], lg["level"])] = fx
            todo += [f for f in fx if fetch._parse_score(f["score"])[0] is not None]
    with ThreadPoolExecutor(max_workers=2) as pool:  # the site starts timing out above ~2 parallel requests
        lineups = dict(zip((f["id"] for f in todo), pool.map(fetch._lineups_cached, todo)))
    matches = {key: [fetch.to_match(f, lineups.get(f["id"])) for f in fx] for key, fx in data.items()}
    return History(matches, [s["season"] for s in reversed(past)], current["season"], level)


class History:
    """Full-season team strengths per league, league-level offsets learnt from teams that moved
    between leagues, and per-player appearances."""

    def __init__(self, data, seasons, current, level):
        self.seasons, self.current, self.level = seasons, current, level  # seasons: oldest first
        self.ratings, self.mu = {}, {}  # (season, level) -> {team: strength}; -> log base rate
        self.team_level = defaultdict(dict)  # team -> {season: level}
        self.player_apps = defaultdict(lambda: defaultdict(Counter))  # player -> season -> {(team, level): n}
        self.outcomes = np.ones(3)
        for (s, lv), ms in data.items():
            done = model.finished(ms)
            if lv >= 4:
                self.outcomes += np.bincount([model.outcome(m) for m in done], minlength=3)
            for m in ms:
                self.team_level[m["home"]["id"]][s] = lv
                self.team_level[m["away"]["id"]][s] = lv
            for m in done:
                for side in ("home", "away"):
                    for p in m[f"{side}_players"] or []:
                        self.player_apps[p["id"]][s][(m[side]["id"], lv)] += 1
            if len(done) < 20:
                continue
            ids, _ = model.teams_of(done)
            fit = model.Fit(done, {t: k for k, t in enumerate(ids)}, ridge_team=RIDGE_FULL_SEASON, mu0=None)
            self.ratings[(s, lv)] = dict(zip(ids, fit.goal.strength))
            self.mu[(s, lv)] = fit.goal.mu
        self.cal = self._calibrate()
        self.new_team = self._new_team_value()

    def _prev(self, season):
        k = self.seasons.index(season) if season in self.seasons else len(self.seasons)
        return self.seasons[k - 1] if k else None

    def _calibrate(self):
        """Absolute strength A = R - gamma * (level - 4); next season A' = rho * A + c. Fitted on
        every team that played in two consecutive seasons (promoted, relegated or staying)."""
        pairs = []
        for (s, lv), r in self.ratings.items():
            k = self.seasons.index(s)
            if k + 1 >= len(self.seasons):
                continue
            s2 = self.seasons[k + 1]
            for t, R in r.items():
                lv2 = self.team_level[t].get(s2)
                if t in self.ratings.get((s2, lv2), {}):
                    pairs.append((lv, R, lv2, self.ratings[(s2, lv2)][t]))
        pairs = np.array(pairs).reshape(-1, 4)
        if len(pairs) < MIN_TRANSITIONS:  # too little history: a fixed guess
            return {"gamma": 0.2, "rho": 0.5, "c": 0.0, "n": len(pairs)}

        def resid(x):
            gamma, rho, c = x
            return pairs[:, 3] - gamma * (pairs[:, 2] - 4) - (rho * (pairs[:, 1] - gamma * (pairs[:, 0] - 4)) + c)

        gamma, rho, c = least_squares(resid, [0.2, 0.5, 0.0]).x
        return {"gamma": gamma, "rho": rho, "c": c, "n": len(pairs)}

    def _carry(self, R, from_level):
        """Strength R in league `from_level` last season -> expected strength in this league now."""
        g = self.cal
        return g["rho"] * (R - g["gamma"] * (from_level - 4)) + g["c"] + g["gamma"] * (self.level - 4)

    def _new_team_value(self):
        """Average strength of teams that played no league the season before."""
        vals = [R for (s, lv), r in self.ratings.items() if self._prev(s) for t, R in r.items()
                if self._prev(s) not in self.team_level[t]]
        return float(np.mean(vals)) if vals else 0.0

    def team_prior(self, team):
        """Last season's strength mapped into this league, or None for a team new to the leagues."""
        p = self.seasons[-1] if self.seasons else None
        lv = self.team_level[team].get(p)
        if lv is None or team not in self.ratings.get((p, lv), {}):
            return None
        return self._carry(self.ratings[(p, lv)][team], lv)

    def player_value(self, pid):
        """Last-season strength of the teams a player played for (appearance-weighted), mapped into
        this league; None for players who did not play last season."""
        p = self.seasons[-1] if self.seasons else None
        tot = w = 0.0
        for (team, lv), n in self.player_apps.get(pid, {}).get(p, {}).items():
            R = self.ratings.get((p, lv), {}).get(team)
            if R is not None:
                tot += n * self._carry(R, lv)
                w += n
        return tot / w if w else None

    def mu_prior(self):
        vals = [v for (s, lv), v in self.mu.items() if lv == self.level] or list(self.mu.values())
        return float(np.mean(vals)) if vals else None

    def base_rates(self):
        """Home/draw/away frequencies in leagues 4 and lower over the history seasons."""
        return self.outcomes / self.outcomes.sum()
