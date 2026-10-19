"""Small-football prediction model with a memory of earlier seasons.

1. Starting estimate. Before a team has played, its strength comes from last season (mapped across
   promotion/relegation, see history.py) and, as the season goes on, partly from the players it
   actually fields: what their teams were worth last season.

2. Goal model. Poisson regression on this season's finished matches:
       log(rate) = mu + home_adv + attack_i - defence_j
   Team effects are ridge-shrunk towards the starting estimate, so early on the model leans on
   history and later on this season's results.

3. Outcomes. Goals vary about twice as much as a Poisson allows (small squads, rotating line-ups),
   so match and season outcomes use a negative binomial with the same mean. Match probabilities come
   from the single best fit; the season simulation adds parameter uncertainty by bootstrapping the
   finished matches (each refit simulates the rest of the season).

All settings below were chosen on seasons 2022/23-2023/24 of leagues 4-7 and checked on 2024/25 to
2026/27. Line-up terms (headcount, regulars) were tried and dropped: they did not improve predictions.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import exp, log

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

RIDGE_TEAM = 10.0  # shrinkage of attack/defence towards the starting estimate
RIDGE_HOME = 1.0
RIDGE_MU = 20.0  # shrinkage of the league's goal rate towards earlier seasons
PRIOR_TEAM = 0.5  # weight of last season's team strength in the starting estimate
PRIOR_PLAYERS = 0.5  # weight of the fielded players' history, reached after PLAYER_RAMP matches
PLAYER_RAMP = 4
DISPERSION = 3.0  # negative-binomial shape: variance = mean + mean^2 / DISPERSION
GRID = 25  # score grid used for 1X2 probabilities


def finished(matches):
    """Matches actually played (forfeits are awarded, so they say nothing about strength)."""
    return [m for m in matches if m["hg"] is not None and not m.get("forfeit")]


def outcome(m):
    return 0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2


def match_day(m):
    return datetime.fromtimestamp(m["ts"], timezone.utc).date()


def teams_of(matches):
    names = {}
    for m in matches:
        names[m["home"]["id"]] = m["home"]["name"]
        names[m["away"]["id"]] = m["away"]["name"]
    ids = sorted(names, key=names.get)
    return ids, names


def starting_strength(matches, ids, hist, weight_team=PRIOR_TEAM, weight_players=PRIOR_PLAYERS):
    """Per team: expected strength (attack + defence) from earlier seasons and the players used so far."""
    apps, played = defaultdict(Counter), Counter()
    for m in finished(matches):
        for side in ("home", "away"):
            tid = m[side]["id"]
            played[tid] += 1
            apps[tid].update(p["id"] for p in m[f"{side}_players"] or [])
    out = np.zeros(len(ids))
    for k, t in enumerate(ids):
        team = hist.team_prior(t)
        v = weight_team * (team if team is not None else hist.new_team)
        if weight_players and apps[t]:
            vals = [(hist.player_value(p), n) for p, n in apps[t].items()]
            roster = sum(n * (hist.new_team if x is None else x) for x, n in vals) / sum(apps[t].values())
            w = weight_players * min(played[t] / PLAYER_RAMP, 1.0)
            v = (1 - w) * v + w * roster
        out[k] = v
    return out


class GoalModel:
    def __init__(self, T, prior, mu0, ridge_team=RIDGE_TEAM):
        self.T, self.prior, self.mu0, self.ridge_team = T, prior, mu0, ridge_team

    def fit(self, i, j, home, goals):
        T = self.T
        X = np.zeros((len(i), 2 + 2 * T))
        X[:, 0] = 1
        X[:, 1] = home
        X[np.arange(len(i)), 2 + i] = 1
        X[np.arange(len(i)), 2 + T + j] = -1
        mu0 = self.mu0 if self.mu0 is not None else log(max(goals.mean(), 0.1))
        centre = np.r_[mu0, 0, self.prior / 2, self.prior / 2]  # strength split evenly into attack and defence
        pen = np.r_[RIDGE_MU if self.mu0 is not None else 0.0, RIDGE_HOME, np.full(2 * T, self.ridge_team)]

        def objective(th):
            eta = np.clip(X @ th, -8, 8)
            lam = np.exp(eta)
            d = th - centre
            return (lam - goals * eta).sum() + 0.5 * (pen * d * d).sum(), X.T @ (lam - goals) + pen * d

        th = minimize(objective, centre.copy(), jac=True, method="L-BFGS-B").x
        self.mu, self.hfa = th[:2]
        self.att, self.dfs = th[2:2 + T], th[2 + T:]
        return self

    def from_prior(self):
        """No matches yet: the starting estimate as is."""
        self.mu = self.mu0 if self.mu0 is not None else log(3.3)
        self.hfa = 0.0
        self.att, self.dfs = self.prior / 2, self.prior / 2
        return self

    @property
    def strength(self):
        return self.att + self.dfs

    def rates(self, hi, ai):
        """Expected goals (home, away)."""
        return (np.exp(self.mu + self.hfa + self.att[hi] - self.dfs[ai]),
                np.exp(self.mu + self.att[ai] - self.dfs[hi]))


class Fit:
    """Goal model fitted on one set of finished matches (or a bootstrap resample of them)."""

    def __init__(self, matches, tindex, prior=None, mu0=None, idx=None, ridge_team=RIDGE_TEAM):
        T = len(tindex)
        done = finished(matches)
        prior = np.zeros(T) if prior is None else prior
        self.goal = GoalModel(T, prior, mu0, ridge_team)
        if not done:
            self.goal.from_prior()
            return
        hi = np.array([tindex[m["home"]["id"]] for m in done])
        ai = np.array([tindex[m["away"]["id"]] for m in done])
        hg = np.array([m["hg"] for m in done], dtype=float)
        ag = np.array([m["ag"] for m in done], dtype=float)
        # one row per team per match
        i, j, home, goals = np.r_[hi, ai], np.r_[ai, hi], np.r_[np.ones(len(done)), np.zeros(len(done))], np.r_[hg, ag]
        if idx is not None:
            i, j, home, goals = i[idx], j[idx], home[idx], goals[idx]
        self.goal.fit(i, j, home, goals)
        self.n_rows = len(goals)

    def predict(self, hi, ai):
        return self.goal.rates(hi, ai)


def nb_pmf(lam, r=DISPERSION):
    """Negative binomial with mean lam and variance lam + lam^2 / r, on 0..GRID-1."""
    k = np.arange(GRID)
    lam = np.atleast_1d(lam)[:, None]
    p = r / (r + lam)
    return np.exp(gammaln(k + r) - gammaln(r) - gammaln(k + 1) + r * np.log(p) + k * np.log1p(-p))


def outcome_probs(lam_h, lam_a):
    """P(home win, draw, away win) from independent negative-binomial goals."""
    k = np.arange(GRID)
    joint = nb_pmf(lam_h)[:, :, None] * nb_pmf(lam_a)[:, None, :]
    diff = k[:, None] - k[None, :]
    probs = np.stack([(joint * (diff > 0)).sum((1, 2)), (joint * (diff == 0)).sum((1, 2)),
                      (joint * (diff < 0)).sum((1, 2))], axis=1)
    return probs / probs.sum(1, keepdims=True)


def standings(matches, ids):
    tindex = {t: k for k, t in enumerate(ids)}
    table = np.zeros((len(ids), 7), dtype=int)  # P W D L GF GA Pts
    for m in matches:
        if m["hg"] is None:
            continue
        for me, op, gf, ga in (("home", "away", m["hg"], m["ag"]), ("away", "home", m["ag"], m["hg"])):
            r = table[tindex[m[me]["id"]]]
            r[0] += 1
            r[1 + (0 if gf > ga else 1 if gf == ga else 2)] += 1
            r[4] += gf
            r[5] += ga
            r[6] += 3 if gf > ga else 1 if gf == ga else 0
    return table


def simulate_season(table, hi, ai, lam_h, lam_a, rng, sims):
    """Play the remaining fixtures `sims` times; returns final points and rank arrays."""
    T, M = len(table), len(hi)
    hg = rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + lam_h), size=(sims, M))
    ag = rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + lam_a), size=(sims, M))
    pts_h = 3 * (hg > ag) + (hg == ag)
    pts_a = 3 * (ag > hg) + (hg == ag)
    H = np.zeros((M, T))
    A = np.zeros((M, T))
    H[np.arange(M), hi] = 1
    A[np.arange(M), ai] = 1
    points = table[:, 6] + pts_h @ H + pts_a @ A
    gd = table[:, 4] - table[:, 5] + (hg - ag) @ H + (ag - hg) @ A
    gf = table[:, 4] + hg @ H + ag @ A
    key = points * 1e6 + gd * 1e3 + gf + rng.random((sims, T)) * 0.5
    order = np.argsort(-key, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(T)[None, :].repeat(sims, 0), axis=1)
    return points, ranks


def season_forecast(matches, ids, hist, boots=150, sims_per_boot=40, seed=7):
    """Fit on all finished matches, then bootstrap for uncertainty and simulate the season."""
    rng = np.random.default_rng(seed)
    tindex = {t: k for k, t in enumerate(ids)}
    done = finished(matches)
    prior = starting_strength(matches, ids, hist)
    mu0 = hist.mu_prior()
    point = Fit(matches, tindex, prior, mu0)

    todo = [m for m in matches if m["hg"] is None]
    hi = np.array([tindex[m["home"]["id"]] for m in todo], dtype=int)
    ai = np.array([tindex[m["away"]["id"]] for m in todo], dtype=int)
    table = standings(matches, ids)

    all_points, all_ranks = [], []
    for _ in range(boots):
        idx = None
        if done:
            match_idx = rng.integers(0, len(done), len(done))  # resample whole matches
            idx = np.concatenate([match_idx, match_idx + len(done)])
        lh, la = Fit(matches, tindex, prior, mu0, idx=idx).predict(hi, ai)
        pts, rk = simulate_season(table, hi, ai, lh, la, rng, sims_per_boot)
        all_points.append(pts)
        all_ranks.append(rk)
    points, ranks = np.vstack(all_points), np.vstack(all_ranks)
    lam_h, lam_a = point.predict(hi, ai)
    probs = outcome_probs(lam_h, lam_a) if len(todo) else np.zeros((0, 3))
    fixtures = [{"match": m, "probs": p, "xg_h": a, "xg_a": b} for m, p, a, b in zip(todo, probs, lam_h, lam_a)]
    return {
        "fit": point, "prior": prior, "table": table, "fixtures": fixtures, "n_rows": 2 * len(done),
        "exp_points": points.mean(0), "sd_points": points.std(0), "sims": len(points),
        "q10": np.percentile(points, 10, axis=0), "q90": np.percentile(points, 90, axis=0),
        "rank_dist": np.stack([(ranks == r).mean(0) for r in range(len(ids))], axis=1),
    }


def effects(fit, hist):
    """Plain-language numbers describing what the model learned."""
    g = hist.cal
    return {
        "home_adv": exp(fit.goal.hfa) - 1,
        "per_league": exp(g["gamma"]) - 1,  # goal-rate gap between neighbouring leagues
        "carry_over": g["rho"],  # share of last season's (relative) strength kept
        "transitions": g["n"],
        "seasons": len(hist.seasons),
    }


def backtest(matches, ids, hist):
    """Walk-forward: predict each match day from matches played on earlier days only."""
    tindex = {t: k for k, t in enumerate(ids)}
    done = finished(matches)
    mu0 = hist.mu_prior()
    results = defaultdict(list)  # variant -> [(probs, outcome)]
    for d in sorted({match_day(m) for m in done}):
        train = [m for m in done if match_day(m) < d]
        test = [m for m in done if match_day(m) == d]
        hi = np.array([tindex[m["home"]["id"]] for m in test])
        ai = np.array([tindex[m["away"]["id"]] for m in test])
        out = np.array([outcome(m) for m in test])
        for name, prior in (("model", starting_strength(train, ids, hist)), ("no_history", np.zeros(len(ids)))):
            lh, la = Fit(train, tindex, prior, mu0).predict(hi, ai)
            results[name] += list(zip(outcome_probs(lh, la), out))
        results["base_rates"] += [(hist.base_rates(), o) for o in out]
    scores = {}
    for name, items in results.items():
        P = np.array([p for p, _ in items])
        y = np.array([o for _, o in items])
        onehot = np.eye(3)[y]
        scores[name] = {
            "logloss": float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean()),
            "brier": float(((P - onehot) ** 2).sum(1).mean()),
            "accuracy": float((P.argmax(1) == y).mean()),
            "n": len(y),
        }
    return scores
