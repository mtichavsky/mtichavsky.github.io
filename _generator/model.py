"""Small-football prediction model that accounts for who (and how many) played.

Two stages, both fitted on the finished matches:

1. Goal model. Poisson regression: goals of team i against team j
       log(rate) = mu + home_adv + attack_i - defence_j + b_n*(n_i - n_j) + b_c*(c_i - c_j)
   n = players who played, c = share of the team's 6 regulars who played (see `core_sets`).
   Team effects are ridge-shrunk because 6-7 matches per team is very little data.

2. Attendance model. Teams do not field the same side every week, so for a *future* match we
   predict n and c per team: each team's habit, plus how they react to the opponent's strength
   and to how evenly matched the two sides are. The goal model then uses those predictions.

Parameter uncertainty comes from bootstrapping the finished matches; each refit also simulates
the rest of the season, which gives the final-table probabilities.
"""
from collections import Counter, defaultdict
from math import exp, lgamma, log

import numpy as np
from scipy.optimize import minimize

CORE_SIZE = 6  # small football is 5+1 on the pitch
N_MIN, N_MAX = 6, 10  # headcounts outside this range are clipped (diminishing effect)
RIDGE_TEAM = 10.0  # shrinkage of attack/defence effects
RIDGE_COEF = 10.0  # shrinkage of the headcount / core coefficients
RIDGE_HOME = 1.0
RIDGE_ATT_SLOPE = 2.0
GRID = 16  # score grid used for 1X2 probabilities


def finished(matches):
    return [m for m in matches if m["hg"] is not None and m["home_players"] and m["away_players"]]


def teams_of(matches):
    names = {}
    for m in matches:
        names[m["home"]["id"]] = m["home"]["name"]
        names[m["away"]["id"]] = m["away"]["name"]
    ids = sorted(names, key=names.get)
    return ids, names


def core_sets(matches):
    """Per team: the 6 most frequent players (ties: more goals) and everyone's appearance counts."""
    apps, goals, names = defaultdict(Counter), defaultdict(Counter), {}
    games = Counter()
    for m in matches:
        for side, plist in (("home", m["home_players"]), ("away", m["away_players"])):
            tid = m[side]["id"]
            games[tid] += 1
            for p in plist:
                apps[tid][p["id"]] += 1
                goals[tid][p["id"]] += p["goals"]
                names[p["id"]] = p["name"]
    core = {tid: sorted(c, key=lambda p: (-c[p], -goals[tid][p]))[:CORE_SIZE] for tid, c in apps.items()}
    return core, apps, games, names


def squad_features(match, side, core):
    """(n, c): headcount (clipped) and share of the team's regulars who played."""
    players = match[f"{side}_players"]
    regulars = core.get(match[side]["id"], [])
    played = {p["id"] for p in players}
    n = min(max(len(players), N_MIN), N_MAX)
    c = sum(p in played for p in regulars) / max(len(regulars), 1)
    return n, c


class Rows:
    """Team-match observations: one row per team per finished match."""

    def __init__(self, matches, tindex, core):
        cols = defaultdict(list)
        for m in matches:
            f = {s: squad_features(m, s, core) for s in ("home", "away")}
            for me, op, goals, home in (("home", "away", m["hg"], 1), ("away", "home", m["ag"], 0)):
                cols["i"].append(tindex[m[me]["id"]])
                cols["j"].append(tindex[m[op]["id"]])
                cols["goals"].append(goals)
                cols["home"].append(home)
                cols["n"].append(f[me][0])
                cols["c"].append(f[me][1])
                cols["n_opp"].append(f[op][0])
                cols["c_opp"].append(f[op][1])
        for k, v in cols.items():
            setattr(self, k, np.array(v, dtype=float if k not in ("i", "j") else int))
        self.size = len(self.goals)
        # Squads are modelled as deviations from the team's own usual squad: a team's typical
        # headcount/core share is part of its strength, the goal model only sees "bigger or
        # stronger than usual this week".
        T = len(tindex)
        self.mean_n, self.mean_c = np.full(T, self.n.mean()), np.full(T, self.c.mean())
        for t in range(T):
            mine = self.i == t
            if mine.any():
                self.mean_n[t], self.mean_c[t] = self.n[mine].mean(), self.c[mine].mean()
        self.n_dev = self.n - self.mean_n[self.i]
        self.c_dev = self.c - self.mean_c[self.i]
        pair = np.arange(self.size) ^ 1  # rows come in (home, away) pairs
        self.n_dev_opp, self.c_dev_opp = self.n_dev[pair], self.c_dev[pair]

    def take(self, idx):
        sub = object.__new__(Rows)
        for k, v in vars(self).items():
            setattr(sub, k, v[idx] if isinstance(v, np.ndarray) and len(v) == self.size else v)
        sub.size = len(idx)
        return sub


def _design(T, home, i, j, dn, dc):
    X = np.zeros((len(i), 4 + 2 * T))
    X[:, 0] = 1
    X[:, 1] = home
    X[:, 2] = dn
    X[:, 3] = dc
    X[np.arange(len(i)), 4 + i] = 1
    X[np.arange(len(i)), 4 + T + j] = -1
    return X


class GoalModel:
    def __init__(self, T, use_lineup=True):
        self.T, self.use_lineup = T, use_lineup

    def fit(self, rows):
        T = self.T
        dn = (rows.n_dev - rows.n_dev_opp) if self.use_lineup else np.zeros(rows.size)
        dc = (rows.c_dev - rows.c_dev_opp) if self.use_lineup else np.zeros(rows.size)
        X = _design(T, rows.home, rows.i, rows.j, dn, dc)
        pen = np.r_[0, RIDGE_HOME, RIDGE_COEF, RIDGE_COEF, np.full(2 * T, RIDGE_TEAM)]
        if not self.use_lineup:
            pen[2:4] = 1e6
        y = rows.goals
        theta0 = np.zeros(X.shape[1])
        theta0[0] = log(max(y.mean(), 0.1))

        def objective(th):
            eta = np.clip(X @ th, -8, 8)
            lam = np.exp(eta)
            return (lam - y * eta).sum() + 0.5 * (pen * th * th).sum(), X.T @ (lam - y) + pen * th

        res = minimize(objective, theta0, jac=True, method="L-BFGS-B")
        th = res.x
        self.mu, self.hfa, self.bn, self.bc = th[:4]
        self.att, self.dfs = th[4:4 + T], th[4 + T:]
        return self

    @property
    def strength(self):
        return self.att + self.dfs

    def rates(self, hi, ai, dev_h, dev_a):
        """Expected goals (home, away) given each side's (n, c) deviation from its usual squad."""
        dn, dc = dev_h[0] - dev_a[0], dev_h[1] - dev_a[1]
        if not self.use_lineup:
            dn = dc = 0.0
        eta_h = self.mu + self.hfa + self.att[hi] - self.dfs[ai] + self.bn * dn + self.bc * dc
        eta_a = self.mu + self.att[ai] - self.dfs[hi] - self.bn * dn - self.bc * dc
        return np.exp(eta_h), np.exp(eta_a)


class AttendanceModel:
    """Predicts how a team's squad differs from its usual one, given who it is playing."""

    def fit(self, rows, strength):
        self.strength = strength
        self.sd = max(strength.std(), 1e-6)
        self.mean_strength = strength.mean()
        opp, close = self._z(strength[rows.i], strength[rows.j])
        X = np.stack([opp, close], axis=1)
        A = X.T @ X + RIDGE_ATT_SLOPE * np.eye(2)
        self.coef = {name: np.linalg.solve(A, X.T @ y)
                     for name, y in (("n", rows.n_dev), ("c", rows.c_dev))}
        return self

    def _z(self, own, opp):
        return (opp - self.mean_strength) / self.sd, -np.abs(own - opp) / self.sd

    def predict(self, i, j):
        """Expected (n, c) deviation for team index i facing team index j."""
        z = np.stack(self._z(self.strength[i], self.strength[j]), axis=1)
        return z @ self.coef["n"], z @ self.coef["c"]


def outcome_probs(lam_h, lam_a):
    """P(home win, draw, away win) from independent Poisson goals."""
    k = np.arange(GRID)
    logfact = np.array([lgamma(x + 1) for x in k])

    def pmf(lam):
        lam = np.atleast_1d(lam)[:, None]
        return np.exp(-lam + k * np.log(lam) - logfact)

    joint = pmf(lam_h)[:, :, None] * pmf(lam_a)[:, None, :]
    diff = k[:, None] - k[None, :]
    probs = np.stack([(joint * (diff > 0)).sum((1, 2)), (joint * (diff == 0)).sum((1, 2)),
                      (joint * (diff < 0)).sum((1, 2))], axis=1)
    return probs / probs.sum(1, keepdims=True)


class Fit:
    """Goal model + attendance model fitted on one set of finished matches."""

    def __init__(self, matches, tindex, core, use_lineup=True, idx=None):
        rows = Rows(matches, tindex, core)
        if idx is not None:
            rows = rows.take(idx)
        self.mean_n, self.mean_c = rows.mean_n, rows.mean_c
        self.goal = GoalModel(len(tindex), use_lineup).fit(rows)
        self.attendance = AttendanceModel().fit(rows, self.goal.strength)

    def predict(self, hi, ai):
        """Expected goals and squads (n, c) for fixtures given as arrays of team indices."""
        dev_h = self.attendance.predict(hi, ai)
        dev_a = self.attendance.predict(ai, hi)
        lam_h, lam_a = self.goal.rates(hi, ai, dev_h, dev_a)
        sq_h = (self.mean_n[hi] + dev_h[0], np.clip(self.mean_c[hi] + dev_h[1], 0, 1))
        sq_a = (self.mean_n[ai] + dev_a[0], np.clip(self.mean_c[ai] + dev_a[1], 0, 1))
        return lam_h, lam_a, sq_h, sq_a


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
    """Play the remaining fixtures `sims` times; returns final points, GD and rank arrays."""
    T, M = len(table), len(hi)
    hg = rng.poisson(lam_h, size=(sims, M))
    ag = rng.poisson(lam_a, size=(sims, M))
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


def season_forecast(matches, ids, boots=150, sims_per_boot=40, seed=7):
    """Fit on all finished matches, then bootstrap for uncertainty and simulate the season."""
    rng = np.random.default_rng(seed)
    tindex = {t: k for k, t in enumerate(ids)}
    done = finished(matches)
    core = core_sets(done)[0]
    point = Fit(done, tindex, core)

    todo = [m for m in matches if m["hg"] is None]
    hi = np.array([tindex[m["home"]["id"]] for m in todo])
    ai = np.array([tindex[m["away"]["id"]] for m in todo])
    table = standings(matches, ids)

    probs, lam_h, lam_a, n_h, c_h, n_a, c_a = [], [], [], [], [], [], []
    all_points, all_ranks = [], []
    for _ in range(boots):
        match_idx = rng.integers(0, len(done), len(done))  # resample whole matches
        idx = np.concatenate([2 * match_idx, 2 * match_idx + 1])
        fit = Fit(done, tindex, core, idx=idx)
        lh, la, sq_h, sq_a = fit.predict(hi, ai)
        probs.append(outcome_probs(lh, la))
        lam_h.append(lh)
        lam_a.append(la)
        n_h.append(sq_h[0]), c_h.append(sq_h[1]), n_a.append(sq_a[0]), c_a.append(sq_a[1])
        pts, rk = simulate_season(table, hi, ai, lh, la, rng, sims_per_boot)
        all_points.append(pts)
        all_ranks.append(rk)
    points, ranks = np.vstack(all_points), np.vstack(all_ranks)
    fixtures = [{
        "match": m, "probs": p, "xg_h": a, "xg_a": b, "n_h": nh, "c_h": ch, "n_a": na, "c_a": ca,
    } for m, p, a, b, nh, ch, na, ca in zip(
        todo, np.mean(probs, 0), np.mean(lam_h, 0), np.mean(lam_a, 0),
        np.mean(n_h, 0), np.mean(c_h, 0), np.mean(n_a, 0), np.mean(c_a, 0))]
    return {
        "fit": point, "table": table, "fixtures": fixtures, "n_rows": 2 * len(done),
        "exp_points": points.mean(0), "sd_points": points.std(0), "sims": len(points),
        "q10": np.percentile(points, 10, axis=0), "q90": np.percentile(points, 90, axis=0),
        "rank_dist": np.stack([(ranks == r).mean(0) for r in range(len(ids))], axis=1),
        "core": core_sets(done),
    }


def attendance_effects(fit):
    """Plain-language numbers describing what the fitted models learned."""
    g, a = fit.goal, fit.attendance
    return {
        "per_player": exp(g.bn) - 1,  # change in goal rate per extra player
        "full_core": exp(g.bc * 0.5) - 1,  # change when 3 more of the 6 regulars play
        "opp_n": a.coef["n"][0],
        "opp_c": a.coef["c"][0],
        "close_n": a.coef["n"][1],
        "close_c": a.coef["c"][1],
        "home_adv": exp(g.hfa) - 1,
    }


def backtest(matches, ids, first_round=3):
    """Walk-forward: predict each round from earlier rounds only, never seeing its line-ups."""
    tindex = {t: k for k, t in enumerate(ids)}
    done = finished(matches)
    rounds = sorted({m["round"] for m in done})
    results = defaultdict(list)  # variant -> [(probs, outcome)]
    for r in rounds:
        train = [m for m in done if m["round"] < r]
        test = [m for m in done if m["round"] == r]
        if r < first_round or not test:
            continue
        core = core_sets(train)[0]
        hi = np.array([tindex[m["home"]["id"]] for m in test])
        ai = np.array([tindex[m["away"]["id"]] for m in test])
        out = np.array([0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2 for m in test])
        for name, use in (("full", True), ("no_lineup", False)):
            lh, la, *_ = Fit(train, tindex, core, use_lineup=use).predict(hi, ai)
            results[name] += list(zip(outcome_probs(lh, la), out))
        base = np.bincount([0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2 for m in train],
                           minlength=3) + 1.0
        results["base_rates"] += [(base / base.sum(), o) for o in out]
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
