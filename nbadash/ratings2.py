"""Component team ratings + home-arena effect (ported from the lab's core lane, numerically identical).

* Per-game offense stats for each side: points/100, 2PA/3PA/FTA per 100 possessions, 2P%, 3P%, FT%, TOV/100, OREB%.
* League mean and home edge for each stat: slow EWMAs over games (as of each game).
* Team offense/defense ratings for every stat solved jointly (ridge-shrunk towards 0.55 x last season,
  shrinkage set per stat from noise/skill variance: e.g. opponent 3P% allowed is mostly noise), re-solved
  before every game date with exponentially decaying weights (40-day half-life). Pace is solved the same way.
* st_c: structural total from the components; st_s: from points/100 directly.
* venue: shrunk, slowly decaying mean of (total - st_s) in earlier games at the home arena.
`build()` walks history and returns per-game features plus the state needed to score upcoming games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["ortg", "a2", "a3", "af", "p2", "p3", "ft", "tov", "orb"]
K = {"ortg": (10, 19), "a2": (3, 9), "a3": (2, 15), "af": (11, 22), "p2": (14, 20),
     "p3": (46, 128), "ft": (13, 700), "tov": (12, 8), "orb": (12, 20), "pace": (6, 6)}
HL_DAYS, RHO, ALPHA_LEAGUE = 40.0, 0.55, 1 / 250
VENUE_HL, VENUE_K = 80.0, 200.0
FEATURES = ["st_c", "st_s", "venue"]


def side_obs(g: pd.DataFrame) -> dict:
    out = {}
    for s, o in (("h", "a"), ("a", "h")):
        fga, fgm = g[f"{s}_field_goals_attempted"], g[f"{s}_field_goals_made"]
        f3a, f3m = g[f"{s}_three_point_field_goals_attempted"], g[f"{s}_three_point_field_goals_made"]
        fta, ftm = g[f"{s}_free_throws_attempted"], g[f"{s}_free_throws_made"]
        poss = g.poss
        d = pd.DataFrame(index=g.index)
        d["ortg"] = 100 * g[f"{s}_team_score"] / poss
        d["a2"] = 100 * (fga - f3a) / poss
        d["a3"] = 100 * f3a / poss
        d["af"] = 100 * fta / poss
        d["p2"] = (fgm - f3m) / (fga - f3a).clip(lower=1)
        d["p3"] = f3m / f3a.clip(lower=1)
        d["ft"] = ftm / fta.clip(lower=1)
        d["tov"] = 100 * g[f"{s}_tov"] / poss
        d["orb"] = g[f"{s}_offensive_rebounds"] / (g[f"{s}_offensive_rebounds"] + g[f"{o}_defensive_rebounds"]).clip(lower=1)
        out[s] = d
    return out


def league_track(g: pd.DataFrame, so: dict) -> tuple[pd.DataFrame, dict]:
    """League mean and home edge per stat (+ pace mean), as of each game; and the final values."""
    stats = STATS + ["pace"]
    neutral = g.neutral_site.fillna(False).astype(bool).values
    yh = np.column_stack([so["h"][s].values for s in STATS] + [g.obs_pace.values])
    ya = np.column_stack([so["a"][s].values for s in STATS] + [g.obs_pace.values])
    is_pace = np.array([s == "pace" for s in stats])
    L = np.nanmean((yh[:100] + ya[:100]) / 2, axis=0)
    HC = np.nanmean(yh[:100] - ya[:100], axis=0)
    HC[is_pace] = 0.0
    lg = np.zeros((len(g), len(stats)))
    hc = np.zeros((len(g), len(stats)))
    for i in range(len(g)):
        f = 0.0 if neutral[i] else 1.0
        lg[i], hc[i] = L, HC * f
        yy = np.where(np.isnan(yh[i]) | np.isnan(ya[i]), L, (yh[i] + ya[i]) / 2)
        L = L + ALPHA_LEAGUE * (yy - L)
        if not neutral[i]:
            dd = np.where(np.isnan(yh[i]) | np.isnan(ya[i]), HC, yh[i] - ya[i])
            HC = np.where(is_pace, 0.0, HC + ALPHA_LEAGUE * (dd - HC))
    cols = {}
    for j, s in enumerate(stats):
        cols[f"lg_{s}"] = lg[:, j]
        cols[f"hca_{s}"] = hc[:, j]
    return pd.DataFrame(cols, index=g.index), {"L": dict(zip(stats, L)), "HC": dict(zip(stats, HC))}


class Batch:
    """Jointly solved, shrunk, decayed offense/defense ratings for every stat (and pace)."""

    def __init__(self, teams: list[str]):
        self.teams = teams
        self.ix = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        S = len(STATS)
        self.T = T
        self.lam = np.zeros((S, 2 * T))
        for j, s in enumerate(STATS):
            self.lam[j, :T] = K[s][0]
            self.lam[j, T:] = K[s][1]
        self.lam_p = np.full(T, float(K["pace"][0]))
        self.A = np.zeros((S, 2 * T, 2 * T)); self.b = np.zeros((S, 2 * T))
        self.Ap = np.zeros((T, T)); self.bp = np.zeros(T)
        self.mu = np.zeros((S, 2 * T)); self.mup = np.zeros(T)
        self.beta = np.zeros((S, 2 * T)); self.betap = np.zeros(T)
        self.dl = 0.5 ** (1.0 / HL_DAYS)
        self.season, self.last_day = None, None
        self.eye = np.arange(2 * T)

    def _advance(self, day: int, season: int) -> None:
        if season != self.season:
            if self.season is not None:
                self.mu = RHO * self.beta
                self.mup = RHO * self.betap
            self.A[:] = 0; self.b[:] = 0; self.Ap[:] = 0; self.bp[:] = 0
            self.season, self.last_day = season, day
        f = self.dl ** (day - self.last_day)
        self.last_day = day
        self.A *= f; self.b *= f; self.Ap *= f; self.bp *= f

    def _solve(self) -> tuple[np.ndarray, np.ndarray]:
        M_ = self.A.copy()
        M_[:, self.eye, self.eye] += self.lam
        beta = np.linalg.solve(M_, (self.b + self.lam * self.mu)[..., None])[..., 0]
        betap = np.linalg.solve(self.Ap + np.diag(self.lam_p), self.bp + self.lam_p * self.mup)
        return beta, betap

    def run(self, g: pd.DataFrame, so: dict, lt: pd.DataFrame) -> pd.DataFrame:
        T = self.T
        hi = g.home.map(self.ix).values; ai = g.away.map(self.ix).values
        dates = pd.to_datetime(g.game_date).values.astype("datetime64[D]").astype(np.int64)
        season = g.season.values
        rh = np.column_stack([so["h"][s].values - lt[f"lg_{s}"].values - lt[f"hca_{s}"].values / 2 for s in STATS])
        ra = np.column_stack([so["a"][s].values - lt[f"lg_{s}"].values + lt[f"hca_{s}"].values / 2 for s in STATS])
        rp = g.obs_pace.values - lt["lg_pace"].values
        out = np.zeros((len(g), len(STATS), 4)); outp = np.zeros((len(g), 2))
        order = np.argsort(dates, kind="stable")
        ud, starts = np.unique(dates[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        for D, st, en in zip(ud, starts, ends):
            idx = order[st:en]
            self._advance(int(D), season[idx[0]])
            self.beta, self.betap = self._solve()
            h, a = hi[idx], ai[idx]
            out[idx, :, 0] = self.beta[:, h].T; out[idx, :, 1] = self.beta[:, T + h].T
            out[idx, :, 2] = self.beta[:, a].T; out[idx, :, 3] = self.beta[:, T + a].T
            outp[idx, 0] = self.betap[h]; outp[idx, 1] = self.betap[a]
            for (o, d_, r) in ((h, a, rh[idx]), (a, h, ra[idx])):
                od = T + d_
                np.add.at(self.A, (slice(None), o, o), 1.0); np.add.at(self.A, (slice(None), od, od), 1.0)
                np.add.at(self.A, (slice(None), o, od), 1.0); np.add.at(self.A, (slice(None), od, o), 1.0)
                np.add.at(self.b, (slice(None), o), r.T); np.add.at(self.b, (slice(None), od), r.T)
            np.add.at(self.Ap, (h, h), 1.0); np.add.at(self.Ap, (a, a), 1.0)
            np.add.at(self.Ap, (h, a), 1.0); np.add.at(self.Ap, (a, h), 1.0)
            np.add.at(self.bp, h, rp[idx]); np.add.at(self.bp, a, rp[idx])
        cols = {}
        for j, s in enumerate(STATS):
            for q, nm in enumerate(("h_off", "h_def", "a_off", "a_def")):
                cols[f"{nm}_{s}"] = out[:, j, q]
            cols[f"lg_{s}"] = lt[f"lg_{s}"].values
            cols[f"hca_{s}"] = lt[f"hca_{s}"].values
        cols["h_off_pace"] = outp[:, 0]; cols["a_off_pace"] = outp[:, 1]
        cols["lg_pace"] = lt["lg_pace"].values
        return pd.DataFrame(cols, index=g.index)

    def ratings_on(self, day: int, season: int) -> tuple[np.ndarray, np.ndarray]:
        """Ratings for games on a future date (copy of state; history state is left untouched)."""
        A, b, Ap, bp, mu, mup = self.A.copy(), self.b.copy(), self.Ap.copy(), self.bp.copy(), self.mu, self.mup
        if season != self.season:
            mu, mup = RHO * self.beta, RHO * self.betap
            A[:] = 0; b[:] = 0; Ap[:] = 0; bp[:] = 0
        else:
            f = self.dl ** (day - self.last_day)
            A *= f; b *= f; Ap *= f; bp *= f
        M_ = A.copy()
        M_[:, self.eye, self.eye] += self.lam
        beta = np.linalg.solve(M_, (b + self.lam * mu)[..., None])[..., 0]
        betap = np.linalg.solve(Ap + np.diag(self.lam_p), bp + self.lam_p * mup)
        return beta, betap


def comp(r: pd.DataFrame) -> pd.DataFrame:
    """st_c and st_s from a frame with h_off_*, h_def_*, a_off_*, a_def_*, lg_*, hca_* columns."""
    def ex(stat, side):
        o, d = ("h", "a") if side == "h" else ("a", "h")
        sgn = 0.5 if side == "h" else -0.5
        return r[f"lg_{stat}"] + r[f"{o}_off_{stat}"] + r[f"{d}_def_{stat}"] + sgn * r[f"hca_{stat}"]
    pace = r.lg_pace + r.h_off_pace + r.a_off_pace
    f = pd.DataFrame(index=r.index)
    oc, os_ = {}, {}
    for s in ("h", "a"):
        e = {st: ex(st, s) for st in STATS}
        oc[s] = 2 * e["a2"] * e["p2"] + 3 * e["a3"] * e["p3"] + e["af"] * e["ft"]
        os_[s] = e["ortg"]
    f["st_c"] = pace * (oc["h"] + oc["a"]) / 100
    f["st_s"] = pace * (os_["h"] + os_["a"]) / 100
    return f


class Venue:
    def __init__(self):
        self.S, self.W = {}, {}
        self.m = None

    def run(self, g: pd.DataFrame, st_s: pd.Series) -> np.ndarray:
        dl = 0.5 ** (1.0 / VENUE_HL)
        res = (g.total - st_s).values
        home = g.home.values
        neutral = g.neutral_site.fillna(False).astype(bool).values
        out = np.zeros(len(g))
        self.m = float(np.nanmean(res[:200]))
        for i in range(len(g)):
            t = home[i]
            if not neutral[i]:
                out[i] = self.S.get(t, 0.0) / (self.W.get(t, 0.0) + VENUE_K)
                self.S[t] = dl * self.S.get(t, 0.0) + (res[i] - self.m)
                self.W[t] = dl * self.W.get(t, 0.0) + 1.0
            self.m += (1 / 500) * (res[i] - self.m)
        return out

    def value(self, home: str, neutral: bool) -> float:
        return 0.0 if neutral else self.S.get(home, 0.0) / (self.W.get(home, 0.0) + VENUE_K)


def build(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """g: completed games in tip order with observed columns (model.add_observed) and box stats.
    Returns (features per game: game_id, st_c, st_s, venue) and the live state."""
    g = g.sort_values("tip", kind="stable").reset_index(drop=True)
    so = side_obs(g)
    lt, league = league_track(g, so)
    teams = sorted(set(g.home) | set(g.away))
    bt = Batch(teams)
    r = bt.run(g, so, lt)
    f = comp(r)
    vn = Venue()
    f["venue"] = vn.run(g, f.st_s)
    f.insert(0, "game_id", g.game_id.values)
    return f, {"batch": bt, "league": league, "venue": vn}


def upcoming(state: dict, games: pd.DataFrame) -> pd.DataFrame:
    """games: game_id, game_date, season, home, away, neutral_site -> st_c, st_s, venue."""
    bt, lg, vn = state["batch"], state["league"], state["venue"]
    T = bt.T
    rows = []
    for r in games.itertuples(index=False):
        day = int(np.datetime64(pd.Timestamp(r.game_date).date(), "D").astype(np.int64))
        beta, betap = bt.ratings_on(day, r.season)
        h, a = bt.ix.get(r.home), bt.ix.get(r.away)
        if h is None or a is None:
            continue
        hcf = 0.0 if r.neutral_site else 1.0
        row = {"game_id": r.game_id}
        for j, s in enumerate(STATS):
            row[f"h_off_{s}"], row[f"h_def_{s}"] = beta[j, h], beta[j, T + h]
            row[f"a_off_{s}"], row[f"a_def_{s}"] = beta[j, a], beta[j, T + a]
            row[f"lg_{s}"], row[f"hca_{s}"] = lg["L"][s], lg["HC"][s] * hcf
        row["h_off_pace"], row["a_off_pace"], row["lg_pace"] = betap[h], betap[a], lg["L"]["pace"]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["game_id"] + FEATURES)
    r = pd.DataFrame(rows)
    f = comp(r)
    f.insert(0, "game_id", r.game_id.values)
    vmap = {row.game_id: vn.value(row.home, bool(row.neutral_site)) for row in games.itertuples(index=False)}
    f["venue"] = f.game_id.map(vmap)
    return f
