"""Player-availability features (ported from the lab's lineups lane, numerically identical).

The team ratings are exponentially weighted averages of past games, so they implicitly reflect the mix of
players who were on the floor. For each team-game compare

    share_ew_i   EW of player i's minute share for this team (same alpha schedule as the ratings)
    share_today  role_i * p_i / sum(role * p)   (role = EW minutes when he plays, p = availability today)

and weight the difference by regressed per-minute box values from earlier games:

    d_x = 240 * sum_i (share_today_i - share_ew_i) * v_x,i

Availability p:
  tip      1 if the player actually appeared (history only; live, p comes from the injury report)
  mor      morning-of: previous team game + running return rates (no injury feed needed)

Everything is updated only after a game's features are written (no look-ahead). `Avail.run` walks history
and keeps the final state so `Avail.features(team, avail)` can score tonight's games.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PLAYER_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
              "espn_nba_player_boxscores/player_box_{y}.parquet")
STATS = ["min", "pts", "gs", "usg", "dfn", "pm", "offpm", "offmin",
         "ast", "orb", "drb", "stl", "blk", "fg3a", "fta", "tov"]
COMP = ["ast", "orb", "drb", "stl", "blk", "fg3a", "fta", "tov"]
VK = ["pts", "gs", "usg", "dfn", "onoff"] + COMP
TIP_COMPONENTS = ["pts", "ast", "orb", "drb", "stl", "blk", "fg3a", "fta", "tov"]
TIP_FEATURES = [f"{s}_{k}_tip" for k in TIP_COMPONENTS for s in ("h", "a")]
MOR_FEATURES = [f"{s}_{k}_mor" for k in ("pts", "dfn") for s in ("h", "a")]
ROT = 10.0
ROLE_PRIOR = 12.0
DECAY = 0.985
K_PRIOR = 250.0
K_PM = 1500.0
PRIOR_FRAC = 0.85
KBUCKETS = [0, 1, 2, 3, 5, 10]


def _bucket(k: int) -> int:
    b = 0
    for i, t in enumerate(KBUCKETS):
        if k >= t:
            b = i
    return b


def download(data_dir: Path, first: int, last: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for y in range(first, last + 1):
        p = data_dir / f"player_box_{y}.parquet"
        if p.exists() and y != last:
            continue
        r = requests.get(PLAYER_URL.format(y=y), timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)


def load_players(data_dir: Path, games: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    base = games[["game_id", "home", "away"]]
    cols = ["game_id", "team_display_name", "athlete_id", "athlete_display_name", "minutes", "did_not_play",
            "starter", "points", "field_goals_made", "field_goals_attempted", "free_throws_made",
            "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "steals", "assists", "blocks",
            "fouls", "turnovers", "plus_minus", "three_point_field_goals_attempted", "reason"]
    out = []
    for y in range(first, last + 1):
        x = pd.read_parquet(data_dir / f"player_box_{y}.parquet", columns=cols)
        x["game_id"] = x.game_id.astype("int64")
        out.append(x[x.game_id.isin(base.game_id)])
    x = pd.concat(out, ignore_index=True).merge(base, on="game_id")
    x = x[(x.team_display_name == x.home) | (x.team_display_name == x.away)].copy()
    x["side"] = np.where(x.team_display_name == x.home, "h", "a")
    x["min"] = x.minutes.fillna(0.0)
    x["played"] = x["min"] > 0
    f = lambda c: x[c].fillna(0.0)  # noqa: E731
    x["pts"] = f("points")
    x["gs"] = (f("points") + 0.4 * f("field_goals_made") - 0.7 * f("field_goals_attempted")
               - 0.4 * (f("free_throws_attempted") - f("free_throws_made")) + 0.7 * f("offensive_rebounds")
               + 0.3 * f("defensive_rebounds") + f("steals") + 0.7 * f("assists") + 0.7 * f("blocks")
               - 0.4 * f("fouls") - f("turnovers"))
    x["usg"] = f("field_goals_attempted") + 0.44 * f("free_throws_attempted") + f("turnovers")
    x["dfn"] = f("steals") + f("blocks") + 0.3 * f("defensive_rebounds")
    x["pm"] = pd.to_numeric(x.plus_minus, errors="coerce").fillna(0.0)
    x["ast"], x["orb"], x["drb"] = f("assists"), f("offensive_rebounds"), f("defensive_rebounds")
    x["stl"], x["blk"], x["fg3a"] = f("steals"), f("blocks"), f("three_point_field_goals_attempted")
    x["fta"], x["tov"] = f("free_throws_attempted"), f("turnovers")
    keep = ["game_id", "side", "team_display_name", "athlete_id", "athlete_display_name", "min", "played",
            "starter", "pts", "gs", "usg", "dfn", "pm"] + COMP
    return x[keep].drop_duplicates(["game_id", "athlete_id"])


class Avail:
    def __init__(self):
        self.S = defaultdict(lambda: np.zeros(len(STATS)))
        self.role: dict = {}
        self.nplay = defaultdict(int)
        self.share = defaultdict(dict)
        self.last_avail = defaultdict(dict)
        self.tn: dict = {}
        self.lg = np.zeros(len(STATS))
        self.cnt = np.ones((len(KBUCKETS), 2))
        self.names: dict = {}             # athlete_id -> name (for the dashboard)
        self.team_of: dict = {}           # athlete_id -> last team
        self._v: dict = {}
        self.I = {s: i for i, s in enumerate(STATS)}

    # ------------------------------------------------------------ values / features
    def values(self, pid):
        if pid in self._v:
            return self._v[pid]
        s = self.S.get(pid)
        if s is None:
            s = np.zeros(len(STATS))
        I, lg = self.I, self.lg
        lgm = max(lg[I["min"]], 1e-9)
        v = {}
        for st in ("pts", "gs", "usg", "dfn") + tuple(COMP):
            prior = PRIOR_FRAC * lg[I[st]] / lgm
            v[st] = (s[I[st]] + K_PRIOR * prior) / (s[I["min"]] + K_PRIOR)
        on = s[I["pm"]] / (s[I["min"]] + K_PM)
        off = s[I["offpm"]] / (s[I["offmin"]] + K_PM)
        v["onoff"] = on - off
        self._v[pid] = v
        return v

    def team_feats(self, avail: dict, shares: dict):
        pool = set(shares) | set(avail)
        if not pool:
            return None
        proj = {i: self.role.get(i, ROLE_PRIOR) * avail.get(i, 0.0) for i in pool}
        tot = sum(proj.values())
        if tot <= 0:
            return None
        ssum = sum(shares.values())
        out = {k: 0.0 for k in VK}
        miss = 0.0
        for i in pool:
            st = proj[i] / tot
            se = shares.get(i, 0.0) / ssum if ssum > 0 else 0.0
            d = st - se
            if d != 0:
                v = self.values(i)
                for k in VK:
                    out[k] += 240.0 * d * v[k]
            miss += se * (1.0 - min(1.0, avail.get(i, 0.0)))
        out["miss"] = miss
        return out

    def morning_probs(self, team) -> dict:
        """Morning-of availability: played last game -> 1 (rotation) ; missed k games -> running return rate."""
        la = self.last_avail[team]
        pr = self.cnt[:, 1] / self.cnt.sum(axis=1)
        return {pid: (1.0 if k == 0 else 0.0) if self.role.get(pid, ROLE_PRIOR) < ROT else float(pr[_bucket(k)])
                for pid, k in la.items()}

    def rotation(self, team) -> list[dict]:
        """Current rotation for a team (for matching the injury report and for display)."""
        out = []
        tot = sum(self.share[team].values()) or 1.0
        for pid, sh in sorted(self.share[team].items(), key=lambda kv: -kv[1]):
            out.append(dict(athlete_id=pid, name=self.names.get(pid), share=sh / tot,
                            role=self.role.get(pid, ROLE_PRIOR), missed_last=self.last_avail[team].get(pid, 0)))
        return out

    def features(self, team: str, season: int, avail: dict | None = None, tag: str = "tip") -> dict:
        """Features for an upcoming game. avail: athlete_id -> p. None -> morning-of probabilities."""
        self._v.clear()
        se, n = self.tn.get(team, (season, 0))
        if se != season or n == 0 or (avail is None and not self.last_avail[team]):
            return {f"{k}_{tag}": 0.0 for k in VK + ["miss"]}
        a = self.morning_probs(team) if avail is None else avail
        f = self.team_feats(a, self.share[team]) or {}
        return {f"{k}_{tag}": f.get(k, 0.0) for k in VK + ["miss"]}

    # ------------------------------------------------------------ history walk
    def run(self, base: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
        base = base.sort_values("tip").reset_index(drop=True)
        groups = {k: v for k, v in players.groupby(["game_id", "side"], sort=False)}
        for pid, nm in zip(players.athlete_id.values, players.athlete_display_name.values):
            self.names[pid] = nm
        rows = []
        for r in base.itertuples(index=False):
            row = {"game_id": r.game_id}
            self._v.clear()
            gl = {}
            for side, team in (("h", r.home), ("a", r.away)):
                g = groups.get((r.game_id, side))
                gl[side] = g
                se, n = self.tn.get(team, (r.season, 0))
                if se != r.season:
                    n = 0
                first = n == 0
                ft = None
                if g is not None and len(g):
                    av = {}
                    for pid, pld in zip(g.athlete_id.values, g.played.values):
                        if pld or self.role.get(pid, ROLE_PRIOR) < ROT:
                            av[pid] = 1.0
                    ft = self.team_feats(av, self.share[team])
                la = self.last_avail[team]
                fm = None if (first or not la) else self.team_feats(self.morning_probs(team), self.share[team])
                for tag, f in (("tip", ft), ("mor", fm)):
                    for k in VK + ["miss"]:
                        row[f"{side}_{k}_{tag}"] = (f or {}).get(k, 0.0)
            rows.append(row)
            self._update(r, gl)
        return pd.DataFrame(rows)

    def _update(self, r, gl) -> None:
        T = 48.0 + 5.0 * (r.periods - 4)
        for side, team in (("h", r.home), ("a", r.away)):
            g = gl[side]
            se, n = self.tn.get(team, (r.season, 0))
            if se != r.season:
                n = 0
            if g is None or not len(g):
                self.tn[team] = (r.season, n + 1)
                continue
            margin = (r.h_team_score - r.a_team_score) * (1 if side == "h" else -1)
            mins = g["min"].values
            totm = mins.sum()
            pids = g.athlete_id.values
            played = g.played.values
            la = self.last_avail[team]
            pset = set(pids[played])
            for pid, k in la.items():
                if self.role.get(pid, ROLE_PRIOR) >= ROT:
                    self.cnt[_bucket(k), int(pid in pset)] += 1
            listed = set(pids)
            newla = {}
            for pid in set(la) | listed:
                if pid in pset or (pid in listed and self.role.get(pid, ROLE_PRIOR) < ROT):
                    newla[pid] = 0
                else:
                    newla[pid] = la.get(pid, 0) + 1
            self.last_avail[team] = {pid: k for pid, k in newla.items() if k < 40}
            a = max(1 / 22, 1.0 / (n + 10))
            D = self.share[team]
            al = a if D else 1.0
            for pid in list(D):
                D[pid] *= (1 - al)
            for pid, m in zip(pids, mins):
                if m > 0:
                    D[pid] = D.get(pid, 0.0) + al * m / totm
            for pid in [q for q, v in D.items() if v < 1e-4]:
                del D[pid]
            for row_ in g.itertuples(index=False):
                if not row_.played:
                    continue
                pid = row_.athlete_id
                self.team_of[pid] = team
                s = self.S[pid]
                s *= DECAY
                m = row_.min
                vec = np.array([m, row_.pts, row_.gs, row_.usg, row_.dfn, row_.pm,
                                margin - row_.pm, max(T - m, 0.0)] + [getattr(row_, c) for c in COMP])
                s += vec
                self.lg[:] = self.lg * 0.9995 + vec
                self.nplay[pid] += 1
                ar = max(0.15, 1.0 / (self.nplay[pid] + 1))
                base_r = self.role.get(pid, ROLE_PRIOR)
                self.role[pid] = base_r * (1 - ar) + ar * m
            self.tn[team] = (r.season, n + 1)
