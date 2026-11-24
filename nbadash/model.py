"""Game-total model: opponent-adjusted pace/efficiency ratings + ridge correction.

Same method as the standalone nba_totals_model.py (backtested 2016-17 to 2025-26):
  * possessions = FGA - OREB + TOV + 0.44*FTA, pace per 48 min (OT-adjusted)
  * team offense/defense/pace ratings, opponent-adjusted, updated after every game
  * structural total = exp. pace x (home exp. ORtg + away exp. ORtg) / 100
  * ridge regression on top (rest, back-to-backs, altitude, recent form, playoffs)
Only information available before tip-off goes into each game's features.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .teams import TEAMS

log = logging.getLogger(__name__)

FIRST_SEASON = 2015  # ESPN season = end year (2015 = 2014-15)
BOX_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/"
           "download/espn_nba_team_boxscores/team_box_{y}.parquet")
SCHED_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/"
             "download/espn_nba_schedules/nba_schedule_{y}.parquet")
NBA_TEAMS = set(TEAMS)
HIGH_ALTITUDE = {"Denver Nuggets", "Utah Jazz"}

RATING_PARAMS = dict(alpha_min=1 / 22, prior_games=10, carry=0.55,
                     alpha_fast=1 / 5, alpha_league=1 / 250)

# From the walk-forward backtest (11,987 games with lines, 2016-17 to 2025-26).
CALIBRATION = dict(
    # Share of a model-vs-CLOSING-line gap that shows up in results (walk-forward, true closing lines).
    # v2 morning-of: +0.05 (2016-25), 0.00 (2025-26) -> no edge vs the close.
    beta_morning=0.0,
    # v2 with the official injury report (backtest on archived reports, true closing lines):
    # +0.15 (2022-23..24-25), +0.06 (2025-26). Small and not stable -> 0.10, i.e. lean mostly on the line.
    beta_report=0.10,
    # vs the OPENING line the morning model's beta is +0.45..+0.68 (lines move toward it by the close);
    # the opening-line forward test tracks that via first_seen / closing-line value, not p_over.
    sigma=18.2,
    model_mae=14.74, line_mae=14.51, games=1251,
)

BOX_KEEP = ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
            "three_point_field_goals_attempted", "free_throws_made", "free_throws_attempted",
            "offensive_rebounds", "defensive_rebounds", "assists", "tov", "team_score"]


def current_season(today: date) -> int:
    return today.year + 1 if today.month >= 8 else today.year


def _download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        return
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def download(data_dir: Path, today: date) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    last = current_season(today)
    for y in range(FIRST_SEASON, last + 1):
        force = y == last
        _download(BOX_URL.format(y=y), data_dir / f"team_box_{y}.parquet", force)
        _download(SCHED_URL.format(y=y), data_dir / f"sched_{y}.parquet", force)


def load(data_dir: Path, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(completed games, one row each; full schedule incl. future games)."""
    box = pd.concat([pd.read_parquet(data_dir / f"team_box_{y}.parquet")
                     for y in range(FIRST_SEASON, season + 1)], ignore_index=True)
    sched = pd.concat([pd.read_parquet(data_dir / f"sched_{y}.parquet")
                       for y in range(FIRST_SEASON, season + 1)], ignore_index=True)
    box = box[box.team_display_name.isin(NBA_TEAMS)
              & box.opponent_team_display_name.isin(NBA_TEAMS)
              & box.season_type.isin([2, 3, 5])].copy()
    box["tov"] = box["total_turnovers"].fillna(box["turnovers"])
    cols = ["game_id", "team_home_away", "team_display_name"] + BOX_KEEP
    box = box[cols]
    home = box[box.team_home_away == "home"].drop(columns="team_home_away")
    away = box[box.team_home_away == "away"].drop(columns="team_home_away")
    home.columns = ["game_id"] + ["h_" + c for c in home.columns[1:]]
    away.columns = ["game_id"] + ["a_" + c for c in away.columns[1:]]
    g = home.merge(away, on="game_id")

    sched = sched[sched.home_display_name.isin(NBA_TEAMS) & sched.away_display_name.isin(NBA_TEAMS)
                  & sched.season_type.isin([2, 3, 5])].copy()
    sched["game_id"] = sched["game_id"].astype("int64")
    sched["tip"] = pd.to_datetime(sched["game_date_time"], utc=True)
    sched["game_date"] = pd.to_datetime(sched["game_date"]).dt.date
    sched = sched.drop_duplicates("game_id")
    g["game_id"] = g["game_id"].astype("int64")
    scols = ["game_id", "season", "season_type", "game_date", "tip", "neutral_site",
             "status_period", "status_type_name", "home_display_name", "away_display_name"]
    g = g.merge(sched[scols], on="game_id", how="inner")
    g = g[(g.status_type_name == "STATUS_FINAL") & (g.h_team_display_name == g.home_display_name)]
    g = g.rename(columns={"home_display_name": "home", "away_display_name": "away"})
    g["periods"] = g["status_period"].fillna(4).clip(lower=4)
    g["total"] = g.h_team_score + g.a_team_score
    g = g[g.total > 120].sort_values("tip").reset_index(drop=True)
    return g, sched


def append_games(games: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra is None or extra.empty:
        return games
    extra = extra[~extra.game_id.isin(games.game_id)]
    if extra.empty:
        return games
    out = pd.concat([games, extra[games.columns.intersection(extra.columns)]], ignore_index=True)
    return out.sort_values("tip").reset_index(drop=True)


# --------------------------------------------------------------------------
# Ratings
# --------------------------------------------------------------------------
def possessions(fga, oreb, tov, fta):
    return fga - oreb + tov + 0.44 * fta


def add_observed(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    hp = possessions(g.h_field_goals_attempted, g.h_offensive_rebounds, g.h_tov, g.h_free_throws_attempted)
    ap = possessions(g.a_field_goals_attempted, g.a_offensive_rebounds, g.a_tov, g.a_free_throws_attempted)
    g["poss"] = (hp + ap) / 2
    g["obs_pace"] = g.poss * 48 / (48 + 5 * (g.periods - 4))
    g["obs_ortg_h"] = 100 * g.h_team_score / g.poss
    g["obs_ortg_a"] = 100 * g.a_team_score / g.poss
    return g


def build_ratings(g: pd.DataFrame, p: dict = RATING_PARAMS):
    g = add_observed(g)
    seed = g.head(100)
    L = dict(ortg=float((seed.obs_ortg_h + seed.obs_ortg_a).mean() / 2),
             pace=float(seed.obs_pace.mean()),
             hca=float((seed.obs_ortg_h - seed.obs_ortg_a).mean()))
    teams: dict[str, dict] = {}

    def team(name, season):
        t = teams.get(name)
        if t is None:
            t = dict(off=0.0, de=0.0, pace=0.0, off_f=0.0, de_f=0.0, pace_f=0.0, n=0, season=season)
            teams[name] = t
        if t["season"] != season:
            for k in ("off", "de", "pace", "off_f", "de_f", "pace_f"):
                t[k] *= p["carry"]
            t["n"], t["season"] = 0, season
        return t

    rows = []
    aL = p["alpha_league"]
    for r in g.itertuples(index=False):
        H, A = team(r.home, r.season), team(r.away, r.season)
        neutral = bool(r.neutral_site)
        hca = 0.0 if neutral else L["hca"]
        rows.append(dict(lg_ortg=L["ortg"], lg_pace=L["pace"], lg_hca=hca,
                         h_off=H["off"], h_def=H["de"], h_pace=H["pace"],
                         a_off=A["off"], a_def=A["de"], a_pace=A["pace"],
                         h_off_f=H["off_f"], h_def_f=H["de_f"], h_pace_f=H["pace_f"],
                         a_off_f=A["off_f"], a_def_f=A["de_f"], a_pace_f=A["pace_f"],
                         h_n=H["n"], a_n=A["n"]))
        oh = r.obs_ortg_h - L["ortg"] - hca / 2
        oa = r.obs_ortg_a - L["ortg"] + hca / 2
        pc = r.obs_pace - L["pace"]
        obs = {("H", "off"): oh - A["de"], ("A", "de"): oh - H["off"],
               ("A", "off"): oa - H["de"], ("H", "de"): oa - A["off"],
               ("H", "pace"): pc - A["pace"], ("A", "pace"): pc - H["pace"]}
        obs_f = {("H", "off"): oh - A["de_f"], ("A", "de"): oh - H["off_f"],
                 ("A", "off"): oa - H["de_f"], ("H", "de"): oa - A["off_f"],
                 ("H", "pace"): pc - A["pace_f"], ("A", "pace"): pc - H["pace_f"]}
        for side, T in (("H", H), ("A", A)):
            a = max(p["alpha_min"], 1.0 / (T["n"] + p["prior_games"]))
            for k in ("off", "de", "pace"):
                T[k] += a * (obs[(side, k)] - T[k])
                T[k + "_f"] += p["alpha_fast"] * (obs_f[(side, k)] - T[k + "_f"])
            T["n"] += 1
        L["ortg"] += aL * ((r.obs_ortg_h + r.obs_ortg_a) / 2 - L["ortg"])
        L["pace"] += aL * (r.obs_pace - L["pace"])
        if not neutral:
            L["hca"] += aL * ((r.obs_ortg_h - r.obs_ortg_a) - L["hca"])
    out = pd.concat([g, pd.DataFrame(rows, index=g.index)], axis=1)
    return add_structural(out), teams, L


def add_structural(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["exp_pace"] = d.lg_pace + d.h_pace + d.a_pace
    d["exp_ortg_h"] = d.lg_ortg + d.h_off + d.a_def + d.lg_hca / 2
    d["exp_ortg_a"] = d.lg_ortg + d.a_off + d.h_def - d.lg_hca / 2
    d["struct_total"] = d.exp_pace * (d.exp_ortg_h + d.exp_ortg_a) / 100
    d["exp_pace_f"] = d.lg_pace + d.h_pace_f + d.a_pace_f
    d["struct_total_fast"] = d.exp_pace_f * (2 * d.lg_ortg + d.h_off_f + d.a_def_f + d.a_off_f + d.h_def_f) / 100
    d["lg_total"] = d.lg_pace * 2 * d.lg_ortg / 100
    return d


def add_rest(d: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    s = sched[["game_id", "game_date", "home_display_name", "away_display_name"]]
    long = pd.concat([
        s.rename(columns={"home_display_name": "team"})[["game_id", "game_date", "team"]],
        s.rename(columns={"away_display_name": "team"})[["game_id", "game_date", "team"]],
    ]).sort_values(["team", "game_date"])
    long["game_date"] = pd.to_datetime(long["game_date"])
    long["prev"] = long.groupby("team")["game_date"].shift(1)
    long["rest"] = (long.game_date - long.prev).dt.days.clip(upper=7).fillna(7)

    def gp(series):
        x = series.values.astype("datetime64[D]")
        lo = np.searchsorted(x, x - np.timedelta64(4, "D"), side="left")
        return pd.Series(np.arange(len(x)) - lo, index=series.index)

    long["g_last4"] = long.groupby("team")["game_date"].transform(gp)
    long = long.drop_duplicates(["game_id", "team"])
    d = d.drop(columns=[c for c in ("h_rest", "a_rest", "h_g4", "a_g4") if c in d.columns])
    d = d.merge(long.rename(columns={"team": "home", "rest": "h_rest", "g_last4": "h_g4"})
                [["game_id", "home", "h_rest", "h_g4"]], on=["game_id", "home"], how="left")
    d = d.merge(long.rename(columns={"team": "away", "rest": "a_rest", "g_last4": "a_g4"})
                [["game_id", "away", "a_rest", "a_g4"]], on=["game_id", "away"], how="left")
    for c, v in (("h_rest", 2), ("a_rest", 2), ("h_g4", 1), ("a_g4", 1)):
        d[c] = d[c].fillna(v)
    d["h_b2b"] = (d.h_rest == 1).astype(int)
    d["a_b2b"] = (d.a_rest == 1).astype(int)
    d["altitude"] = d.home.isin(HIGH_ALTITUDE).astype(int)
    d["playoff"] = (d.season_type == 3).astype(int)
    d["early"] = np.exp(-(d.h_n + d.a_n) / 20)
    return d


FEATURES = ["struct_total", "struct_total_fast", "lg_total", "exp_pace", "exp_ortg_h", "exp_ortg_a",
            "h_off_f", "h_def_f", "a_off_f", "a_def_f", "h_pace_f", "a_pace_f",
            "h_rest", "a_rest", "h_b2b", "a_b2b", "h_g4", "a_g4", "altitude", "playoff", "early"]


def fit(train: pd.DataFrame):
    m = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    m.fit(train[FEATURES], train.total - train.struct_total)
    return m


def predict(m, d: pd.DataFrame) -> np.ndarray:
    return d.struct_total.values + m.predict(d[FEATURES])


def upcoming_frame(sched: pd.DataFrame, teams: dict, L: dict, season: int,
                   start: date, days: int) -> pd.DataFrame:
    up = sched[(sched.game_date >= start) & (sched.game_date < start + timedelta(days=days))
               & (sched.status_type_name != "STATUS_FINAL")].copy()
    rows = []
    for r in up.itertuples(index=False):
        H, A = teams.get(r.home_display_name), teams.get(r.away_display_name)
        if H is None or A is None:
            continue

        def cur(T):
            k = 1.0 if T["season"] == season else RATING_PARAMS["carry"]
            n = T["n"] if T["season"] == season else 0
            return {key: T[key] * k for key in ("off", "de", "pace", "off_f", "de_f", "pace_f")} | {"n": n}

        h, a = cur(H), cur(A)
        hca = 0.0 if r.neutral_site else L["hca"]
        rows.append(dict(game_id=r.game_id, season=season, season_type=r.season_type,
                         game_date=r.game_date, tip=r.tip, home=r.home_display_name,
                         away=r.away_display_name, neutral_site=r.neutral_site,
                         lg_ortg=L["ortg"], lg_pace=L["pace"], lg_hca=hca,
                         h_off=h["off"], h_def=h["de"], h_pace=h["pace"],
                         a_off=a["off"], a_def=a["de"], a_pace=a["pace"],
                         h_off_f=h["off_f"], h_def_f=h["de_f"], h_pace_f=h["pace_f"],
                         a_off_f=a["off_f"], a_def_f=a["de_f"], a_pace_f=a["pace_f"],
                         h_n=h["n"], a_n=a["n"]))
    if not rows:
        return pd.DataFrame()
    return add_rest(add_structural(pd.DataFrame(rows)), sched)


def p_over(proj: float, line: float, beta: float = CALIBRATION["beta_morning"],
           sigma: float = CALIBRATION["sigma"]) -> float:
    """Calibrated chance the game goes over ``line`` given the model's projection."""
    from math import erf, sqrt
    z = beta * (proj - line) / sigma
    return 0.5 * (1 + erf(z / sqrt(2)))
