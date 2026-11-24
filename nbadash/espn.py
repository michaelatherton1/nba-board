"""Fills the gap between the daily sportsdataverse release and last night's games.

The release usually has yesterday's box scores by morning, but not always. For any
game in the last few days that ESPN shows as final and the release doesn't have yet,
pull the box score straight from ESPN's public site API.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from .teams import TEAMS

log = logging.getLogger(__name__)
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "nba-dashboard (github actions)"}


def _get(path: str, **params):
    r = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _stat(stats: list[dict], name: str) -> str | None:
    for s in stats:
        if s.get("name") == name:
            return s.get("displayValue")
    return None


def _pair(v: str | None) -> tuple[float, float]:
    try:
        a, b = (v or "").split("-")
        return float(a), float(b)
    except ValueError:
        return float("nan"), float("nan")


def _num(v: str | None) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def finals_since(days_back: int, today: date, known_ids: set[int], raw_dump: list | None = None) -> pd.DataFrame:
    rows = []
    for k in range(1, days_back + 1):
        d = today - timedelta(days=k)
        try:
            sb = _get("scoreboard", dates=d.strftime("%Y%m%d"))
        except Exception as e:  # network trouble shouldn't stop the run
            log.warning("ESPN scoreboard %s failed: %s", d, e)
            continue
        for ev in sb.get("events") or []:
            try:
                gid = int(ev["id"])
            except (KeyError, ValueError):
                continue
            completed = (((ev.get("status") or {}).get("type") or {}).get("completed"))
            if gid in known_ids or not completed:
                continue
            try:
                summ = _get("summary", event=gid)
            except Exception as e:
                log.warning("ESPN summary %s failed: %s", gid, e)
                continue
            if raw_dump is not None and len(raw_dump) < 2:
                raw_dump.append(summ)
            row = _parse_summary(gid, summ)
            if row:
                rows.append(row)
    if rows:
        log.info("ESPN: added %d final games missing from the release", len(rows))
    return pd.DataFrame(rows)


def _parse_summary(gid: int, s: dict) -> dict | None:
    try:
        comp = s["header"]["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        home = teams["home"]["team"]["displayName"]
        away = teams["away"]["team"]["displayName"]
        if home not in TEAMS or away not in TEAMS:
            return None
        box = {t["team"]["displayName"]: t.get("statistics") or [] for t in s["boxscore"]["teams"]}
        row = dict(game_id=gid, home=home, away=away,
                   season=int(s["header"]["season"]["year"]),
                   season_type=int(s["header"]["season"]["type"]),
                   game_date=pd.Timestamp(comp["date"]).tz_convert("America/New_York").date(),
                   tip=pd.Timestamp(comp["date"]).tz_convert("UTC"),
                   neutral_site=bool(comp.get("neutralSite")),
                   periods=max(4, int((comp.get("status") or {}).get("period") or 4)))
        for side, name in (("h", home), ("a", away)):
            st = box[name]
            fgm, fga = _pair(_stat(st, "fieldGoalsMade-fieldGoalsAttempted"))
            tpm, tpa = _pair(_stat(st, "threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            ftm, fta = _pair(_stat(st, "freeThrowsMade-freeThrowsAttempted"))
            tov = _num(_stat(st, "totalTurnovers"))
            if tov != tov:
                tov = _num(_stat(st, "turnovers"))
            row.update({
                f"{side}_team_score": float(teams["home" if side == "h" else "away"]["score"]),
                f"{side}_field_goals_made": fgm, f"{side}_field_goals_attempted": fga,
                f"{side}_three_point_field_goals_made": tpm, f"{side}_three_point_field_goals_attempted": tpa,
                f"{side}_free_throws_made": ftm, f"{side}_free_throws_attempted": fta,
                f"{side}_offensive_rebounds": _num(_stat(st, "offensiveRebounds")),
                f"{side}_defensive_rebounds": _num(_stat(st, "defensiveRebounds")),
                f"{side}_assists": _num(_stat(st, "assists")),
                f"{side}_tov": tov,
            })
        row["total"] = row["h_team_score"] + row["a_team_score"]
        needed = ["h_field_goals_attempted", "h_offensive_rebounds", "h_tov", "h_free_throws_attempted",
                  "a_field_goals_attempted", "a_offensive_rebounds", "a_tov", "a_free_throws_attempted"]
        if any(row[c] != row[c] for c in needed) or row["season_type"] not in (2, 3, 5):
            return None
        return row
    except (KeyError, IndexError, TypeError, ValueError) as e:
        log.warning("ESPN summary %s unparseable: %s", gid, e)
        return None
