"""Daily injury feed.

`fetch(today, raw, roster)` returns one row per listed player:
  team (ESPN display name), player_name, espn_athlete_id, status, report_time, source
Statuses use the official report's words: Out / Doubtful / Questionable / Probable / Available, plus a
'NOT YET SUBMITTED' row for a team that hasn't filed yet (that team then uses morning-of availability).

Rows carry game_date; a game uses the report only if the report covers that date (the official report
also lists tomorrow's games once teams file).

Sources, in order (the first that answers is used for the whole slate):
  1. Official NBA injury report PDF (ak-static.cms.nba.com), latest report published before now
     (before the first report of the day, yesterday's last report, which already lists today's games).
     This is the source the status -> P(plays) table in status_probs.json was estimated on (2021-22..2025-26).
  2. ESPN injuries JSON (site.api.espn.com) as a fallback. ESPN 'Day-To-Day' is treated as Questionable.
Any failure returns an empty frame and the model falls back to morning-of availability.
Parsing and name -> ESPN id matching live in injuries_live.py (tested against saved PDFs in tests/fixtures).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta

import pandas as pd

from . import injuries_live as IL
from .teams import resolve

log = logging.getLogger(__name__)
COLS = IL.COLUMNS
NYS = IL.NYS


def fetch(today: date, raw: list | None = None, roster: pd.DataFrame | None = None) -> pd.DataFrame:
    mapper = IL.IdMapper(roster) if roster is not None and len(roster) else None
    df = pd.DataFrame()
    try:
        df = IL.fetch_official(today)
        if df.empty:   # early morning: yesterday's last report already covers today's games
            y = today - timedelta(days=1)
            df = IL.fetch_official(y, until=datetime.combine(y, time(23, 59)))
        if not df.empty:
            df["game_date"] = df.game_date.fillna(today)
            df = df[df.game_date >= today]
            if raw is not None:
                raw.append({"official_url": df.url.iloc[0] if "url" in df and len(df) else None,
                            "rows": len(df)})
            df = IL._finish(df, mapper)
            df["source"] = "official"
            log.info("injuries: official report %d rows (%s), %d teams not yet submitted, ids matched %d/%d",
                     len(df), df.report_time.max(), int((df.status == NYS).sum()),
                     int(df.espn_athlete_id.notna().sum()), int((df.status != NYS).sum()))
    except Exception as e:
        log.warning("injuries: official report failed: %s", e)
        df = pd.DataFrame()
    if df.empty:
        try:
            import requests
            r = requests.get(IL.ESPN_URL, headers=IL.UA, timeout=30)
            r.raise_for_status()
            js = r.json()
            if raw is not None:
                raw.append(js)
            df = IL._finish(IL.parse_espn(js), mapper)
            df["source"] = "espn"
            df["game_date"] = today          # ESPN statuses are current: apply them to today's games only
            log.info("injuries: official report unavailable; ESPN %d rows", len(df))
        except Exception as e:
            log.warning("injuries: ESPN failed: %s", e)
            df = pd.DataFrame(columns=COLS)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df["team"] = df.team.map(lambda t: resolve(t) if isinstance(t, str) else None)
    df = df.dropna(subset=["team"])
    return df.reset_index(drop=True)
