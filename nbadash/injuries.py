"""Daily injury feed.

`fetch(today)` returns one row per listed player: team (ESPN display name), player_name, espn_athlete_id
(when known), status (lower-case: out / doubtful / questionable / probable / available / day-to-day),
report_time and source. Sources are tried in order and combined; any failure returns an empty frame and
the model falls back to morning-of availability.

  1. ESPN injuries JSON  (site.api.espn.com) - uses ESPN athlete ids, which match the box-score data.
  2. Official NBA injury report PDF (ak-static.cms.nba.com), when the parser is installed
     (nbadash.injuries_official, built from the lab's injury lane).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from .teams import resolve

log = logging.getLogger(__name__)
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
UA = {"User-Agent": "nba-board (github actions; public data)"}
COLS = ["team", "player_name", "espn_athlete_id", "status", "report_time", "source"]


def _status(s: str | None) -> str:
    s = (s or "").strip().lower()
    for k in ("out", "doubtful", "questionable", "probable", "available", "day-to-day", "suspension"):
        if s.startswith(k):
            return k
    return s or "unknown"


def fetch_espn(raw: list | None = None) -> pd.DataFrame:
    r = requests.get(ESPN_URL, headers=UA, timeout=30)
    r.raise_for_status()
    d = r.json()
    if raw is not None:
        raw.append(d)
    rows = []
    for t in d.get("injuries") or []:
        team = resolve(t.get("displayName") or t.get("name") or "")
        for inj in t.get("injuries") or []:
            a = inj.get("athlete") or {}
            aid = a.get("id")
            if aid is None:
                # some payloads nest the id in a link like .../player/_/id/4065648/...
                for lk in a.get("links") or []:
                    href = lk.get("href") or ""
                    if "/id/" in href:
                        aid = href.split("/id/")[1].split("/")[0]
                        break
            rows.append(dict(team=team, player_name=a.get("displayName"),
                             espn_athlete_id=float(aid) if aid not in (None, "") else None,
                             status=_status(inj.get("status") or (inj.get("type") or {}).get("description")),
                             report_time=inj.get("date"), source="espn"))
    return pd.DataFrame(rows, columns=COLS)


def fetch(today: date, raw: list | None = None) -> pd.DataFrame:
    frames = []
    try:
        frames.append(fetch_espn(raw))
        log.info("injuries: ESPN %d rows", len(frames[-1]))
    except Exception as e:
        log.warning("injuries: ESPN failed: %s", e)
    try:
        from . import injuries_official
        o = injuries_official.fetch(today)
        if len(o):
            frames.append(o[COLS])
            log.info("injuries: official report %d rows", len(o))
    except ImportError:
        pass
    except Exception as e:
        log.warning("injuries: official report failed: %s", e)
    if not frames:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["team"])
    # prefer the official report where both list a player; it's the league-mandated source
    df["_rank"] = df.source.map({"official": 0, "espn": 1}).fillna(2)
    df["_key"] = df.espn_athlete_id.astype(str) + "|" + df.player_name.fillna("").str.lower()
    df = df.sort_values("_rank").drop_duplicates(["team", "_key"]).drop(columns=["_rank", "_key"])
    return df.reset_index(drop=True)
