"""Tables for Michael's NBA workbook, refreshed every run and published with the board at /wb/.

The workbook's "Update NBA data" button (NBA_Update.bas) downloads these files and pastes them into:
  players.tsv   -> '2-6'!A1:AE   Basketball-Reference per-game player stats, current season
  ratings.tsv   -> '2-6'!AH2:AV  Basketball-Reference team ratings (row 2 headers, 30 teams)
  pace.tsv      -> '2-6'!BH2:BO  TeamRankings possessions per game (row 2 headers, 30 teams)
  schedule.tsv  -> 'Over Calc'!R2:AC  Basketball-Reference schedule for the current month (ET)
  meta.tsv      key/value lines: when each table was fetched, season, month, row counts, errors

Tab-separated, UTF-8, header row first, blanks as empty fields. A source that fails keeps the last good file
(they're committed to the repo), and meta.tsv records the failure so the button can say so.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]


def season_end_year(d) -> int:
    return d.year + 1 if d.month >= 8 else d.year


def _get(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    return r.content.decode("utf-8", "replace").replace("<!--", "").replace("-->", "")


def _table(html: str, table_id: str | None = None) -> pd.DataFrame:
    kw = {"attrs": {"id": table_id}} if table_id else {}
    return pd.read_html(io.StringIO(html), **kw)[0]


def pace() -> pd.DataFrame:
    t = _table(_get("https://www.teamrankings.com/nba/stat/possessions-per-game"))
    if len(t) != 30 or list(t.columns[:2]) != ["Rank", "Team"] or t.shape[1] != 8:
        raise ValueError(f"unexpected TeamRankings table {t.shape} {list(t.columns)}")
    return t


def ratings(yr: int) -> pd.DataFrame:
    t = _table(_get(f"https://www.basketball-reference.com/leagues/NBA_{yr}_ratings.html"), "ratings")
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [c[-1] for c in t.columns]
    t = t[pd.to_numeric(t["Rk"], errors="coerce").notna()]
    if len(t) != 30 or t.shape[1] != 15 or t.columns[1] != "Team":
        raise ValueError(f"unexpected ratings table {t.shape} {list(t.columns)}")
    return t


def players(yr: int) -> pd.DataFrame:
    t = _table(_get(f"https://www.basketball-reference.com/leagues/NBA_{yr}_per_game.html"), "per_game_stats")
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [c[-1] for c in t.columns]
    t = t[t["Player"].astype(str) != "Player"]          # repeated header rows, if any
    if t.shape[1] != 31 or t.columns[1] != "Player" or len(t) < 100:
        raise ValueError(f"unexpected per-game table {t.shape} {list(t.columns)}")
    return t


def schedule(yr: int, month: str) -> pd.DataFrame:
    t = _table(_get(f"https://www.basketball-reference.com/leagues/NBA_{yr}_games-{month}.html"), "schedule")
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [c[-1] for c in t.columns]
    t = t[t["Date"].astype(str) != "Date"]
    cols = [("" if str(c).startswith("Unnamed") else ("PTS" if str(c).startswith("PTS") else str(c)))
            for c in t.columns]
    t.columns = cols
    if t.shape[1] != 12 or cols[0] != "Date":
        raise ValueError(f"unexpected schedule table {t.shape} {cols}")
    return t


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if v.is_integer():
            return str(int(v))
        return repr(v)
    s = str(v)
    return "" if s.lower() == "nan" else s.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    lines = ["\t".join(_cell(c) for c in df.columns)]
    for row in df.itertuples(index=False):
        lines.append("\t".join(_cell(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(out: Path, now: datetime | None = None) -> dict:
    now = (now or datetime.now(ET)).astimezone(ET)
    yr, month = season_end_year(now), MONTHS[now.month - 1]
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.tsv"
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("\t")
            meta[k] = v
    meta.update(season=f"{yr - 1}-{str(yr)[2:]}", month=f"{month.title()} {now.year}",
                last_run=now.strftime("%Y-%m-%d %I:%M %p ET"))
    jobs = [("pace", pace), ("ratings", lambda: ratings(yr)), ("players", lambda: players(yr)),
            ("schedule", lambda: schedule(yr, month))]
    for i, (name, fn) in enumerate(jobs):
        if i > 1:
            time.sleep(4)   # Basketball-Reference allows ~20 requests a minute; we make 3
        try:
            df = fn()
            write_tsv(df, out / f"{name}.tsv")
            meta[f"{name}_rows"] = str(len(df))
            meta[f"{name}_updated"] = now.strftime("%Y-%m-%d %I:%M %p ET")
            meta[f"{name}_error"] = ""
            log.info("workbook feed: %s %d rows", name, len(df))
        except Exception as e:
            meta[f"{name}_error"] = f"{type(e).__name__}: {e}"[:200].replace("\t", " ").replace("\n", " ")
            log.warning("workbook feed: %s failed: %s (keeping last good file)", name, e)
    meta_path.write_text("".join(f"{k}\t{v}\n" for k, v in meta.items()), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/wb")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(Path(a.out))


if __name__ == "__main__":
    main()
