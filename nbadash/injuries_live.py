"""Live NBA injury feed for the totals model.

    fetch_injuries(date) -> DataFrame[team, player_name, espn_athlete_id, status, report_time, source]

Primary source: the league's official injury report PDF
    https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{SLOT}.pdf
    SLOT (US Eastern): 'HHAM|PM' = the HH:30 report until 2025-12-22 09:00 (hourly), then every 15 minutes
    'HH_MMAM|PM' (e.g. 05_30PM = the 5:30 PM report). A slot that was not published answers HTTP 403
    (S3 AccessDenied), so the latest report is found by probing slots backwards from now.
    (Spec: sportsdataverse-py nba_official_injury.py and mxufc29/nbainjuries; checked against the PDFs in
    akng8/nba-injury-scraper and Seed808/theesness-data.)
Fallback: ESPN's injuries JSON https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
    (ESPN athlete ids, coarse statuses Out / Day-To-Day / Questionable / Doubtful / Suspension).
If neither answers, an empty frame is returned and the model falls back to its morning-of availability.

Statuses: Out, Doubtful, Questionable, Probable, Available, plus 'NOT YET SUBMITTED' team rows
(player_name empty) so callers can tell "team has not reported" from "nobody listed".

Command line:  python injuries_live.py [YYYY-MM-DD] [--out file.csv]
"""
from __future__ import annotations

import datetime as dt
import io
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

PDF_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{slot}.pdf"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
QUARTER_FROM = dt.datetime(2025, 12, 22, 9, 0)
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
NYS = "NOT YET SUBMITTED"
COLUMNS = ["team", "player_name", "espn_athlete_id", "status", "report_time", "source"]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) nba-totals-injury-feed/1.0"}

TEAMS = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
         "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
         "Houston Rockets", "Indiana Pacers", "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
         "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
         "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
         "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"]
_TEAM_ALIAS = {"Los Angeles Clippers": "LA Clippers", "L.A. Clippers": "LA Clippers", "LA Lakers": "Los Angeles Lakers",
               "Timberwolves": "Minnesota Timberwolves", "Minnesota": "Minnesota Timberwolves"}


# ----------------------------------------------------------------------------------------------- names
_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(s: str, drop_suffix: bool = True) -> str:
    """'Jones Jr., Derrick' / 'Derrick Jones Jr.' -> 'derrick jones'; accents, dots, hyphens, apostrophes out."""
    s = str(s or "")
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first} {last}"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = s.split()
    if drop_suffix:
        toks = [t for t in toks if t not in _SUFFIX] or toks
    return " ".join(toks)


def display_name(s: str) -> str:
    """'Bagley III, Marvin' -> 'Marvin Bagley III'."""
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    if "," in s:
        last, _, first = s.partition(",")
        return f"{first.strip()} {last.strip()}".strip()
    return s


def canon_team(s: str) -> str | None:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    if s in TEAMS:
        return s
    if s in _TEAM_ALIAS:
        return _TEAM_ALIAS[s]
    k = norm_name(s, drop_suffix=False)
    for t in TEAMS:
        if norm_name(t, False) == k:
            return t
    return None


# ----------------------------------------------------------------------------------------------- PDF parsing
_HEADER = {"Game": None, "Matchup": "matchup", "Team": "team", "Player": "player", "Current": "status",
           "Reason": "reason", "Category": "category", "Previous": "previous"}
_RE_DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_RE_TIME = re.compile(r"^\d{1,2}:\d{2}\s*\(ET\)$")
_RE_MATCH = re.compile(r"^[A-Z]{2,4}@[A-Z]{2,4}$")
_RE_RTIME = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2,4})\s+(\d{1,2}:\d{2})\s*([AP]M)")


def _lines(words: list[dict], tol: float = 2.5) -> list[tuple[float, list[dict]]]:
    out: list[tuple[float, list[dict]]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if out and abs(out[-1][0] - w["top"]) < tol:
            out[-1][1].append(w)
        else:
            out.append((w["top"], [w]))
    return [(t, sorted(ws, key=lambda w: w["x0"])) for t, ws in out]


def _columns(words: list[dict]) -> tuple[float, list[tuple[float, str]]] | None:
    """Column x-starts from the header line ('Game Date Game Time Matchup Team Player Name Current Status Reason')."""
    for top, ws in _lines(words):
        txt = [w["text"] for w in ws]
        if "Matchup" in txt and "Reason" in txt and ("Player" in txt or "PlayerName" in txt):
            cols = []
            games = 0
            for w in ws:
                t = w["text"]
                if t == "Game":
                    cols.append((w["x0"] - 2, "date" if games == 0 else "time"))
                    games += 1
                elif t in _HEADER and _HEADER[t]:
                    cols.append((w["x0"] - 2, _HEADER[t]))
                elif t in ("PlayerName",):
                    cols.append((w["x0"] - 2, "player"))
                elif t in ("CurrentStatus",):
                    cols.append((w["x0"] - 2, "status"))
            cols.sort()
            return top, cols
    return None


def _cells(ws: list[dict], cols: list[tuple[float, str]]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for w in ws:
        name = cols[0][1]
        for x, c in cols:
            if w["x0"] >= x:
                name = c
        out.setdefault(name, []).append(w["text"])
    return {k: " ".join(v) for k, v in out.items()}


def parse_report_time(text: str) -> dt.datetime | None:
    m = _RE_RTIME.search(text or "")
    if not m:
        return None
    d = m.group(1)
    fmt = "%m/%d/%y" if len(d) == 8 else "%m/%d/%Y"
    return dt.datetime.strptime(f"{d} {m.group(2)} {m.group(3)}", f"{fmt} %I:%M %p")


def parse_pdf(pdf: bytes | str | Path) -> pd.DataFrame:
    """Official injury-report PDF -> one row per listed player (and per 'NOT YET SUBMITTED' team).

    Word-position parser (pdfplumber): column bounds from the header line repeated on every page; game date,
    game time, matchup and team carry forward across rows and pages; reason (and the rare player/team name)
    fragments that wrap onto lines above/below a row are attached to the nearest row on the page.
    Columns: report_time, game_date, game_time, matchup, team, player_name (as printed, 'Last, First'),
    status, reason, page.
    """
    import pdfplumber

    src = io.BytesIO(pdf) if isinstance(pdf, (bytes, bytearray)) else pdf
    rows: list[dict] = []
    rtime = None
    carry = {"date": None, "time": None, "matchup": None, "team": None}
    cols = None
    with pdfplumber.open(src) as doc:
        for pno, page in enumerate(doc.pages, 1):
            words = page.extract_words(x_tolerance=1.5, y_tolerance=2)
            if rtime is None:
                rtime = parse_report_time(page.extract_text(x_tolerance=1.5) or "")
            hc = _columns(words)
            if hc is not None:
                hdr_top, cols = hc
            elif cols is None:
                continue
            else:
                hdr_top = 0.0
            page_rows: list[dict] = []
            frags: list[tuple[float, dict]] = []
            for top, ws in _lines(words):
                if top <= hdr_top + 1:
                    continue
                line = " ".join(w["text"] for w in ws)
                if line.startswith("Injury Report:") or re.fullmatch(r"Page \d+ of \d+", line):
                    continue
                c = _cells(ws, cols)
                for key, rx in (("date", _RE_DATE), ("time", _RE_TIME), ("matchup", _RE_MATCH)):
                    v = c.get(key, "").strip()
                    if v and rx.match(v):
                        carry[key] = v
                team_txt = c.get("team", "").strip()
                player = c.get("player", "").strip()
                status = c.get("status", "").strip()
                reason = c.get("reason", "").strip()
                nys = NYS in line.upper()
                if team_txt and canon_team(team_txt):
                    carry["team"] = canon_team(team_txt)
                if nys:
                    r = dict(carry, player_name="", status=NYS, reason=NYS, top=top, team_txt=team_txt)
                    page_rows.append(r)
                elif status in STATUSES:
                    r = dict(carry, player_name=player, status=status, reason=reason, top=top, team_txt=team_txt)
                    page_rows.append(r)
                else:
                    frags.append((top, {"player": player, "reason": reason, "team": team_txt}))
            # attach wrapped fragments to the nearest row on this page
            for top, f in frags:
                if not page_rows:
                    if rows and f["reason"]:
                        rows[-1]["reason"] = (rows[-1]["reason"] + " " + f["reason"]).strip()
                    continue
                r = min(page_rows, key=lambda r: abs(r["top"] - top))
                if abs(r["top"] - top) > 14:
                    continue
                above = top < r["top"]
                for k, dst in (("reason", "reason"), ("player", "player_name"), ("team", "team_txt")):
                    if f[k] and (k != "player" or r["status"] != NYS):
                        r[dst] = (f[k] + " " + r[dst]) if above else (r[dst] + " " + f[k])
                        r[dst] = r[dst].strip()
            for r in page_rows:
                t = canon_team(r.pop("team_txt"))
                if t:
                    r["team"] = t
                r["page"] = pno
                r.pop("top")
                rows.append(r)
    out = pd.DataFrame(rows, columns=["date", "time", "matchup", "team", "player_name", "status", "reason", "page"])
    out = out.rename(columns={"date": "game_date", "time": "game_time"})
    # a team name that wrapped (e.g. a fragment above the row) can only be fixed after the page: carry forward
    out["team"] = out["team"].ffill()
    out["game_date"] = pd.to_datetime(out.game_date, format="%m/%d/%Y", errors="coerce").dt.date
    out.insert(0, "report_time", pd.Timestamp(rtime) if rtime else pd.NaT)
    out["player_name"] = out.player_name.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


# ----------------------------------------------------------------------------------------------- slots
def slot_name(t: dt.datetime) -> str:
    h, ap = f"{t.hour % 12 or 12:02d}", ("AM" if t.hour < 12 else "PM")
    return f"{h}_{t.minute:02d}{ap}" if t >= QUARTER_FROM else f"{h}{ap}"


def slots_back(day: dt.date, until: dt.datetime) -> list[dt.datetime]:
    """Candidate report times for `day`, newest first, not after `until` (US Eastern, naive).
    Hourly era: HH:30 reports. 15-minute era: HH:00/15/30/45. Earliest slot considered 06:00."""
    out = []
    t0 = dt.datetime.combine(day, dt.time(6, 0))
    t = min(until, dt.datetime.combine(day, dt.time(23, 59)))
    if t >= QUARTER_FROM:
        t = t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0)
        while t >= t0 and t >= QUARTER_FROM:
            out.append(t)
            t -= dt.timedelta(minutes=15)
    t = t.replace(second=0, microsecond=0)
    t = t.replace(minute=30) if t.minute >= 30 else (t - dt.timedelta(hours=1)).replace(minute=30)
    while t >= t0:
        if t < QUARTER_FROM:
            out.append(t)
        t -= dt.timedelta(hours=1)
    return out


def now_eastern() -> dt.datetime:
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)


# ----------------------------------------------------------------------------------------------- id mapping
class IdMapper:
    """Map report names to ESPN athlete ids using player-box rosters (athlete_id, athlete_display_name, team).

    Order: exact normalized name on the team -> same with nickname aliases -> unique last name + first initial
    on the team -> unique exact name league-wide (traded / two-way players) -> close fuzzy match on the team.
    """

    _PAIRS = [  # (official report spelling, ESPN spelling), normalized; both directions are tried
        ("nic claxton", "nicolas claxton"), ("cam thomas", "cameron thomas"), ("herb jones", "herbert jones"),
        ("moe wagner", "moritz wagner"), ("ish smith", "ishmael smith"), ("mo bamba", "mohamed bamba"),
        ("bones hyland", "nahshon hyland"), ("cam payne", "cameron payne"), ("kj martin", "kenyon martin"),
        ("gg jackson", "gregory jackson"), ("bub carrington", "carlton carrington"),
        ("nate williams", "jeenathan williams"), ("jae sean tate", "jaesean tate"), ("ron holland", "ronald holland"),
        ("alexandre sarr", "alex sarr"), ("enes kanter", "enes freedom"), ("nene hilario", "nene"),
        ("cui yongxi", "yongxi cui"), ("marcos louzada silva", "didi louzada"), ("david jones garcia", "david jones"),
        ("trey jemison", "trey jemison iii"), ("xavier tillman sr", "xavier tillman"), ("jakob poltl", "jakob poeltl"),
        ("svi mykhailiuk", "sviatoslav mykhailiuk"), ("guillermo hernangomez", "willy hernangomez"),
        ("juan hernangomez", "juancho hernangomez"), ("frank mason", "frank mason iii"),
        ("wes iwundu", "wesley iwundu"), ("lou williams", "louis williams"), ("pj dozier", "pj dozier"),
    ]
    ALIASES: dict[str, list[str]] = {}
    for _a, _b in _PAIRS:
        if _a != _b:
            ALIASES.setdefault(_a, []).append(_b)
            ALIASES.setdefault(_b, []).append(_a)

    def __init__(self, roster: pd.DataFrame):
        """roster: athlete_id, name (display), team, and optionally season."""
        r = roster.dropna(subset=["athlete_id", "name"]).copy()
        r["k"] = r.name.map(norm_name)
        r["last"] = r.k.str.split().str[-1]
        r["fi"] = r.k.str[:1]
        self.r = r
        self.by_team = {t: g.drop_duplicates("athlete_id") for t, g in r.groupby("team")}
        g = r.drop_duplicates(["athlete_id", "k"]).groupby("k").athlete_id.agg(lambda s: list(dict.fromkeys(s)))
        self.glob = g.to_dict()

    def match(self, name: str, team: str | None) -> tuple[int | None, str]:
        k = norm_name(name)
        if not k:
            return None, "empty"
        keys = [k] + self.ALIASES.get(k, [])
        T = self.by_team.get(team) if team else None
        if T is not None:
            for kk in keys:
                hit = T[T.k == kk]
                if len(hit) == 1:
                    return int(hit.athlete_id.iloc[0]), "team_exact" if kk == k else "team_alias"
            toks = k.split()
            if toks:
                hit = T[(T["last"] == toks[-1]) & (T.fi == toks[0][:1])]
                if len(hit) == 1:
                    return int(hit.athlete_id.iloc[0]), "team_last_initial"
        for kk in keys:
            ids = self.glob.get(kk, [])
            if len(ids) == 1:
                return int(ids[0]), "league_exact"
        if T is not None and len(T):
            import difflib
            best = difflib.get_close_matches(k, T.k.tolist(), n=2, cutoff=0.85)
            if len(best) >= 1 and (len(best) == 1 or best[0] != best[1]):
                hit = T[T.k == best[0]]
                if len(hit) == 1:
                    return int(hit.athlete_id.iloc[0]), "team_fuzzy"
        return None, "miss"


def default_roster(seasons: Iterable[int] | None = None, data_dir: str | Path | None = None) -> pd.DataFrame | None:
    """Roster from cached ESPN player boxes (sportsdataverse player_box_{season}.parquet), if present."""
    cands = [Path(data_dir)] if data_dir else [Path("data/raw/players"), Path("data/raw"),
                                               Path("/home/claude/lab/agents/lineups/data")]
    for d in cands:
        files = sorted(d.glob("player_box_*.parquet")) if d.exists() else []
        if seasons is not None:
            files = [f for f in files if int(re.sub(r"\D", "", f.stem) or 0) in set(seasons)]
        if files:
            cols = ["athlete_id", "athlete_display_name", "team_display_name", "game_date"]
            x = pd.concat([pd.read_parquet(f, columns=cols) for f in files[-2:]], ignore_index=True)
            x = x.sort_values("game_date").drop_duplicates(["athlete_id", "team_display_name"], keep="last")
            x["team"] = x.team_display_name.map(lambda s: canon_team(s) or s)
            return x.rename(columns={"athlete_display_name": "name"})[["athlete_id", "name", "team"]]
    return None


# ----------------------------------------------------------------------------------------------- fetchers
def _get(session, url: str, timeout: float = 20.0):
    try:
        return session.get(url, timeout=timeout, headers=UA)
    except Exception:
        return None


def fetch_official(date: dt.date, until: dt.datetime | None = None, session=None, max_probes: int = 40,
                   pause: float = 0.3) -> pd.DataFrame:
    """Latest official report for `date` published at or before `until` (default: now, US Eastern)."""
    import requests
    s = session or requests.Session()
    until = until or now_eastern()
    for i, t in enumerate(slots_back(date, until)):
        if i >= max_probes:
            break
        url = PDF_URL.format(date=date.isoformat(), slot=slot_name(t))
        r = _get(s, url)
        if r is not None and r.status_code == 200 and r.content[:4] == b"%PDF":
            try:
                df = parse_pdf(r.content)
            except Exception:
                continue
            if df.empty:
                continue
            if df.report_time.isna().all():
                df["report_time"] = pd.Timestamp(t)
            df["source"] = "nba_official_pdf"
            df["url"] = url
            return df
        time.sleep(pause)
    return pd.DataFrame()


ESPN_STATUS = {"out": "Out", "doubtful": "Doubtful", "questionable": "Questionable", "day-to-day": "Questionable",
               "probable": "Probable", "available": "Available", "active": "Available", "suspension": "Out",
               "out for season": "Out", "injured reserve": "Out"}


def parse_espn(js: dict) -> pd.DataFrame:
    """ESPN injuries JSON -> rows (ESPN athlete ids straight from the athlete links)."""
    rows = []
    for t in js.get("injuries", []) or []:
        team = canon_team(t.get("displayName", "")) or t.get("displayName")
        for i in t.get("injuries", []) or []:
            a = i.get("athlete", {}) or {}
            aid = None
            for ln in a.get("links", []) or []:
                m = re.search(r"/id/(\d+)", ln.get("href", ""))
                if m:
                    aid = int(m.group(1))
                    break
            if aid is None and str(a.get("id", "")).isdigit():
                aid = int(a["id"])
            raw = str(i.get("status") or (i.get("type") or {}).get("description") or "")
            ts = pd.to_datetime(i.get("date"), utc=True, errors="coerce")
            rows.append(dict(team=team, player_name=a.get("displayName"), espn_athlete_id=aid,
                             status=ESPN_STATUS.get(raw.lower(), raw.title() or None), status_raw=raw,
                             report_time=(ts.tz_convert("America/New_York").tz_localize(None) if pd.notna(ts) else pd.NaT),
                             return_date=((i.get("details") or {}).get("returnDate")), source="espn_json"))
    return pd.DataFrame(rows)


def fetch_espn(session=None) -> pd.DataFrame:
    import requests
    s = session or requests.Session()
    r = _get(s, ESPN_URL)
    if r is None or r.status_code != 200:
        return pd.DataFrame()
    try:
        return parse_espn(r.json())
    except Exception:
        return pd.DataFrame()


def _finish(df: pd.DataFrame, mapper: IdMapper | None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    if "espn_athlete_id" not in df:
        df["espn_athlete_id"] = None
        if mapper is not None:
            ids = [mapper.match(n, t)[0] if s != NYS else None
                   for n, t, s in zip(df.player_name, df.team, df.status)]
            df["espn_athlete_id"] = ids
        df["player_name"] = df.player_name.map(display_name)
    df["espn_athlete_id"] = pd.array(df.espn_athlete_id, dtype="Int64")
    extra = [c for c in ("game_date", "game_time", "matchup", "reason", "url") if c in df]
    return df[COLUMNS + extra].reset_index(drop=True)


def fetch_injuries(date: dt.date | str | None = None, until: dt.datetime | None = None,
                   roster: pd.DataFrame | None = None, use_espn: bool = True) -> pd.DataFrame:
    """Latest pre-game injury statuses for `date` (default today, US Eastern).

    Returns DataFrame[team, player_name, espn_athlete_id, status, report_time, source] (+ game_date, game_time,
    matchup, reason for the official PDF). Empty frame if no feed answers; never raises on network errors.
    """
    date = pd.Timestamp(date or now_eastern().date()).date()
    if roster is None:
        try:
            roster = default_roster()
        except Exception:
            roster = None
    mapper = IdMapper(roster) if roster is not None and len(roster) else None
    try:
        df = fetch_official(date, until)
    except Exception:
        df = pd.DataFrame()
    if not df.empty:
        df = df[df.game_date.isna() | (df.game_date == date)]
        return _finish(df, mapper)
    if use_espn:
        try:
            df = fetch_espn()
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            return _finish(df, mapper)
    return pd.DataFrame(columns=COLUMNS)


def availability(inj: pd.DataFrame, probs: dict[str, float]) -> dict[str, dict[int, float]]:
    """team -> {espn_athlete_id: P(plays)} from a fetch_injuries() frame and status_probs.json['p'].
    Teams with a 'NOT YET SUBMITTED' row are left out (caller uses morning-of for them)."""
    out: dict[str, dict[int, float]] = {}
    if inj is None or inj.empty:
        return out
    nys = set(inj.loc[inj.status == NYS, "team"])
    for r in inj.itertuples(index=False):
        if r.team in nys or pd.isna(r.espn_athlete_id) or r.status not in probs:
            continue
        out.setdefault(r.team, {})[int(r.espn_athlete_id)] = float(probs[r.status])
    return out


def _bucket(k: int) -> int:
    return sum(k >= t for t in (1, 2, 3, 5, 10))


def report_availability(listed: dict[int, str], last_avail: dict[int, int], role: dict[int, float],
                        probs: dict, rot: float = 10.0, role_prior: float = 12.0) -> dict[int, float]:
    """P(plays) for a team's pool given its submitted report (same rule as features_inj / the evaluation).

    listed      espn_athlete_id -> report status for this team (from fetch_injuries; empty dict = nobody listed)
    last_avail  athlete_id -> consecutive team games missed before today (nbadash Avail.last_avail[team])
    role        athlete_id -> EW minutes when playing (nbadash Avail.role)
    probs       status_probs.json: uses probs['p'] and probs['p_not_listed_by_missed']
    Listed -> P(status); rotation player not listed -> P(not listed | games missed); low-role not listed ->
    1 if he was available last game else 0. Pass the result as `avail` to Avail.features(team, season, avail).
    Use only for teams that submitted (no 'NOT YET SUBMITTED' row); otherwise keep morning-of probabilities.
    """
    ps, pn = probs["p"], list(probs["p_not_listed_by_missed"].values())
    out = {}
    for pid in set(last_avail) | set(listed):
        k = last_avail.get(pid, 0)
        if pid in listed and listed[pid] in ps:
            out[pid] = float(ps[listed[pid]])
        elif role.get(pid, role_prior) < rot:
            out[pid] = 1.0 if k == 0 else 0.0
        else:
            out[pid] = float(pn[_bucket(k)])
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0] if args else None
    df = fetch_injuries(d)
    print(f"{len(df)} rows; sources: {df.source.value_counts().to_dict() if len(df) else {}}")
    if "--out" in sys.argv:
        df.to_csv(sys.argv[sys.argv.index("--out") + 1], index=False)
    else:
        print(df.head(40).to_string())
