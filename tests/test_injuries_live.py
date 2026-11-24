"""Offline tests for injuries_live.py (no network: sample PDFs/JSON from GitHub archives in fixtures/).

    python tests/test_injuries_live.py        (or: pytest tests/)

Fixtures:
  Injury-Report_2026-11-22_05_30PM.pdf   15-minute era official report (Seed808/theesness-data), 9 pages
  Injury-Report_2023-01-10_05PM.pdf      hourly era official report = the 5:30 PM report (akng8/nba-injury-scraper)
  espn_injuries_20261123T2318Z.json      ESPN injuries JSON, trimmed to 4 teams (Seed808/theesness-data)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nbadash"))
import injuries_live as L  # noqa: E402

FX = HERE / "fixtures"
NEW = FX / "Injury-Report_2026-11-22_05_30PM.pdf"
OLD = FX / "Injury-Report_2023-01-10_05PM.pdf"
ESPN = FX / "espn_injuries_20261123T2318Z.json"


def test_parse_new_format():
    d = L.parse_pdf(NEW)
    assert d.report_time.iloc[0] == pd.Timestamp("2026-11-22 17:30")
    assert len(d) == 148 and d.page.max() == 9
    assert set(d.status) <= set(L.STATUSES) | {L.NYS}
    assert d.team.notna().all() and set(d.team) <= set(L.TEAMS)
    # reasons wrapped above and below the row are re-attached
    r = d[d.player_name == "Holland II, Ronald"].iloc[0]
    assert r.status == "Available" and r.reason.endswith("Contusion / Bone Bruise"), r.reason
    # a row continuing on the next page keeps the previous page's game and team
    w = d[d.player_name == "Wilson, Jalen"].iloc[0]
    assert w.page == 2 and w.team == "Atlanta Hawks" and w.matchup == "DET@ATL"
    # the report also lists tomorrow's games, with NOT YET SUBMITTED team rows
    assert set(d.game_date.astype(str)) == {"2026-11-22", "2026-11-23"}
    nys = d[d.status == L.NYS]
    assert len(nys) == 9 and (nys.player_name == "").all()
    assert ("OKC@IND", "Oklahoma City Thunder") in set(zip(nys.matchup, nys.team))


def test_parse_old_format():
    d = L.parse_pdf(OLD)
    assert d.report_time.iloc[0] == pd.Timestamp("2023-01-10 17:30")      # file '05PM' = the 5:30 PM report
    assert len(d) == 128 and d.status.value_counts()["Out"] == 87
    e = d[d.player_name == "Embiid, Joel"].iloc[0]
    assert (e.team, e.status, e.matchup, e.game_time) == ("Philadelphia 76ers", "Questionable", "DET@PHI", "07:00 (ET)")
    t = d[d.player_name == "Tucker, P.J."].iloc[0]
    assert t.status == "Probable"


def test_names_and_teams():
    assert L.display_name("Bagley III, Marvin") == "Marvin Bagley III"
    assert L.norm_name("Jones Jr., Derrick") == L.norm_name("Derrick Jones Jr.") == "derrick jones"
    assert L.norm_name("Tucker, P.J.") == L.norm_name("P.J. Tucker") == "pj tucker"
    assert L.norm_name("Dončić, Luka") == "luka doncic"
    assert L.canon_team("Los Angeles Clippers") == "LA Clippers"
    assert L.canon_team("Portland Trail Blazers") == "Portland Trail Blazers"


def test_id_mapper():
    ros = pd.DataFrame({"athlete_id": [1, 2, 3, 4, 5],
                        "name": ["Derrick Jones Jr.", "Nicolas Claxton", "P.J. Tucker", "Jalen Williams", "Jaylin Williams"],
                        "team": ["Chicago Bulls", "Brooklyn Nets", "Philadelphia 76ers", "Oklahoma City Thunder",
                                 "Oklahoma City Thunder"]})
    m = L.IdMapper(ros)
    assert m.match("Jones Jr., Derrick", "Chicago Bulls") == (1, "team_exact")
    assert m.match("Claxton, Nic", "Brooklyn Nets") == (2, "team_alias")
    assert m.match("Tucker, P.J.", None) == (3, "league_exact")
    assert m.match("Williams, Jaylin", "Oklahoma City Thunder")[0] == 5
    assert m.match("Nobody, Some", "Chicago Bulls") == (None, "miss")


def test_slots():
    assert L.slot_name(dt.datetime(2024, 1, 15, 17, 30)) == "05PM"
    assert L.slot_name(dt.datetime(2026, 1, 15, 17, 30)) == "05_30PM"
    assert L.slot_name(dt.datetime(2026, 1, 15, 0, 15)) == "12_15AM"
    s = L.slots_back(dt.date(2026, 1, 15), dt.datetime(2026, 1, 15, 17, 50))
    assert s[0] == dt.datetime(2026, 1, 15, 17, 45) and s[1] == dt.datetime(2026, 1, 15, 17, 30)
    s = L.slots_back(dt.date(2024, 1, 15), dt.datetime(2024, 1, 15, 17, 50))
    assert s[0] == dt.datetime(2024, 1, 15, 17, 30) and s[1] == dt.datetime(2024, 1, 15, 16, 30)
    s = L.slots_back(dt.date(2024, 1, 15), dt.datetime(2024, 1, 15, 17, 10))
    assert s[0] == dt.datetime(2024, 1, 15, 16, 30)


def test_parse_espn():
    d = L.parse_espn(json.loads(ESPN.read_text()))
    assert len(d) == 11 and d.espn_athlete_id.notna().all()
    r = d[d.player_name == "Jock Landale"].iloc[0]
    assert r.espn_athlete_id == 3146557 and r.status == "Out" and r.team == "Atlanta Hawks"
    assert set(d.status) <= {"Out", "Questionable", "Doubtful", "Probable", "Available"}


class _Resp:
    def __init__(self, code, content=b"", js=None):
        self.status_code, self.content, self._js = code, content, js

    def json(self):
        return self._js


class _Session:
    """Serves the fixture PDF at its official URL and the ESPN JSON; everything else answers 403 like S3."""

    def __init__(self, pdf_ok=True, espn_ok=True):
        self.pdf_ok, self.espn_ok, self.urls = pdf_ok, espn_ok, []

    def get(self, url, timeout=None, headers=None):
        self.urls.append(url)
        if self.pdf_ok and url.endswith("Injury-Report_2026-11-22_05_30PM.pdf"):
            return _Resp(200, NEW.read_bytes())
        if self.espn_ok and url == L.ESPN_URL:
            return _Resp(200, js=json.loads(ESPN.read_text()))
        if self.espn_ok is None:
            raise ConnectionError("offline")
        return _Resp(403, b"<Error><Code>AccessDenied</Code></Error>")


def _with_session(sess, fn):
    import requests
    orig = requests.Session
    requests.Session = lambda: sess
    try:
        return fn()
    finally:
        requests.Session = orig


def test_fetch_official_latest_slot():
    sess = _Session()
    ros = pd.DataFrame({"athlete_id": [99], "name": ["Cade Cunningham"], "team": ["Detroit Pistons"]})
    df = _with_session(sess, lambda: L.fetch_injuries("2026-11-22", until=dt.datetime(2026, 11, 22, 18, 5),
                                                      roster=ros))
    # probed 18:00 and 17:45 (403) before finding 17:30
    assert sess.urls[0].endswith("2026-11-22_06_00PM.pdf") and sess.urls[2].endswith("_05_30PM.pdf")
    assert list(df.columns[:6]) == L.COLUMNS and (df.source == "nba_official_pdf").all()
    assert (pd.to_datetime(df.game_date) == "2026-11-22").all() and len(df) == 83
    c = df[df.player_name == "Cade Cunningham"].iloc[0]
    assert c.espn_athlete_id == 99 and c.status == "Available"


def test_fallback_to_espn_and_empty():
    df = _with_session(_Session(pdf_ok=False), lambda: L.fetch_injuries(
        "2026-11-22", until=dt.datetime(2026, 11, 22, 9, 5), roster=pd.DataFrame(columns=["athlete_id", "name", "team"])))
    assert (df.source == "espn_json").all() and len(df) == 11
    df = _with_session(_Session(pdf_ok=False, espn_ok=None), lambda: L.fetch_injuries(
        "2026-11-22", until=dt.datetime(2026, 11, 22, 7, 5), roster=None))
    assert df.empty and list(df.columns) == L.COLUMNS


def test_availability():
    inj = pd.DataFrame({"team": ["A", "A", "B"], "player_name": ["x", "y", ""], "espn_athlete_id": [1, 2, None],
                        "status": ["Out", "Questionable", L.NYS]})
    inj["espn_athlete_id"] = pd.array(inj.espn_athlete_id, dtype="Int64")
    av = L.availability(inj, {"Out": 0.01, "Questionable": 0.6})
    assert av == {"A": {1: 0.01, 2: 0.6}}


def test_report_availability():
    probs = {"p": {"Out": 0.01, "Questionable": 0.6, "Available": 0.97},
             "p_not_listed_by_missed": {"0": 0.98, "1": 0.5, "2": 0.4, "3-4": 0.3, "5-9": 0.2, "10+": 0.05}}
    la = {1: 0, 2: 0, 3: 4, 4: 1, 5: 12}
    role = {1: 30.0, 2: 25.0, 3: 20.0, 4: 5.0, 5: 18.0}
    av = L.report_availability({1: "Out", 2: "Questionable", 6: "Available"}, la, role, probs)
    assert av == {1: 0.01, 2: 0.6, 3: 0.3, 4: 0.0, 5: 0.05, 6: 0.97}


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k)
    print(f"{n} tests passed")
