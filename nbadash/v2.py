"""Model v2: production ratings + component ratings/venue (ratings2) + player availability (avail).

Two ridge models share everything except the availability features:
  * MORNING  uses morning-of availability (who missed the team's last game + return rates). Always available.
  * REPORT   uses availability from today's injury report (trained on who actually played).
             Used for a game when today's report covers both teams (neither 'NOT YET SUBMITTED').
Walk-forward lab results (MAE, points; closing line 14.06 / 14.51):
                       2022-23..24-25   2025-26
  production v1            14.49         14.85
  v2 morning               14.37         14.74
  v2 with actual lineups   14.30         14.70   (upper bound for the injury-report version)
  v2 + official report     14.32         14.76   (archived reports, latest before tip; status_probs.json)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import avail as AV
from . import injuries_live as IL
from . import model as M
from . import ratings2 as R2

log = logging.getLogger(__name__)
F_MORNING = M.FEATURES + R2.FEATURES + AV.MOR_FEATURES
F_REPORT = M.FEATURES + R2.FEATURES + AV.TIP_FEATURES

# Walk-forward estimates (official reports 2021-22..2025-26): P(plays | status) and, for rotation players the
# report doesn't list, P(plays | consecutive games missed). Used when status_probs.json is missing.
DEFAULT_STATUS_PROBS = {
    "p": {"Out": 0.0008, "Doubtful": 0.026, "Questionable": 0.542, "Probable": 0.925, "Available": 0.885},
    "p_not_listed_by_missed": {"0": 0.946, "1": 0.549, "2": 0.377, "3-4": 0.239, "5-9": 0.095, "10+": 0.012},
}


def _fit(d: pd.DataFrame, feats: list[str]):
    tr = d.dropna(subset=feats + ["total", "struct_total"])
    m = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    m.fit(tr[feats], tr.total - tr.struct_total)
    return m


class ModelV2:
    def __init__(self, data_dir: Path, season: int):
        self.data_dir, self.season = data_dir, season

    def build(self, games: pd.DataFrame, rated: pd.DataFrame) -> pd.DataFrame:
        """rated: production-rated history (build_ratings + add_rest). Returns rated + v2 features."""
        f2, self.rstate = R2.build(M.add_observed(games))
        pdir = self.data_dir / "players"
        AV.download(pdir, M.FIRST_SEASON, self.season)
        players = AV.load_players(pdir, rated, M.FIRST_SEASON, self.season)
        self.av = AV.Avail()
        fa = self.av.run(rated, players)
        d = rated.merge(f2, on="game_id", how="left").merge(fa, on="game_id", how="left")
        train = d[(d.season > M.FIRST_SEASON) & (d.season < self.season)]
        self.m_morning = _fit(train, F_MORNING)
        self.m_report = _fit(train, F_REPORT)
        d["proj_v2"] = d.struct_total + self.m_morning.predict(d[F_MORNING].fillna(0.0))
        return d

    def project(self, slate_df: pd.DataFrame, injuries: pd.DataFrame | None,
                status_p: dict | None = None) -> pd.DataFrame:
        """slate_df: upcoming_frame() output. Adds proj_morning, proj_report, proj, model_used, absences."""
        if slate_df.empty:
            return slate_df
        sp = status_p if status_p and "p" in status_p and "p_not_listed_by_missed" in status_p \
            else DEFAULT_STATUS_PROBS
        u = slate_df.copy()
        r2 = R2.upcoming(self.rstate, u[["game_id", "game_date", "season", "home", "away", "neutral_site"]])
        u = u.merge(r2, on="game_id", how="left")
        inj_by_team, dates = {}, None
        feed_ok = injuries is not None and len(injuries) > 0   # an empty team in a live report = nobody listed
        if feed_ok:
            inj = injuries.copy()
            if "game_date" in inj:
                inj["_d"] = inj.game_date.map(_day)
                dates = set(inj._d)
            else:
                inj["_d"] = None
            for (team, d), g in inj.groupby(["team", "_d"], dropna=False):
                inj_by_team[(team, d if dates is not None else None)] = g
        rows = []
        for r in u.itertuples(index=False):
            row = {"game_id": r.game_id}
            info = {}
            for side, team in (("h", r.home), ("a", r.away)):
                fm = self.av.features(team, r.season, None, "mor")
                row.update({f"{side}_{k}": v for k, v in fm.items()})
                d = _day(r.game_date) if dates is not None else None
                ok = feed_ok and (dates is None or d in dates)   # report must cover this game's date
                probs, absences, covered = self._report_probs(team, inj_by_team.get((team, d)), sp, ok)
                info[side] = (covered, absences)
                ft = self.av.features(team, r.season, probs if covered else None, "tip")
                row.update({f"{side}_{k}": v for k, v in ft.items()})
            row["report_covered"] = info["h"][0] and info["a"][0]
            row["absences_h"], row["absences_a"] = info["h"][1], info["a"][1]
            rows.append(row)
        f = pd.DataFrame(rows)
        u = u.merge(f, on="game_id", how="left")
        u["proj_morning"] = u.struct_total + self.m_morning.predict(u[F_MORNING].fillna(0.0))
        u["proj_report"] = u.struct_total + self.m_report.predict(u[F_REPORT].fillna(0.0))
        u["proj"] = np.where(u.report_covered, u.proj_report, u.proj_morning)
        u["model_used"] = np.where(u.report_covered, "v2 + injury report", "v2 morning-of")
        return u

    def _report_probs(self, team: str, inj: pd.DataFrame | None, sp: dict, feed_ok: bool = True):
        """athlete_id -> p for the team's pool from today's report; (probs, notable absences, covered).

        Listed players get P(plays | status). Rotation players the report leaves off get P(plays | games
        missed in a row), so a player who has been out for weeks (or was traded) isn't assumed to play.
        A team marked NOT YET SUBMITTED isn't covered and keeps morning-of availability.
        """
        if not feed_ok:
            return None, [], False
        listed = {}
        for x in (inj.itertuples(index=False) if inj is not None else []):
            st = str(x.status or "").strip()
            if st.upper() == IL.NYS:
                return None, [], False
            st = IL.ESPN_STATUS.get(st.lower(), st.title())
            aid = getattr(x, "espn_athlete_id", None)
            if aid is None or pd.isna(aid):
                aid = self._match_name(team, x.player_name)
            if aid is not None:
                listed[int(aid)] = st
        la = {int(k): v for k, v in self.av.last_avail[team].items()}
        role = {int(k): v for k, v in self.av.role.items() if int(k) in la or int(k) in listed}
        probs = IL.report_availability(listed, la, role, sp, rot=AV.ROT, role_prior=AV.ROLE_PRIOR)
        absences = []
        for pid, st in listed.items():
            r = role.get(pid, AV.ROLE_PRIOR)
            if st in ("Out", "Doubtful", "Questionable") and r >= 12:
                absences.append(dict(name=self.av.names.get(pid) or self.av.names.get(float(pid)),
                                     status=st, role=round(r, 1)))
        absences.sort(key=lambda a: -a["role"])
        return probs, absences, True

    def _match_name(self, team: str, name: str):
        """Official reports list 'Last, First'; match against the team's recent players."""
        if not name:
            return None
        n = name.strip()
        if "," in n:
            last, first = [p.strip() for p in n.split(",", 1)]
            n = f"{first} {last}"
        key = _norm(n)
        pool = set(self.av.share[team]) | set(self.av.last_avail[team])
        for pid in pool:
            if _norm(self.av.names.get(pid) or "") == key:
                return pid
        return None


def _day(x) -> str | None:
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return None


def _norm(s: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)
