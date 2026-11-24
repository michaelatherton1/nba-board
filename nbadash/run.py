"""Daily job: refresh data, project today's games, pull lines, find arbitrage, write the site data.

    python -m nbadash.run                      # full run (what GitHub Actions does)
    python -m nbadash.run --no-venues          # model + stats only (no market calls)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import espn, lines, model, venues
from .teams import ABBR, NICK

log = logging.getLogger("nbadash")
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")


def _r(x, n=1):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def clean(o):
    """JSON-safe copy: numpy scalars -> Python, NaN/inf -> None, dates -> ISO strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (date, datetime, pd.Timestamp)):
        return o.isoformat()
    return o


def tip_label(ts: pd.Timestamp) -> str:
    t = ts.tz_convert(CT)
    return t.strftime("%a %b %-d, %-I:%M %p CT")


def team_line(row, side: str) -> dict:
    """Box-score rates for one team in one game."""
    p = side
    poss = row["poss"]
    fga = row[f"{p}_field_goals_attempted"]
    return dict(
        team=row["home" if p == "h" else "away"], abbr=ABBR[row["home" if p == "h" else "away"]],
        pts=_r(row[f"{p}_team_score"], 0), ortg=_r(100 * row[f"{p}_team_score"] / poss),
        efg=_r(100 * (row[f"{p}_field_goals_made"] + 0.5 * row[f"{p}_three_point_field_goals_made"]) / fga),
        tov_pct=_r(100 * row[f"{p}_tov"] / poss),
        three_rate=_r(100 * row[f"{p}_three_point_field_goals_attempted"] / fga),
        three_pct=_r(100 * row[f"{p}_three_point_field_goals_made"] / max(row[f"{p}_three_point_field_goals_attempted"], 1)),
        ft_rate=_r(100 * row[f"{p}_free_throws_attempted"] / fga),
        oreb=_r(row[f"{p}_offensive_rebounds"], 0), ast=_r(row[f"{p}_assists"], 0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--history", default="data/history")
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--debug-dir", default="debug")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD in US Eastern (default: now)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-venues", action="store_true")
    ap.add_argument("--no-espn", action="store_true")
    ap.add_argument("--no-injuries", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    now = datetime.now(timezone.utc)
    today = date.fromisoformat(args.today) if args.today else now.astimezone(ET).date()
    season = model.current_season(today)
    data_dir, hist_dir, out_dir, dbg = (Path(p) for p in (args.data_dir, args.history, args.out, args.debug_dir))
    for d in (hist_dir, out_dir, dbg):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    if not args.no_download:
        model.download(data_dir, today)
    games, sched = model.load(data_dir, season)
    if not args.no_espn:
        espn_raw: list = []
        extra = espn.finals_since(3, today, set(games.game_id), espn_raw)
        games = model.append_games(games, extra)
        if espn_raw:
            (dbg / "espn_summary_sample.json").write_text(json.dumps(espn_raw, default=str)[:400000])
    log.info("games through %s (%d)", games.game_date.max(), len(games))

    # ---------------- model ----------------
    rated, teams_state, L = model.build_ratings(games)
    rated = model.add_rest(rated, sched)
    train = rated[(rated.season > model.FIRST_SEASON) & (rated.season < season)]
    m = model.fit(train)
    rated["proj"] = model.predict(m, rated)

    # model v2 (component ratings + arena effect + player availability); falls back to v1 on any failure
    from . import injuries, v2
    mv2, v2_ok = v2.ModelV2(data_dir, season), False
    try:
        dv2 = mv2.build(games, rated)
        rated = rated.merge(dv2[["game_id", "proj_v2"]], on="game_id", how="left")
        rated["proj"] = rated["proj_v2"].fillna(rated["proj"])
        v2_ok = True
    except Exception as e:  # keep the board running on v1
        log.exception("model v2 failed, using v1: %s", e)

    inj = pd.DataFrame(columns=injuries.COLS)
    if not args.no_injuries:
        inj_raw: list = []
        roster = None
        if v2_ok:
            roster = pd.DataFrame([dict(athlete_id=int(k), name=v, team=mv2.av.team_of.get(k))
                                   for k, v in mv2.av.names.items() if v and mv2.av.team_of.get(k)])
        inj = injuries.fetch(today, inj_raw, roster)
        if len(inj):
            log.info("injuries: using %s (%d rows, %d teams)", inj.source.iloc[0], len(inj), inj.team.nunique())
        if inj_raw:
            (dbg / "injuries_raw.json").write_text(json.dumps(inj_raw, default=str)[:400000])
    sp_path = Path(__file__).with_name("status_probs.json")
    status_p = json.loads(sp_path.read_text()) if sp_path.exists() else None

    slate_df = model.upcoming_frame(sched, teams_state, L, season, today, 2)
    if not slate_df.empty:
        slate_df["proj_v1"] = model.predict(m, slate_df)
        slate_df["proj"] = slate_df["proj_v1"]
        slate_df["model_used"] = "v1"
        if v2_ok:
            try:
                slate_df = mv2.project(slate_df, inj, status_p)
            except Exception as e:
                log.exception("model v2 projection failed, using v1: %s", e)
        slate_df = slate_df.sort_values("tip")
    slate = []
    for r in slate_df.itertuples(index=False):
        slate.append(dict(
            game_id=int(r.game_id), date=str(r.game_date), tip_ts=r.tip.timestamp(), tip_et=tip_label(r.tip),
            started=r.tip.to_pydatetime() <= now,
            home=r.home, away=r.away, home_short=NICK[r.home], away_short=NICK[r.away],
            home_abbr=ABBR[r.home], away_abbr=ABBR[r.away],
            proj=_r(r.proj), struct_total=_r(r.struct_total), exp_pace=_r(r.exp_pace),
            exp_ortg_h=_r(r.exp_ortg_h), exp_ortg_a=_r(r.exp_ortg_a),
            h_rest=int(r.h_rest), a_rest=int(r.a_rest), h_b2b=bool(r.h_b2b), a_b2b=bool(r.a_b2b),
            proj_v1=_r(getattr(r, "proj_v1", None)), proj_morning=_r(getattr(r, "proj_morning", None)),
            proj_report=_r(getattr(r, "proj_report", None)), model_used=getattr(r, "model_used", "v1"),
            absences_h=getattr(r, "absences_h", None) or [], absences_a=getattr(r, "absences_a", None) or []))
    games_by_id = {g["game_id"]: g for g in slate}
    for g in slate:
        log.info("slate %s %s@%s v1=%s v2=%s used=%s out/q=%s", g["date"], g["away_abbr"], g["home_abbr"], g["proj_v1"],
                 g["proj"], g["model_used"], [a["name"] + ":" + a["status"] for a in g["absences_a"] + g["absences_h"]][:6])

    # ---------------- venues ----------------
    venue_status, quotes = {}, []
    if not args.no_venues and slate:
        results = [venues.fetch_kalshi(), venues.fetch_pmus(), venues.fetch_novig()]
        venues.dump_raw(results, dbg / "venues_raw.json")
        open_slate = [g for g in slate if not g["started"]]
        slate_pairs = {frozenset((g["home"], g["away"])) for g in open_slate}
        for res in results:
            assigned = venues.assign(res.quotes, open_slate)
            quotes.extend(assigned)
            # diagnostics for the log: unmatched quotes by date and whether the matchup is on the slate
            um = [q for q in res.quotes if q.game_id is None]
            summ = {}
            for q in um:
                d = datetime.fromtimestamp(q.start, timezone.utc).astimezone(ET).date().isoformat() if q.start else "no-start"
                key = (d, "pair-on-slate" if frozenset(q.teams) in slate_pairs else "pair-not-on-slate")
                summ[key] = summ.get(key, 0) + 1
            log.info("%s unmatched quotes by (ET date, pair): %s", res.name, dict(sorted(summ.items())[:12]))
            seen = set()
            for q in um:
                key = (q.teams, q.start)
                if key in seen or len(seen) >= 6:
                    continue
                seen.add(key)
                d = datetime.fromtimestamp(q.start, timezone.utc).isoformat() if q.start else None
                log.info("  unmatched %s: teams=%s start=%s market=%s id=%s", res.name, q.teams, d, q.market, q.market_id)
            if res.name == "Kalshi":
                log.info("  slate: %s", [(g["date"], g["away_abbr"], g["home_abbr"], g["started"]) for g in slate])
            if res.name == "Polymarket US":
                log.info("  PMUS marketCounts sample: %s", [ev.get("marketCounts") for ev in res.raw_samples[:2]])
            if res.name == "Polymarket US":
                for ev in res.raw_samples[2:4] or res.raw_samples[:2]:
                    ms = ev.get("markets") or []
                    log.info("PMUS sample event: title=%r start=%r keys=%s nmarkets=%d", ev.get("title"),
                             ev.get("startTime") or ev.get("eventDate"), sorted(ev.keys())[:25], len(ms))
                    for m_ in ms[:4]:
                        log.info("  market: type=%r smt=%r line=%r q=%r sides=%s bid=%r ask=%r", m_.get("marketType"),
                                 m_.get("sportsMarketType"), m_.get("line"), (m_.get("question") or "")[:60],
                                 [(s.get("long"), s.get("description"), (s.get("team") or {}).get("name"))
                                  for s in m_.get("marketSides") or []], m_.get("bestBidQuote"), m_.get("bestAskQuote"))
            venue_status[res.name] = dict(ok=res.ok, error=res.error, markets_seen=res.markets_seen,
                                          quotes=len(res.quotes), matched=len(assigned), skipped=res.skipped)
            log.info("%s: ok=%s markets=%d quotes=%d matched=%d skipped=%d %s", res.name, res.ok,
                     res.markets_seen, len(res.quotes), len(assigned), res.skipped, res.error or "")
        # depth for Polymarket US legs that could be part of an arb
        cand = lines.arbs(quotes, games_by_id)
        pm_ids = {leg["market_id"] for a in cand for leg in a["legs"] if leg["venue"] == "Polymarket US"}
        if pm_ids:
            venues.pmus_book_sizes([q for q in quotes if q.venue == "Polymarket US" and q.market_id in pm_ids])

    arb_list = lines.arbs(quotes, games_by_id)
    grouped = lines.group(quotes)
    cal = model.CALIBRATION
    for g in slate:
        gl = lines.game_lines(grouped.get(g["game_id"], {}), g)
        g["lines"] = gl
        mt = gl["main_total"]
        g["market_total"] = mt
        if mt is not None and g["proj"] is not None:
            gap = g["proj"] - mt
            beta = cal["beta_report"] if g["model_used"] == "v2 + injury report" else cal["beta_morning"]
            p_o = model.p_over(g["proj"], mt, beta=beta)
            lean = "OVER" if gap > 0 else "UNDER" if gap < 0 else "PASS"
            row = next((t for t in gl["totals"] if t["line"] == mt), None)
            bestq = (row or {}).get("yes" if lean == "OVER" else "no")
            p_lean = p_o if lean == "OVER" else 1 - p_o
            g["model_vs_market"] = dict(
                gap=_r(gap), lean=lean, p_lean=_r(p_lean, 3),
                best_price=bestq, ev=_r(p_lean / bestq["eff"] - 1, 3) if bestq else None,
                big_gap=abs(gap) >= 6)
        else:
            g["model_vs_market"] = None

    # ---------------- history snapshot (pre-game numbers, kept for grading) ----------------
    snap_path = hist_dir / f"{today}.json"
    snap = json.loads(snap_path.read_text()) if snap_path.exists() else {"date": str(today), "games": {}}
    for g in slate:
        if g["date"] != str(today):
            continue
        key = str(g["game_id"])
        prev = snap["games"].get(key)
        if g["started"] and prev:
            continue    # don't overwrite pre-game numbers with in-game ones
        entry = dict(
            away=g["away"], home=g["home"], proj=g["proj"], proj_morning=g["proj_morning"],
            proj_report=g["proj_report"], model_used=g["model_used"], market_total=g["market_total"],
            venue_main_totals=g["lines"]["venue_main_totals"],
            lean=(g["model_vs_market"] or {}).get("lean"), gap=(g["model_vs_market"] or {}).get("gap"),
            absences=[a["name"] for a in g["absences_h"] + g["absences_a"]],
            captured_at=now.isoformat(timespec="minutes"))
        # opening-line forward test: keep the FIRST market line we saw and the morning projection then
        first = (prev or {}).get("first_seen")
        if first is None and g["market_total"] is not None:
            pm = g["proj_morning"] if g["proj_morning"] is not None else g["proj"]
            first = dict(time=now.isoformat(timespec="minutes"), market_total=g["market_total"],
                         venue_main_totals=g["lines"]["venue_main_totals"], proj_morning=pm,
                         lean=("OVER" if pm > g["market_total"] else "UNDER") if pm is not None else None,
                         gap=_r(pm - g["market_total"]) if pm is not None else None)
        entry["first_seen"] = first
        snap["games"][key] = entry
        g["first_seen"] = first
    snap["updated"] = now.isoformat(timespec="minutes")
    snap_path.write_text(json.dumps(snap, indent=1))

    # ---------------- yesterday ----------------
    yday = today - timedelta(days=1)
    yd = rated[rated.game_date == yday]
    ysnap_path = hist_dir / f"{yday}.json"
    ysnap = json.loads(ysnap_path.read_text())["games"] if ysnap_path.exists() else {}
    yesterday = []
    for r in yd.to_dict("records"):
        s = ysnap.get(str(int(r["game_id"])), {})
        proj = s.get("proj", _r(r["proj"]))
        mkt = s.get("market_total")
        total = int(r["total"])
        res_vs_line = None
        if mkt is not None:
            res_vs_line = "OVER" if total > mkt else "UNDER" if total < mkt else "PUSH"
        yesterday.append(dict(
            game_id=int(r["game_id"]), away=r["away"], home=r["home"],
            away_short=NICK[r["away"]], home_short=NICK[r["home"]],
            a_score=int(r["a_team_score"]), h_score=int(r["h_team_score"]), total=total,
            ot=int(r["periods"]) - 4, pace=_r(r["obs_pace"]),
            proj=_r(proj), proj_source="live" if "proj" in s else "retro",
            market_total=mkt, result_vs_line=res_vs_line,
            model_lean=s.get("lean"), model_err=_r(total - proj) if proj is not None else None,
            lean_correct=(s.get("lean") == res_vs_line) if s.get("lean") and res_vs_line in ("OVER", "UNDER") else None,
            teams=[team_line(r, "a"), team_line(r, "h")]))

    # ---------------- track record from history ----------------
    graded = []
    finals = rated.set_index("game_id")
    for f in sorted(hist_dir.glob("*.json")):
        if f.stem >= str(today):
            continue
        for gid, s in json.loads(f.read_text()).get("games", {}).items():
            gid = int(gid)
            if gid not in finals.index or s.get("proj") is None:
                continue
            tot = float(finals.loc[gid, "total"])
            fs = s.get("first_seen") or {}
            graded.append(dict(date=f.stem, total=tot, proj=s["proj"], mkt=s.get("market_total"),
                               lean=s.get("lean"), gap=s.get("gap"),
                               open_mkt=fs.get("market_total"), open_lean=fs.get("lean"), open_gap=fs.get("gap")))
    track = None
    if graded:
        gd = pd.DataFrame(graded)
        wl = gd.dropna(subset=["mkt"])
        wl = wl[wl.total != wl.mkt]
        won = ((wl.total > wl.mkt) == (wl.lean == "OVER"))
        big = wl[wl.gap.abs() >= 6]
        bigwon = ((big.total > big.mkt) == (big.lean == "OVER"))
        track = dict(games=len(gd), since=gd.date.min(),
                     mae_model=_r((gd.total - gd.proj).abs().mean(), 2),
                     mae_market=_r((wl.total - wl.mkt).abs().mean(), 2) if len(wl) else None,
                     leans=dict(w=int(won.sum()), l=int((~won).sum())),
                     big_gap_leans=dict(w=int(bigwon.sum()), l=int((~bigwon).sum())))
        # opening-line forward test: |morning projection - first line we saw| >= 6, graded vs that line,
        # and closing-line value = how far the last pre-tip line moved toward the lean (points)
        op = gd.dropna(subset=["open_mkt", "open_gap"])
        op = op[op.open_gap.abs() >= 6]
        if len(op):
            sgn = np.where(op.open_lean == "OVER", 1.0, -1.0)
            res = np.sign(op.total - op.open_mkt) * sgn
            clv = (op.mkt - op.open_mkt) * sgn
            track["opener_test"] = dict(
                flagged=int(len(op)), w=int((res > 0).sum()), l=int((res < 0).sum()), push=int((res == 0).sum()),
                avg_clv=_r(np.nanmean(clv), 2) if clv.notna().any() else None,
                beat_close=int((clv > 0).sum()), lost_close=int((clv < 0).sum()))

    # ---------------- team ratings ----------------
    tr = []
    for name, t in teams_state.items():
        k = 1.0 if t["season"] == season else model.RATING_PARAMS["carry"]
        tr.append(dict(team=name, abbr=ABBR[name], off=_r(t["off"] * k), de=_r(t["de"] * k),
                       net=_r((t["off"] - t["de"]) * k), pace=_r(t["pace"] * k),
                       off_recent=_r(t["off_f"] * k), de_recent=_r(t["de_f"] * k),
                       gp=t["n"] if t["season"] == season else 0))
    tr.sort(key=lambda x: -x["net"])

    payload = dict(
        generated_at=now.isoformat(timespec="minutes"),
        generated_ct=now.astimezone(CT).strftime("%a %b %-d, %-I:%M %p CT"),
        today=str(today), data_through=str(games.game_date.max()),
        league=dict(pace=_r(L["pace"]), ortg=_r(L["ortg"]), avg_total=_r(L["pace"] * 2 * L["ortg"] / 100)),
        calibration=cal, venues=venue_status, games=slate, arbs=arb_list,
        yesterday=dict(date=str(yday), games=yesterday), track_record=track, team_ratings=tr)
    (out_dir / "latest.json").write_text(json.dumps(clean(payload), allow_nan=False))
    log.info("wrote %s: %d games, %d arbs, %d yesterday", out_dir / "latest.json", len(slate), len(arb_list), len(yesterday))


if __name__ == "__main__":
    main()
