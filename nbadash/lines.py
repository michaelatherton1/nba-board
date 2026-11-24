"""Best available prices and fee-adjusted arbitrage across venues."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

from .venues import Quote

MAX_SANE_EDGE = 0.05   # >5c per $1 pair has in practice meant a rules mismatch or stale quote


def fee(q: Quote, pregame: bool = True) -> float:
    """Taker fee per contract at the quoted price."""
    if q.fee_live_only and pregame:
        return 0.0
    return q.fee_coef * q.ask * (1 - q.ask)


def eff(q: Quote) -> float:
    return q.ask + fee(q)


def american(p: float) -> str:
    """Price of a $1 contract (fees included) as American odds."""
    if p <= 0 or p >= 1:
        return "—"
    if p >= 0.5:
        return f"{-100 * p / (1 - p):.0f}"
    return f"+{100 * (1 - p) / p:.0f}"


def qdict(q: Quote) -> dict:
    e = eff(q)
    return dict(venue=q.venue, label=q.label, ask=round(q.ask, 4), fee=round(fee(q), 4),
                eff=round(e, 4), american=american(e), size=q.size, url=q.url, market_id=q.market_id)


def group(quotes: list[Quote]) -> dict:
    """{game_id: {(market, line): {"yes": [q...], "no": [q...]}}}"""
    g: dict = defaultdict(lambda: defaultdict(lambda: {"yes": [], "no": []}))
    for q in quotes:
        g[q.game_id][(q.market, q.line)][q.side].append(q)
    return g


def best(qs: list[Quote]) -> Quote | None:
    return min(qs, key=eff) if qs else None


def main_line(rows: dict, market: str) -> float | None:
    """The line whose over/home side is closest to a coin flip (by mid price)."""
    cands = []
    for (mk, ln), sides in rows.items():
        if mk != market or ln is None:
            continue
        by = best(sides["yes"]), best(sides["no"])
        if by[0] is None or by[1] is None:
            continue
        mid = (by[0].ask + (1 - by[1].ask)) / 2
        cands.append((abs(mid - 0.5), ln))
    return min(cands)[1] if cands else None


def venue_main_lines(rows: dict, market: str) -> dict[str, float]:
    """Each venue's own main line (closest to 50/50 on that venue)."""
    per: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: {"yes": [], "no": []}))
    for (mk, ln), sides in rows.items():
        if mk != market:
            continue
        for side in ("yes", "no"):
            for q in sides[side]:
                per[q.venue][(mk, ln)][side].append(q)
    return {v: main_line(r, market) for v, r in per.items() if main_line(r, market) is not None}


def arbs(quotes: list[Quote], games: dict[int, dict]) -> list[dict]:
    """Every YES/NO pair on the same canonical proposition costing < $1 after fees."""
    out = []
    for gid, rows in group(quotes).items():
        game = games.get(gid)
        if not game:
            continue
        for (mk, ln), sides in rows.items():
            if ln is not None and abs(ln - round(ln)) < 1e-9:
                continue   # whole-number line: pushes are possible and venues settle them differently
            for y, n in product(sides["yes"], sides["no"]):
                if y.venue == n.venue and y.market_id == n.market_id:
                    continue   # the two sides of one book
                cost = eff(y) + eff(n)
                profit = 1 - cost
                if profit <= 0:
                    continue
                size = None
                if y.size is not None and n.size is not None:
                    size = min(y.size, n.size)
                flags = []
                if profit > MAX_SANE_EDGE:
                    flags.append("Edge over 5c: often a rules mismatch or a stale quote. Check both rulebooks.")
                if y.size is None or n.size is None:
                    flags.append("Size at this price unknown on one leg.")
                if y.venue == n.venue:
                    flags.append("Both legs on one venue (two different markets).")
                out.append(dict(
                    game_id=gid, matchup=f"{game['away_short']} @ {game['home_short']}", tip=game["tip_et"],
                    market=mk, line=ln, description=describe(mk, ln, game),
                    legs=[qdict(y), qdict(n)], cost=round(cost, 4), profit=round(profit, 4),
                    roi=round(profit / cost, 4), max_contracts=size,
                    max_profit=round(size * profit, 2) if size else None, flags=flags))
    out.sort(key=lambda a: -a["profit"])
    return out


def describe(mk: str, ln: float | None, game: dict) -> str:
    h, a = game["home_short"], game["away_short"]
    if mk == "ml":
        return f"Moneyline: {a} vs {h}"
    if mk == "spread":
        return f"Spread: {h} {-ln:+g} / {a} {ln:+g}"
    return f"Total {ln:g}"


def game_lines(rows: dict, game: dict) -> dict:
    """Best price on every side of every line we have for one game."""
    h, a = game["home_short"], game["away_short"]
    out = {"ml": None, "spreads": [], "totals": []}
    for (mk, ln), sides in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else 0)):
        by, bn = best(sides["yes"]), best(sides["no"])
        venues = sorted({q.venue for q in sides["yes"] + sides["no"]})
        entry = dict(line=ln, venues=venues,
                     yes=qdict(by) if by else None, no=qdict(bn) if bn else None,
                     all_yes=[qdict(q) for q in sorted(sides["yes"], key=eff)],
                     all_no=[qdict(q) for q in sorted(sides["no"], key=eff)])
        if mk == "ml":
            entry.update(yes_label=f"{h} win", no_label=f"{a} win")
            out["ml"] = entry
        elif mk == "spread":
            entry.update(yes_label=f"{h} {-ln:+g}", no_label=f"{a} {ln:+g}")
            out["spreads"].append(entry)
        else:
            entry.update(yes_label=f"Over {ln:g}", no_label=f"Under {ln:g}")
            out["totals"].append(entry)
    # spreads sorted from the home side's point of view
    out["spreads"].sort(key=lambda e: e["line"])
    out["totals"].sort(key=lambda e: e["line"])
    out["main_total"] = main_line(rows, "total")
    out["main_spread"] = main_line(rows, "spread")
    out["venue_main_totals"] = venue_main_lines(rows, "total")
    return out
