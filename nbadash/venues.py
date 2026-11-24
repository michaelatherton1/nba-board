"""Public NBA market data from Kalshi, Polymarket US and Novig, normalised.

Every price becomes a Quote on one of three canonical propositions per game:

    ml       "home team wins"                         side yes/no
    spread   "home margin > X"   (X can be negative)  side yes/no
    total    "total points > L"                       side yes (over) / no (under)

``ask`` is what one $1 contract on that side costs to buy right now (taker),
``size`` how many contracts are offered at that price (None if unknown).

None of these endpoints needs an API key. Field names follow the venues' public
docs and the open-source arbscan project; anything a venue returns that doesn't
parse is skipped and counted, and raw samples are saved for debugging.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable

import requests

from .teams import resolve, resolve_all, resolve_code

log = logging.getLogger(__name__)
UA = {"User-Agent": "nba-dashboard (github actions; read-only public data)"}

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
PMUS = "https://gateway.polymarket.us"
NOVIG = "https://api.novig.com"
KALSHI_SERIES = {"ml": "KXNBAGAME", "spread": "KXNBASPREAD", "total": "KXNBATOTAL"}


@dataclass
class Quote:
    venue: str
    teams: tuple[str, str]          # the two teams, any order (game matching fixes home/away)
    start: float | None             # epoch seconds, if the venue gives one
    market: str                     # ml | spread | total
    # Venue-native description of what YES on this quote means, resolved later:
    #   ml:     ("win", team)
    #   spread: ("cover", team, line)   team's margin + line > 0
    #   total:  ("over", L) or ("under", L)
    claim: tuple
    ask: float
    size: float | None
    fee_coef: float
    fee_live_only: bool
    label: str
    url: str | None
    market_id: str
    # filled in by assign():
    game_id: int | None = None
    line: float | None = None
    side: str | None = None


@dataclass
class VenueResult:
    name: str
    quotes: list[Quote] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    markets_seen: int = 0
    skipped: int = 0
    raw_samples: list = field(default_factory=list)


class Http:
    def __init__(self, base: str, rps: float):
        self.base, self.interval, self._next = base.rstrip("/"), 1.0 / rps, 0.0
        self.s = requests.Session()
        self.s.headers.update(UA)

    def get(self, path: str, params: Any = None) -> Any:
        for attempt in range(5):
            wait = self._next - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next = time.monotonic() + self.interval
            try:
                r = self.s.get(self.base + path, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()


def _f(v) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _ts(s) -> float | None:
    if not s:
        return None
    if isinstance(s, (int, float)):
        return s / 1000 if s > 1e11 else float(s)
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _num_in(text: str) -> float | None:
    m = re.search(r"([+-]?\d+(?:\.\d+)?)", text or "")
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------
# Kalshi
# --------------------------------------------------------------------------
def kalshi_coef(series: dict) -> float:
    mult = series.get("fee_multiplier")
    mult = 1.0 if mult is None else float(mult)
    if series.get("fee_type") in ("quadratic", "quadratic_with_maker_fees", "quadratic_with_combo_maker_fees"):
        return 0.07 * mult
    return 0.07 * max(mult, 1.0)   # unknown fee type: never understate


_K_MONTHS = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                          "NOV", "DEC"), start=1)}


def _kalshi_date(event_ticker: str) -> float | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker or "")
    if not m or m.group(2) not in _K_MONTHS:
        return None
    # noon ET that day, as a matching anchor (NBA tickers don't carry tip time)
    return datetime(2000 + int(m.group(1)), _K_MONTHS[m.group(2)], int(m.group(3)), 17,
                    tzinfo=timezone.utc).timestamp()


def fetch_kalshi() -> VenueResult:
    res = VenueResult("Kalshi")
    api = Http(KALSHI, 8)
    try:
        markets: list[tuple[str, dict, dict, float]] = []
        for kind, series in KALSHI_SERIES.items():
            try:
                s = api.get(f"/series/{series}").get("series") or {}
            except requests.HTTPError as e:
                log.warning("Kalshi series %s: %s", series, e)
                continue
            coef = kalshi_coef(s)
            cursor = None
            while True:
                params = {"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                d = api.get("/events", params)
                for ev in d.get("events") or []:
                    if len(res.raw_samples) < 3:
                        res.raw_samples.append({"series": series, "event": ev})
                    for m in ev.get("markets") or []:
                        markets.append((kind, ev, m, coef))
                cursor = d.get("cursor")
                if not cursor or not d.get("events"):
                    break
        res.markets_seen = len(markets)
        books = _kalshi_books(api, [m["ticker"] for _, _, m, _ in markets if m.get("status") in ("active", "open")])
        for kind, ev, m, coef in markets:
            try:
                res.quotes.extend(_kalshi_quotes(kind, ev, m, coef, books.get(m["ticker"])))
            except Exception as e:  # one odd market shouldn't sink the venue
                res.skipped += 1
                log.debug("Kalshi skip %s: %s", m.get("ticker"), e)
    except Exception as e:
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        log.warning("Kalshi failed: %s", res.error)
    return res


def _kalshi_books(api: Http, tickers: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            d = api.get("/markets/orderbooks", [("tickers", t) for t in batch])
            for ob in d.get("orderbooks") or []:
                out[ob.get("ticker")] = ob.get("orderbook_fp") or ob.get("orderbook") or {}
        except Exception as e:
            log.warning("Kalshi orderbooks: %s", e)
    return out


def _best_bid(levels) -> tuple[float | None, float | None]:
    """Highest bid from a list of [price, qty] (dollars or cents)."""
    best_p, best_q = None, None
    for lv in levels or []:
        try:
            p, q = float(lv[0]), float(lv[1])
        except (TypeError, ValueError, IndexError):
            continue
        if p > 1.0:
            p /= 100.0
        if best_p is None or p > best_p:
            best_p, best_q = p, q
    return best_p, best_q


def _kalshi_quotes(kind: str, ev: dict, m: dict, coef: float, book: dict | None) -> list[Quote]:
    if m.get("status") not in ("active", "open"):
        return []
    title = " | ".join(x for x in (ev.get("title"), ev.get("sub_title")) if x)
    teams = resolve_all(title)
    if len(teams) != 2:
        return []
    yes_txt = m.get("yes_sub_title") or m.get("subtitle") or m.get("title") or ""
    if kind == "ml":
        t = resolve(yes_txt) or resolve_code((m.get("ticker") or "").rsplit("-", 1)[-1])
        if t not in teams:
            return []
        claim_yes, claim_no = ("win", t), ("win", teams[1] if t == teams[0] else teams[0])
        lab_yes, lab_no = f"{t} win", f"{claim_no[1]} win"
    elif kind == "spread":
        t = resolve(yes_txt) or resolve_code((m.get("ticker") or "").rsplit("-", 1)[-1].rstrip("0123456789"))
        s = _f(m.get("floor_strike"))
        if s is None:
            s = _num_in(yes_txt)
        if t not in teams or s is None:
            return []
        other = teams[1] if t == teams[0] else teams[0]
        # YES: t wins by more than s  ==  t covers -s ; NO: other covers +s
        claim_yes, claim_no = ("cover", t, -abs(s)), ("cover", other, abs(s))
        lab_yes, lab_no = f"{t} -{abs(s):g}", f"{other} +{abs(s):g}"
    else:
        L = _f(m.get("floor_strike"))
        if L is None:
            L = _num_in(yes_txt)
        if L is None:
            return []
        claim_yes, claim_no = ("over", L), ("under", L)
        lab_yes, lab_no = f"Over {L:g}", f"Under {L:g}"

    yes_ask = _f(m.get("yes_ask_dollars"))
    no_ask = _f(m.get("no_ask_dollars"))
    if yes_ask is None and _f(m.get("yes_ask")) is not None:
        yes_ask = _f(m.get("yes_ask")) / 100
    if no_ask is None and _f(m.get("no_ask")) is not None:
        no_ask = _f(m.get("no_ask")) / 100
    yes_size = no_size = None
    if book:
        nb_p, nb_q = _best_bid(book.get("no_dollars") or book.get("no"))
        yb_p, yb_q = _best_bid(book.get("yes_dollars") or book.get("yes"))
        if nb_p is not None:
            yes_ask, yes_size = round(1 - nb_p, 4), nb_q
        if yb_p is not None:
            no_ask, no_size = round(1 - yb_p, 4), yb_q
    start = _ts(m.get("occurrence_datetime")) or _kalshi_date(ev.get("event_ticker", ""))
    url = f"https://kalshi.com/markets/{(ev.get('series_ticker') or '').lower()}/x/{(ev.get('event_ticker') or '').lower()}"
    out = []
    for ask, size, claim, lab in ((yes_ask, yes_size, claim_yes, lab_yes), (no_ask, no_size, claim_no, lab_no)):
        if ask is not None and 0 < ask < 1:
            out.append(Quote("Kalshi", (teams[0], teams[1]), start, kind, claim, ask, size, coef, False,
                             lab, url, m["ticker"]))
    return out


# --------------------------------------------------------------------------
# Polymarket US
# --------------------------------------------------------------------------
def fetch_pmus() -> VenueResult:
    res = VenueResult("Polymarket US")
    api = Http(PMUS, 8)
    try:
        events, offset = [], 0
        while True:
            d = api.get("/v2/leagues/nba/events", {"limit": 50, "offset": offset})
            evs = d.get("events") or []
            events.extend(evs)
            if len(evs) < 50 or offset > 1000:
                break
            offset += 50
        now = time.time()
        for ev in events:
            if len(res.raw_samples) < 2:
                res.raw_samples.append(ev)
            # The league endpoint returns only the main market; pull the game's full moneyline/spread/total set.
            mkts = {m.get("slug") or m.get("id"): m for m in ev.get("markets") or []}
            st = _ts(ev.get("startTime")) or _ts(ev.get("eventDate"))
            eid = ev.get("id")
            if eid and st and -6 * 3600 < st - now < 60 * 3600:
                try:
                    d = api.get(f"/v1/events/{eid}")
                    full = (d.get("event") or d) if isinstance(d, dict) else {}
                    for m in full.get("markets") or []:
                        if (m.get("sportsMarketType") or "").startswith("basketball_team_full_game_"):
                            mkts.setdefault(m.get("slug") or m.get("id"), m)
                except Exception as e:
                    log.warning("PM US event %s: %s", eid, e)
            if len(res.raw_samples) < 4 and len(mkts) > 1:
                res.raw_samples.append({"title": ev.get("title"), "startTime": ev.get("startTime"),
                                        "markets": list(mkts.values())[:6]})
            for m in mkts.values():
                res.markets_seen += 1
                try:
                    res.quotes.extend(_pm_quotes(ev, m))
                except Exception as e:
                    res.skipped += 1
                    log.debug("PM US skip %s: %s", m.get("slug"), e)
    except Exception as e:
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        log.warning("Polymarket US failed: %s", res.error)
    return res


def _amount(a) -> float | None:
    if isinstance(a, dict):
        return _f(a.get("value"))
    return _f(a)


def _pm_side_team(side: dict, teams: list[str]) -> str | None:
    t = side.get("team") or {}
    for v in (t.get("name"), t.get("safeName"), side.get("description"), side.get("identifier")):
        r = resolve(v or "")
        if r in teams:
            return r
    ab = resolve_code(t.get("abbreviation"))
    return ab if ab in teams else None


def _pm_quotes(ev: dict, m: dict) -> list[Quote]:
    if m.get("closed") or m.get("active") is False or m.get("hidden"):
        return []
    kind_raw = (m.get("marketType") or m.get("sportsMarketType") or "").lower()
    smt = (m.get("sportsMarketType") or "").lower()
    if smt and smt.startswith("basketball_") and not smt.startswith("basketball_team_full_game_"):
        return []
    if "first" in smt or "half" in smt or "1h" in smt or "quarter" in smt or "team_total" in smt or "player" in smt:
        return []
    if kind_raw.startswith("moneyline"):
        kind = "ml"
    elif kind_raw.startswith("spread"):
        kind = "spread"
    elif kind_raw.startswith("total"):
        kind = "total"
    else:
        return []
    teams = resolve_all(" | ".join(str(x) for x in (ev.get("title"), ev.get("subtitle"), ev.get("description")) if x))
    if len(teams) != 2:
        tl = [resolve((t or {}).get("name") or "") for t in ev.get("teams") or []]
        teams = [t for t in tl if t]
    if len(teams) != 2:
        return []
    sides = m.get("marketSides") or []
    long_side = next((s for s in sides if s.get("long") is True), None)
    short_side = next((s for s in sides if s.get("long") is False), None)
    if long_side is None:
        return []
    line = _f(m.get("line"))
    if kind == "ml":
        t = _pm_side_team(long_side, teams)
        if t is None:
            return []
        other = teams[1] if t == teams[0] else teams[0]
        claim_l, claim_s = ("win", t), ("win", other)
        lab_l, lab_s = f"{t} win", f"{other} win"
    elif kind == "spread":
        t = _pm_side_team(long_side, teams)
        ln = _num_in(long_side.get("description") or "")
        if ln is None:
            ln = line
        if t is None or ln is None:
            return []
        other = teams[1] if t == teams[0] else teams[0]
        claim_l, claim_s = ("cover", t, ln), ("cover", other, -ln)
        lab_l, lab_s = f"{t} {ln:+g}", f"{other} {-ln:+g}"
    else:
        L = line if line is not None else _num_in(long_side.get("description") or m.get("question") or "")
        if L is None:
            return []
        L = abs(L)
        desc = (long_side.get("description") or "").lower()
        if desc.startswith("under"):
            claim_l, claim_s, lab_l, lab_s = ("under", L), ("over", L), f"Under {L:g}", f"Over {L:g}"
        else:
            claim_l, claim_s, lab_l, lab_s = ("over", L), ("under", L), f"Over {L:g}", f"Under {L:g}"
    bid, ask = _amount(m.get("bestBidQuote")), _amount(m.get("bestAskQuote"))
    coef = _f(m.get("feeCoefficient")) or 0.0695
    start = _ts(ev.get("startTime")) or _ts(m.get("gameStartTime")) or _ts(ev.get("eventDate")) or _ts(ev.get("startDate"))
    url = f"https://polymarket.us/market/{m.get('slug')}" if m.get("slug") else None
    out = []
    if ask is not None and 0 < ask < 1:
        out.append(Quote("Polymarket US", (teams[0], teams[1]), start, kind, claim_l, ask, None, coef, False,
                         lab_l, url, m.get("slug") or str(m.get("id"))))
    if bid is not None and 0 < bid < 1:
        out.append(Quote("Polymarket US", (teams[0], teams[1]), start, kind, claim_s, round(1 - bid, 4), None,
                         coef, False, lab_s, url, m.get("slug") or str(m.get("id"))))
    return out


def pmus_book_sizes(quotes: Iterable[Quote]) -> None:
    """Fill ``size`` for the given Polymarket US quotes from the order book."""
    api = Http(PMUS, 8)
    by_slug: dict[str, list[Quote]] = {}
    for q in quotes:
        if q.venue == "Polymarket US":
            by_slug.setdefault(q.market_id, []).append(q)
    for slug, qs in by_slug.items():
        try:
            md = api.get(f"/v1/markets/{slug}/book").get("marketData") or {}
        except Exception as e:
            log.warning("PM US book %s: %s", slug, e)
            continue
        offers = [(_amount(o.get("px")), _f(o.get("qty"))) for o in md.get("offers") or []]
        bids = [(_amount(o.get("px")), _f(o.get("qty"))) for o in md.get("bids") or []]
        offers = [o for o in offers if o[0] is not None]
        bids = [b for b in bids if b[0] is not None]
        best_offer = min(offers, default=None)
        best_bid = max(bids, default=None)
        for q in qs:
            # long-side quote was priced from bestAsk, short side from 1 - bestBid
            if best_offer and abs(q.ask - best_offer[0]) < 1e-6:
                q.size = best_offer[1]
            elif best_bid and abs(q.ask - (1 - best_bid[0])) < 1e-6:
                q.size = best_bid[1]


# --------------------------------------------------------------------------
# Novig
# --------------------------------------------------------------------------
NOVIG_TYPES = {"MONEY": "ml", "SPREAD": "spread", "TOTAL": "total"}


def _novig_pages(api: Http, path: str, params: dict) -> list[dict]:
    out, after = [], None
    for _ in range(50):
        q = dict(params, limit=500)
        if after:
            q["after"] = after
        d = api.get(path, q)
        out.extend(d.get("items") or [])
        after = d.get("next")
        if not after or not d.get("items"):
            break
    return out


def fetch_novig() -> VenueResult:
    res = VenueResult("Novig")
    api = Http(NOVIG, 1.8)       # Novig limits public reads to ~2/s per IP
    try:
        events = {e["eventId"]: e for e in _novig_pages(api, "/v3/public/catalog/events",
                                                        {"league": "NBA", "status": "OPEN_PREGAME"})}
        markets = _novig_pages(api, "/v3/public/catalog/markets",
                               {"league": "NBA", "marketType": "MONEY,SPREAD,TOTAL", "eventStatus": "OPEN_PREGAME"})
        res.raw_samples = [{"event": next(iter(events.values()), None), "markets": markets[:3]}]
        res.markets_seen = len(markets)
        for m in markets:
            ev = events.get(m.get("eventId"))
            if ev is None or m.get("status") != "OPEN" or m.get("marketType") not in NOVIG_TYPES:
                continue
            try:
                book = api.get(f"/v3/public/catalog/markets/{m['marketId']}/book")
            except Exception as e:
                log.warning("Novig book %s: %s", m.get("marketId"), e)
                book = None
            try:
                res.quotes.extend(_novig_quotes(ev, m, book))
            except Exception as e:
                res.skipped += 1
                log.debug("Novig skip %s: %s", m.get("marketId"), e)
    except Exception as e:
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        log.warning("Novig failed: %s", res.error)
    return res


def _novig_quotes(ev: dict, m: dict, book: dict | None) -> list[Quote]:
    desc = ev.get("description") or ""
    if " @ " not in desc:
        return []
    away_txt, home_txt = desc.split(" @ ", 1)
    away, home = resolve(away_txt) or resolve_code(away_txt), resolve(home_txt) or resolve_code(home_txt)
    if not away or not home:
        return []
    teams = [away, home]
    kind = NOVIG_TYPES[m["marketType"]]
    outs = m.get("outcomes") or []
    if len(outs) != 2:
        return []
    orders = (book or {}).get("orders") or {}

    def best_bid(oid, outcome):
        lv = orders.get(oid) or []
        best = None
        for o in lv:
            p, q = _f(o.get("price")), _f(o.get("qty"))
            if p is None:
                continue
            if best is None or p > best[0]:
                best = (p, (q or 0) / 100)       # Novig qty counts 1-cent contracts
        if best is None and _f(outcome.get("bid")) is not None:
            best = (_f(outcome.get("bid")), None)
        return best

    fee = m.get("fee") or {}
    coef = _f(fee.get("coefficient")) or 0.03
    live_only = (fee.get("charged") or "WHEN_LIVE") == "WHEN_LIVE"
    strike = _f(m.get("strike"))
    start = _ts(m.get("startsTs")) or _ts(ev.get("startsTs"))
    out = []
    for i, o in enumerate(outs):
        other = outs[1 - i]
        ob = best_bid(other["outcomeId"], other)   # buying o = selling into other's bids
        if ob is None:
            continue
        ask, size = round(1 - ob[0], 4), ob[1]
        name = (o.get("name") or "").strip()
        if kind == "ml":
            t = resolve(name) or resolve_code(name.split()[0] if name else "")
            if t not in teams:
                continue
            claim, lab = ("win", t), f"{t} win"
        elif kind == "spread":
            parts = name.split()
            t = resolve_code(parts[0]) if parts else None
            t = t or resolve(name)
            ln = _num_in(" ".join(parts[1:])) if len(parts) > 1 else None
            if ln is None and strike is not None:
                ln = strike
            if t not in teams or ln is None:
                continue
            claim, lab = ("cover", t, ln), f"{t} {ln:+g}"
        else:
            L = strike if strike is not None else _num_in(name)
            if L is None:
                continue
            if name.lower().startswith("over"):
                claim, lab = ("over", abs(L)), f"Over {abs(L):g}"
            elif name.lower().startswith("under"):
                claim, lab = ("under", abs(L)), f"Under {abs(L):g}"
            else:
                continue
        if 0 < ask < 1:
            out.append(Quote("Novig", (away, home), start, kind, claim, ask, size, coef, live_only,
                             lab, None, m["marketId"]))
    return out


# --------------------------------------------------------------------------
# Canonical mapping
# --------------------------------------------------------------------------
def canonical(q: Quote, home: str, away: str) -> tuple[float | None, str] | None:
    """(line, side) on the canonical proposition for this game, or None."""
    c = q.claim
    if q.market == "ml":
        return (None, "yes" if c[1] == home else "no")
    if q.market == "spread":
        _, team, ln = c
        # team covers ln  <=>  margin_team > -ln
        if team == home:
            return (-ln, "yes")          # home margin > -ln
        return (ln, "no")                # away margin > -ln  <=>  NOT(home margin > ln)
    kind, L = c
    return (L, "yes" if kind == "over" else "no")


def assign(quotes: list[Quote], slate: list[dict]) -> list[Quote]:
    """Attach each quote to a game on the slate (same two teams, start within 30 h)."""
    out = []
    for q in quotes:
        pair = set(q.teams)
        best = None
        for g in slate:
            if {g["home"], g["away"]} != pair:
                continue
            dt = abs((q.start or g["tip_ts"]) - g["tip_ts"])
            if dt <= 30 * 3600 and (best is None or dt < best[0]):
                best = (dt, g)
        if best is None:
            continue
        g = best[1]
        ls = canonical(q, g["home"], g["away"])
        if ls is None:
            continue
        q.game_id, (q.line, q.side) = g["game_id"], ls
        out.append(q)
    return out


def dump_raw(results: list[VenueResult], path) -> None:
    with open(path, "w") as f:
        json.dump({r.name: {"ok": r.ok, "error": r.error, "markets_seen": r.markets_seen,
                            "quotes": len(r.quotes), "skipped": r.skipped,
                            "samples": r.raw_samples,
                            "quote_samples": [asdict(q) for q in r.quotes[:8]]} for r in results},
                  f, indent=1, default=str)
