"""Offline checks of venue parsing, canonical mapping and arbitrage math.

Payloads mimic each venue's documented fields; the live shapes are confirmed by the
raw samples the GitHub Action saves on every run (debug/venues_raw.json).
"""

from nbadash import lines, venues
from nbadash.teams import resolve, resolve_all, resolve_code

HOME, AWAY = "Toronto Raptors", "Boston Celtics"
GAME = dict(game_id=1, home=HOME, away=AWAY, home_short="Raptors", away_short="Celtics",
            tip_ts=1_794_445_200.0, tip_et="Wed 7:00 PM", started=False)


def test_team_resolution():
    assert resolve("Boston") == "Boston Celtics"
    assert resolve("Los Angeles L") == "Los Angeles Lakers"
    assert resolve("LA Clippers") == "LA Clippers"
    assert resolve("BOS -4.5") == "Boston Celtics"
    assert resolve("Trail Blazers") == "Portland Trail Blazers"
    assert resolve("Nets") == "Brooklyn Nets" and resolve("Hornets") == "Charlotte Hornets"
    assert resolve_code("gsw") == "Golden State Warriors"
    assert resolve_all("Boston at Toronto") == ["Boston Celtics", "Toronto Raptors"]
    assert resolve_all("Celtics vs. Raptors") == ["Boston Celtics", "Toronto Raptors"]
    assert resolve("Oklahoma City") == "Oklahoma City Thunder"
    assert resolve("New York") == "New York Knicks"


def kalshi_ev(series, markets):
    return {"event_ticker": f"{series}-26NOV11BOSTOR", "series_ticker": series,
            "title": "Boston at Toronto", "markets": markets}


def test_kalshi_mapping():
    ml = venues._kalshi_quotes("ml", kalshi_ev("KXNBAGAME", []),
                               {"ticker": "KXNBAGAME-26NOV11BOSTOR-BOS", "status": "active",
                                "yes_sub_title": "Boston", "yes_ask_dollars": "0.6200", "no_ask_dollars": "0.3900"},
                               0.07, None)
    sp = venues._kalshi_quotes("spread", kalshi_ev("KXNBASPREAD", []),
                               {"ticker": "KXNBASPREAD-26NOV11BOSTOR-BOS4", "status": "active",
                                "yes_sub_title": "Boston wins by over 4.5 Points", "floor_strike": 4.5,
                                "yes_ask_dollars": "0.48", "no_ask_dollars": "0.54"},
                               0.07, {"yes_dollars": [["0.45", "100"], ["0.46", "250"]],
                                      "no_dollars": [["0.51", "80"], ["0.52", "300"]]})
    tot = venues._kalshi_quotes("total", kalshi_ev("KXNBATOTAL", []),
                                {"ticker": "KXNBATOTAL-26NOV11BOSTOR-215", "status": "active",
                                 "yes_sub_title": "Over 215.5 points scored", "floor_strike": 215.5,
                                 "yes_ask_dollars": "0.50", "no_ask_dollars": "0.52"}, 0.07, None)
    q = venues.assign(ml + sp + tot, [GAME])
    got = {(x.market, x.line, x.side, x.label): x.ask for x in q}
    assert got[("ml", None, "no", "Boston Celtics win")] == 0.62          # away win = NO on home win
    assert got[("ml", None, "yes", "Toronto Raptors win")] == 0.39
    # Boston (away) wins by > 4.5  ==  NOT(home margin > -4.5)
    assert got[("spread", -4.5, "no", "Boston Celtics -4.5")] == 0.48     # 1 - best NO bid 0.52
    assert got[("spread", -4.5, "yes", "Toronto Raptors +4.5")] == 0.54   # 1 - best YES bid 0.46
    sizes = {x.label: x.size for x in q if x.market == "spread"}
    assert sizes["Boston Celtics -4.5"] == 300 and sizes["Toronto Raptors +4.5"] == 250
    assert got[("total", 215.5, "yes", "Over 215.5")] == 0.50


def test_pmus_mapping():
    ev = {"title": "Celtics vs. Raptors", "startTime": "2026-11-12T00:00:00Z"}
    ml = venues._pm_quotes(ev, {"slug": "aec-nba-bos-tor-2026-11-11", "marketType": "moneyline", "active": True,
                                "closed": False, "bestBidQuote": {"value": "0.40"}, "bestAskQuote": {"value": "0.42"},
                                "feeCoefficient": "0.0695",
                                "marketSides": [{"long": True, "description": "Raptors", "team": {"name": "Raptors", "abbreviation": "tor"}},
                                                {"long": False, "description": "Celtics", "team": {"name": "Celtics", "abbreviation": "bos"}}]})
    sp = venues._pm_quotes(ev, {"slug": "asc-nba-bos-tor-2026-11-11-4pt5", "marketType": "spreads", "line": -4.5,
                                "active": True, "closed": False, "bestBidQuote": {"value": "0.47"},
                                "bestAskQuote": {"value": "0.49"},
                                "marketSides": [{"long": True, "description": "-4.50", "team": {"name": "Celtics", "abbreviation": "bos"}},
                                                {"long": False, "description": "+4.50", "team": {"name": "Raptors", "abbreviation": "tor"}}]})
    tot = venues._pm_quotes(ev, {"slug": "tsc-nba-bos-tor-2026-11-11-215pt5", "marketType": "totals", "line": 215.5,
                                 "active": True, "closed": False, "bestBidQuote": {"value": "0.51"},
                                 "bestAskQuote": {"value": "0.53"},
                                 "marketSides": [{"long": True, "description": "Over"}, {"long": False, "description": "Under"}]})
    q = venues.assign(ml + sp + tot, [GAME])
    got = {(x.market, x.line, x.side): x.ask for x in q}
    assert got[("ml", None, "yes")] == 0.42 and got[("ml", None, "no")] == 0.60
    assert got[("spread", -4.5, "no")] == 0.49      # Celtics -4.5 (away) long side
    assert got[("spread", -4.5, "yes")] == 0.53     # Raptors +4.5 = 1 - bid 0.47
    assert got[("total", 215.5, "yes")] == 0.53 and got[("total", 215.5, "no")] == 0.49


def test_novig_mapping_and_arb():
    ev = {"eventId": "e1", "description": "Boston Celtics @ Toronto Raptors", "startsTs": 1_794_445_200_000}
    m = {"marketId": "m1", "eventId": "e1", "marketType": "TOTAL", "strike": "215.5", "status": "OPEN",
         "fee": {"coefficient": 0.03, "charged": "WHEN_LIVE"},
         "outcomes": [{"outcomeId": "o", "name": "Over 215.5"}, {"outcomeId": "u", "name": "Under 215.5"}]}
    book = {"orders": {"o": [{"price": 0.47, "qty": 5000}], "u": [{"price": 0.55, "qty": 20000}]}}
    nq = venues._novig_quotes(ev, m, book)
    got = {x.claim: (x.ask, x.size) for x in nq}
    assert got[("over", 215.5)] == (0.45, 200.0)     # 1 - best Under bid 0.55, 20000 cents = 200 contracts
    assert got[("under", 215.5)] == (0.53, 50.0)
    # Kalshi Under 215.5 at 0.52, Novig Over at 0.45 (pregame fee 0) -> arb
    k = venues._kalshi_quotes("total", kalshi_ev("KXNBATOTAL", []),
                              {"ticker": "KXNBATOTAL-26NOV11BOSTOR-215", "status": "active",
                               "yes_sub_title": "Over 215.5 points scored", "floor_strike": 215.5},
                              0.07, {"yes_dollars": [["0.48", "60"]], "no_dollars": [["0.49", "40"]]})
    q = venues.assign(nq + k, [GAME])
    arbs = lines.arbs(q, {1: GAME})
    assert arbs, "expected an arb"
    a = arbs[0]
    kal_fee = 0.07 * 0.52 * 0.48
    assert abs(a["cost"] - (0.45 + 0.52 + kal_fee)) < 1e-4
    assert a["max_contracts"] == 60
    assert {leg["venue"] for leg in a["legs"]} == {"Novig", "Kalshi"}


def test_whole_number_lines_excluded():
    a = venues.Quote("Kalshi", (HOME, AWAY), None, "total", ("over", 216.0), 0.40, 10, 0.07, False, "Over 216", None, "a")
    b = venues.Quote("Novig", (HOME, AWAY), None, "total", ("under", 216.0), 0.40, 10, 0.03, True, "Under 216", None, "b")
    assert lines.arbs(venues.assign([a, b], [GAME]), {1: GAME}) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
