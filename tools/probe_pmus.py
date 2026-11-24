"""Probe Polymarket US endpoints for one upcoming NBA event and print what each returns (diagnostics)."""
import json, requests
B = "https://gateway.polymarket.us"
S = requests.Session(); S.headers["User-Agent"] = "nba-board probe"
def get(path, params=None):
    r = S.get(B + path, params=params, timeout=30)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:300])
code, d = get("/v2/leagues/nba/events", {"limit": 20})
evs = d.get("events") or []
ev = next((e for e in evs if (e.get("marketCounts") or {}).get("numSpreadsAndTotalsMarkets")), evs[0])
print("event", ev.get("id"), ev.get("slug"), ev.get("gameId"), ev.get("fixtureId"), ev.get("title"), ev.get("marketCounts"))
print("marketGroups in list payload:", json.dumps(ev.get("marketGroups"))[:600])
def summ(label, code, d):
    ms = (d.get("markets") if isinstance(d, dict) else None) or ((d.get("event") or {}).get("markets") if isinstance(d, dict) else None) or []
    types = {}
    for m in ms:
        k = (m.get("marketType"), m.get("sportsMarketType"))
        types[k] = types.get(k, 0) + 1
    print(f"{label}: HTTP {code}, markets={len(ms)}, types={dict(list(types.items())[:12])}")
    if ms:
        m = next((x for x in ms if (x.get("marketType") or "").startswith("total")), ms[0])
        print("   sample:", json.dumps({k: m.get(k) for k in ("slug", "marketType", "sportsMarketType", "line", "question", "bestBidQuote", "bestAskQuote", "marketSides", "active", "closed")})[:900])
    elif isinstance(d, dict):
        print("   keys:", list(d.keys())[:20], json.dumps(d)[:300])
summ("markets?gameId", *get("/v1/markets", [("gameId", str(ev.get("gameId"))), ("limit", "500")]))
summ("markets?gameId+types", *get("/v1/markets", [("gameId", str(ev.get("gameId"))), ("limit", "500"), ("sportsMarketTypes", "SPORTS_MARKET_TYPE_TOTAL"), ("sportsMarketTypes", "SPORTS_MARKET_TYPE_SPREAD")]))
summ("events/{id}", *get(f"/v1/events/{ev.get('id')}"))
summ("events/slug/{slug}", *get(f"/v1/events/slug/{ev.get('slug')}"))
code, d = get(f"/v1/events/slug/{ev.get('slug')}", {"groups": "true"})
summ("events/slug?groups=true", code, d)
grp = (d.get("event") or d).get("marketGroups") if isinstance(d, dict) else None
print("marketGroups (groups=true):", json.dumps(grp)[:1500])
