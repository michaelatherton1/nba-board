"""Diagnostics: can the GitHub runner read the workbook's data sources, and what do the tables look like?"""
import io, sys, time
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}
URLS = {
    "tr_poss": "https://www.teamrankings.com/nba/stat/possessions-per-game",
    "bref_ratings": "https://www.basketball-reference.com/leagues/NBA_2027_ratings.html",
    "bref_per_game": "https://www.basketball-reference.com/leagues/NBA_2027_per_game.html",
    "bref_sched_nov": "https://www.basketball-reference.com/leagues/NBA_2027_games-november.html",
}
for k, u in URLS.items():
    try:
        r = requests.get(u, headers=UA, timeout=30)
        print(f"== {k}: HTTP {r.status_code}, {len(r.text)} chars, server={r.headers.get('server')}, "
              f"cf={'cf-ray' in r.headers}")
        if r.ok:
            html = r.text.replace("<!--", "").replace("-->", "")
            tabs = pd.read_html(io.StringIO(html))
            for i, t in enumerate(tabs[:4]):
                cols = [" | ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
                print(f"   table {i}: {t.shape}, cols={cols}")
                print("   first row:", t.iloc[0].tolist())
                print("   last row:", t.iloc[-1].tolist())
        else:
            print("   body:", r.text[:300].replace("\n", " "))
    except Exception as e:
        print(f"== {k}: FAILED {type(e).__name__}: {e}")
    time.sleep(4)   # stay well under Sports Reference's 20 requests/minute limit
