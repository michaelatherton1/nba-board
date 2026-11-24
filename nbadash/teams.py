"""The 30 NBA teams and every way the venues spell them.

`resolve()` turns a venue's label ("Boston", "BOS", "Los Angeles L", "Celtics",
"BOS -4.5 · Boston Celtics (bos)") into the ESPN display name used everywhere else.
"""

from __future__ import annotations

import re

# ESPN display name -> (city words, nickname, abbreviations)
TEAMS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "Atlanta Hawks": ("Atlanta", "Hawks", ("ATL",)),
    "Boston Celtics": ("Boston", "Celtics", ("BOS",)),
    "Brooklyn Nets": ("Brooklyn", "Nets", ("BKN", "BRK", "BK", "BKLYN")),
    "Charlotte Hornets": ("Charlotte", "Hornets", ("CHA", "CHO", "CHAR")),
    "Chicago Bulls": ("Chicago", "Bulls", ("CHI",)),
    "Cleveland Cavaliers": ("Cleveland", "Cavaliers", ("CLE", "CAVS")),
    "Dallas Mavericks": ("Dallas", "Mavericks", ("DAL", "MAVS")),
    "Denver Nuggets": ("Denver", "Nuggets", ("DEN",)),
    "Detroit Pistons": ("Detroit", "Pistons", ("DET",)),
    "Golden State Warriors": ("Golden State", "Warriors", ("GSW", "GS", "GOS")),
    "Houston Rockets": ("Houston", "Rockets", ("HOU",)),
    "Indiana Pacers": ("Indiana", "Pacers", ("IND",)),
    "LA Clippers": ("Los Angeles C", "Clippers", ("LAC",)),
    "Los Angeles Lakers": ("Los Angeles L", "Lakers", ("LAL",)),
    "Memphis Grizzlies": ("Memphis", "Grizzlies", ("MEM",)),
    "Miami Heat": ("Miami", "Heat", ("MIA",)),
    "Milwaukee Bucks": ("Milwaukee", "Bucks", ("MIL",)),
    "Minnesota Timberwolves": ("Minnesota", "Timberwolves", ("MIN", "MINN", "WOLVES")),
    "New Orleans Pelicans": ("New Orleans", "Pelicans", ("NOP", "NO", "NOR", "NOLA")),
    "New York Knicks": ("New York", "Knicks", ("NYK", "NY")),
    "Oklahoma City Thunder": ("Oklahoma City", "Thunder", ("OKC", "OKL")),
    "Orlando Magic": ("Orlando", "Magic", ("ORL",)),
    "Philadelphia 76ers": ("Philadelphia", "76ers", ("PHI", "PHL", "SIXERS")),
    "Phoenix Suns": ("Phoenix", "Suns", ("PHX", "PHO")),
    "Portland Trail Blazers": ("Portland", "Trail Blazers", ("POR", "BLAZERS")),
    "Sacramento Kings": ("Sacramento", "Kings", ("SAC",)),
    "San Antonio Spurs": ("San Antonio", "Spurs", ("SAS", "SA")),
    "Toronto Raptors": ("Toronto", "Raptors", ("TOR",)),
    "Utah Jazz": ("Utah", "Jazz", ("UTA", "UTAH", "UTH")),
    "Washington Wizards": ("Washington", "Wizards", ("WAS", "WSH", "WASH")),
}

ABBR = {name: abbrs[0] for name, (_, _, abbrs) in TEAMS.items()}
NICK = {name: nick for name, (_, nick, _) in TEAMS.items()}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


# Phrase aliases (multi-word first so "golden state" beats "state").
_PHRASES: list[tuple[str, str]] = []
for _name, (_city, _nick, _abbrs) in TEAMS.items():
    _PHRASES.append((_norm(_name), _name))
    _PHRASES.append((_norm(_nick), _name))
    if not _city.startswith("Los Angeles"):
        _PHRASES.append((_norm(_city), _name))
    else:
        _PHRASES.append((_norm(_city), _name))                      # "los angeles l"
        _PHRASES.append((_norm("LA " + _nick), _name))
        _PHRASES.append((_norm("L.A. " + _nick), _name))
_PHRASES.append(("trail blazers", "Portland Trail Blazers"))
_PHRASES.append(("sixers", "Philadelphia 76ers"))
_PHRASES.append(("la clippers", "LA Clippers"))
_PHRASES.append(("los angeles clippers", "LA Clippers"))
_PHRASES.sort(key=lambda x: -len(x[0]))

_ABBRS: dict[str, str] = {}
for _name, (_, _, _abbrs) in TEAMS.items():
    for _a in _abbrs:
        _ABBRS[_a] = _name


def resolve(text: str | None) -> str | None:
    """Best single team named in ``text``; None if none or ambiguous."""
    if not text:
        return None
    t = " " + _norm(text) + " "
    hits: list[str] = []
    for phrase, name in _PHRASES:
        if f" {phrase} " in t:
            hits.append(name)
            t = t.replace(f" {phrase} ", " ")
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return None
    # Abbreviations: exact upper-case tokens only (avoid words like "no", "min", "was").
    found = {_ABBRS.get(tok) for tok in re.findall(r"\b[A-Z]{2,5}\b", text)}
    found.discard(None)
    if len(found) == 1:
        return found.pop()
    return None


def resolve_code(code: str | None) -> str | None:
    """A bare team code in any case ("bos", "GSW", "no")."""
    return _ABBRS.get((code or "").strip().upper())


def resolve_all(text: str | None) -> list[str]:
    """All teams named in ``text`` in order of appearance (phrases, then caps abbrs)."""
    if not text:
        return []
    t = " " + _norm(text) + " "
    out: list[tuple[int, str]] = []
    for phrase, name in _PHRASES:
        i = t.find(f" {phrase} ")
        if i >= 0 and name not in [n for _, n in out]:
            out.append((i, name))
            t = t.replace(f" {phrase} ", " " + "_" * len(phrase) + " ")
    if not out:
        for m in re.finditer(r"\b[A-Z]{2,5}\b", text):
            name = _ABBRS.get(m.group(0))
            if name and name not in [n for _, n in out]:
                out.append((m.start(), name))
    return [n for _, n in sorted(out)]
