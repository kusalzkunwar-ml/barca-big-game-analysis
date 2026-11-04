"""Download StatsBomb open-data match lists and event files for every La Liga season.

Raw JSON is cached outside the repo (default ~/statsbomb-cache, override with SB_CACHE)
so re-runs only fetch what is missing.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = Path(os.environ.get("SB_CACHE", Path.home() / "statsbomb-cache"))
LA_LIGA = 11
# StatsBomb season ids for La Liga 2004/05 - 2020/21 (1973/74 is skipped: one match)
SEASONS = [37, 38, 39, 40, 41, 21, 22, 23, 24, 25, 26, 27, 2, 1, 4, 42, 90]


def get_json(rel_path: str):
    dest = CACHE / rel_path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{BASE}/{rel_path}", timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return json.loads(dest.read_text(encoding="utf-8"))


def all_matches():
    out = []
    for s in SEASONS:
        out += get_json(f"matches/{LA_LIGA}/{s}.json")
    return out


def main():
    matches = all_matches()
    ids = [m["match_id"] for m in matches]
    print(f"{len(ids)} matches across {len(SEASONS)} seasons; fetching events to {CACHE}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, _ in enumerate(pool.map(lambda mid: get_json(f"events/{mid}.json"), ids), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(ids)}")
    print("done")


if __name__ == "__main__":
    main()
