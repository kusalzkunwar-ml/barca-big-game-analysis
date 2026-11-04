"""Flatten cached StatsBomb JSON into three analysis tables (data/processed/*.parquet).

matches     one row per Barça La Liga match (season, opponent, score, manager, elite flag)
team_match  one row per team per Barça match: xG, PPDA, defensive height, field tilt, ...
shots       every shot in every cached match (incl. the full 2015/16 season) for the xG model

StatsBomb pitch is 120 x 80 and every event is recorded with its own team attacking
left-to-right, so x = 120 is always the opponent's goal line.
"""
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from fetch import LA_LIGA, SEASONS, get_json

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"
BARCA = "Barcelona"

# Elite opposition: Real Madrid every season, Atlético from 2012/13, when the Simeone
# side became a fixture in the top three (3rd, 1st, 3rd, 3rd, 3rd, 2nd, 2nd, 3rd, 1st).
ATLETI_ELITE_FROM = "2012/2013"

# Fallback when the match file has no manager listed.
MANAGER_BY_SEASON = {
    "2004/2005": "Rijkaard", "2005/2006": "Rijkaard", "2006/2007": "Rijkaard", "2007/2008": "Rijkaard",
    "2008/2009": "Guardiola", "2009/2010": "Guardiola", "2010/2011": "Guardiola", "2011/2012": "Guardiola",
    "2012/2013": "Vilanova", "2013/2014": "Martino", "2014/2015": "Luis Enrique", "2015/2016": "Luis Enrique",
    "2016/2017": "Luis Enrique", "2017/2018": "Valverde", "2018/2019": "Valverde", "2019/2020": "Valverde",
    "2020/2021": "Koeman",
}
SURNAME = {"Josep Guardiola i Sala": "Guardiola", "Francesc Vilanova Bayó": "Vilanova",
           "Gerardo Daniel Martino": "Martino", "Luis Enrique Martínez García": "Luis Enrique",
           "Ernesto Valverde Tejedor": "Valverde", "Quique Setién Solar": "Setién",
           "Ronald Koeman": "Koeman", "Frank Rijkaard": "Rijkaard",
           "Tito Vilanova": "Vilanova", "Quique Setién": "Setién"}

PPDA_DEF_ACTIONS = {"Interception", "Foul Committed"}           # + tackles (Duel/Tackle)
DEF_ACTIONS = {"Duel", "Interception", "Ball Recovery", "Clearance", "Block", "Pressure", "Foul Committed"}


def is_elite(opponent: str, season: str) -> bool:
    return opponent == "Real Madrid" or (opponent == "Atlético Madrid" and season >= ATLETI_ELITE_FROM)


def barca_manager(m: dict, barca_home: bool) -> str:
    side = m["home_team" if barca_home else "away_team"]
    names = [SURNAME.get(x["name"], x.get("nickname") or x["name"]) for x in side.get("managers", [])]
    names = [SURNAME.get(n, n) for n in names]
    return names[0] if names else MANAGER_BY_SEASON[m["season"]["season_name"]]


def shot_row(e: dict, match_id: int) -> dict:
    s = e["shot"]
    x, y = e["location"][:2]
    dx, dy = 120 - x, 40 - y
    # angle the goal mouth (posts at y = 36 and 44) subtends from the shot location
    angle = abs(math.atan2(44 - y, dx) - math.atan2(36 - y, dx))
    ff = s.get("freeze_frame") or []
    in_cone = 0
    gk_dist = None
    for p in ff:
        px, py = p["location"][:2]
        if not p["teammate"] and p["position"]["name"] == "Goalkeeper":
            gk_dist = math.hypot(120 - px, 40 - py)
        elif not p["teammate"] and _in_triangle(px, py, x, y):
            in_cone += 1
    return {
        "match_id": match_id, "team": e["team"]["name"], "player": e["player"]["name"],
        "period": e["period"], "minute": e["minute"], "x": x, "y": y,
        "distance": math.hypot(dx, dy), "angle": angle,
        "body_part": s["body_part"]["name"], "technique": s["technique"]["name"],
        "shot_type": s["type"]["name"], "play_pattern": e["play_pattern"]["name"],
        "first_time": s.get("first_time", False), "under_pressure": e.get("under_pressure", False),
        "defenders_in_cone": in_cone if ff else None, "gk_distance": gk_dist,
        "xg_statsbomb": s["statsbomb_xg"], "outcome": s["outcome"]["name"],
        "goal": s["outcome"]["name"] == "Goal",
    }


def _in_triangle(px, py, sx, sy) -> bool:
    """Is (px, py) inside the triangle formed by the shooter and the two posts?"""
    pts = [(sx, sy), (120, 36), (120, 44)]
    def sign(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])
    d1, d2, d3 = (sign((px, py), pts[i], pts[(i + 1) % 3]) for i in range(3))
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def team_metrics(events: list, teams: list) -> dict:
    keys = ["passes", "passes_completed", "passes_own_60", "passes_final_third", "shots", "xg", "goals",
            "xg_counter", "big_chances", "offsides", "errors", "def_actions_opp_60", "pressures_final_third",
            "high_recoveries", "red_cards"]
    c = {t: defaultdict(float, dict.fromkeys(keys, 0.0)) for t in teams}
    def_x = {t: [] for t in teams}
    for e in events:
        team = e["team"]["name"]
        typ = e["type"]["name"]
        x = e["location"][0] if e.get("location") else None
        t = c[team]
        if typ == "Pass":
            t["passes"] += 1
            if "outcome" not in e["pass"]:
                t["passes_completed"] += 1
            if x < 72:                       # pass in own defensive 60%: the denominator of opponent PPDA
                t["passes_own_60"] += 1
            if x >= 80:
                t["passes_final_third"] += 1
        elif typ == "Shot":
            t["shots"] += 1
            t["xg"] += e["shot"]["statsbomb_xg"]
            t["goals"] += e["shot"]["outcome"]["name"] == "Goal"
            if e["play_pattern"]["name"] == "From Counter":
                t["xg_counter"] += e["shot"]["statsbomb_xg"]
            if e["shot"]["statsbomb_xg"] >= 0.3:
                t["big_chances"] += 1
        elif typ == "Own Goal For":
            t["goals"] += 1
        elif typ == "Offside":
            t["offsides"] += 1
        elif typ == "Error":
            t["errors"] += 1
        if typ == "Pass" and e["pass"].get("outcome", {}).get("name") == "Pass Offside":
            t["offsides"] += 1
        if typ in DEF_ACTIONS and x is not None:
            def_x[team].append(x)
            if x >= 48 and (typ in PPDA_DEF_ACTIONS or (typ == "Duel" and e["duel"]["type"]["name"] == "Tackle")):
                t["def_actions_opp_60"] += 1
            if typ == "Pressure" and x >= 80:
                t["pressures_final_third"] += 1
            if typ in ("Ball Recovery", "Interception") and x >= 80:
                t["high_recoveries"] += 1
        for k in ("foul_committed", "bad_behaviour"):
            card = (e.get(k) or {}).get("card", {}).get("name", "")
            if card in ("Red Card", "Second Yellow"):
                t["red_cards"] += 1
    rows = {}
    for team in teams:
        opp = next(o for o in teams if o != team)
        t, o = c[team], c[opp]
        rows[team] = {
            **t,
            "ppda": o["passes_own_60"] / t["def_actions_opp_60"] if t["def_actions_opp_60"] else None,
            "def_action_height": sum(def_x[team]) / len(def_x[team]) if def_x[team] else None,
            "pass_share": t["passes"] / (t["passes"] + o["passes"]),
            "field_tilt": t["passes_final_third"] / ((t["passes_final_third"] + o["passes_final_third"]) or 1),
        }
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    match_rows, tm_rows, shot_rows = [], [], []
    for s in SEASONS:
        for m in get_json(f"matches/{LA_LIGA}/{s}.json"):
            mid = m["match_id"]
            home, away = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
            events = get_json(f"events/{mid}.json")
            shot_rows += [shot_row(e, mid) for e in events if e["type"]["name"] == "Shot"]
            if BARCA not in (home, away):
                continue
            season = m["season"]["season_name"]
            barca_home = home == BARCA
            opponent = away if barca_home else home
            match_rows.append({
                "match_id": mid, "season": season, "date": m["match_date"],
                "home": home, "away": away, "barca_home": barca_home, "opponent": opponent,
                "barca_goals": m["home_score"] if barca_home else m["away_score"],
                "opp_goals": m["away_score"] if barca_home else m["home_score"],
                "manager": barca_manager(m, barca_home), "elite": is_elite(opponent, season),
            })
            for team, row in team_metrics(events, [home, away]).items():
                tm_rows.append({"match_id": mid, "team": team, "is_barca": team == BARCA, **row})
    matches = pd.DataFrame(match_rows)
    matches["result"] = (matches.barca_goals - matches.opp_goals).apply(lambda d: "W" if d > 0 else "D" if d == 0 else "L")
    matches.to_parquet(OUT / "matches.parquet", index=False)
    pd.DataFrame(tm_rows).to_parquet(OUT / "team_match.parquet", index=False)
    pd.DataFrame(shot_rows).to_parquet(OUT / "shots.parquet", index=False)
    print(f"matches {len(matches)}, team_match {len(tm_rows)}, shots {len(shot_rows)}")


if __name__ == "__main__":
    main()
