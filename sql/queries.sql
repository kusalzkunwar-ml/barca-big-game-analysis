-- Barça's big-game problem: analysis queries (DuckDB).
-- Run with: python src/run_sql.py   (creates the views below over data/processed/*.parquet)
--
-- Views:
--   matches     one row per Barça La Liga match
--   barca       Barça's team_match row joined to its match
--   opp         the opponent's team_match row joined to the same match

-- name: elite_vs_rest
-- Results and chance quality per game, elite opponents vs everyone else.
SELECT
    CASE WHEN m.elite THEN 'Elite (RMA / ATM)' ELSE 'Rest of La Liga' END AS opposition,
    COUNT(*)                                                     AS games,
    ROUND(AVG(CASE m.result WHEN 'W' THEN 3 WHEN 'D' THEN 1 ELSE 0 END), 2) AS points_per_game,
    ROUND(100 * AVG((m.result = 'W')::INT), 1)                   AS win_pct,
    ROUND(AVG(b.goals), 2)                                       AS goals_for,
    ROUND(AVG(o.goals), 2)                                       AS goals_against,
    ROUND(AVG(b.xg), 2)                                          AS xg_for,
    ROUND(AVG(o.xg), 2)                                          AS xg_against,
    ROUND(AVG(b.xg - o.xg), 2)                                   AS xg_diff
FROM matches m
JOIN barca b USING (match_id)
JOIN opp   o USING (match_id)
GROUP BY ALL
ORDER BY opposition;

-- name: style_vs_elite
-- Does the press and the high line survive against elite sides?
-- Lower PPDA = more intense press. Defensive height = mean x (metres from own goal line, 0-120) of defensive actions.
SELECT
    CASE WHEN m.elite THEN 'Elite (RMA / ATM)' ELSE 'Rest of La Liga' END AS opposition,
    ROUND(AVG(b.ppda), 2)                    AS barca_ppda,
    ROUND(AVG(b.def_action_height), 1)       AS barca_def_height,
    ROUND(100 * AVG(b.pass_share), 1)        AS barca_pass_share_pct,
    ROUND(100 * AVG(b.field_tilt), 1)        AS barca_field_tilt_pct,
    ROUND(AVG(b.high_recoveries), 1)         AS barca_high_recoveries,
    ROUND(AVG(b.pressures_final_third), 1)   AS barca_final_third_pressures,
    ROUND(AVG(o.offsides), 2)                AS opp_offsides_per_game
FROM matches m
JOIN barca b USING (match_id)
JOIN opp   o USING (match_id)
GROUP BY ALL
ORDER BY opposition;

-- name: counter_exposure
-- How much of what Barça concede comes from counter-attacks and errors?
SELECT
    CASE WHEN m.elite THEN 'Elite (RMA / ATM)' ELSE 'Rest of La Liga' END AS opposition,
    ROUND(AVG(o.xg_counter), 3)                              AS opp_counter_xg_per_game,
    ROUND(100 * SUM(o.xg_counter) / SUM(o.xg), 1)            AS counter_share_of_xg_against_pct,
    ROUND(AVG(o.big_chances), 2)                             AS big_chances_conceded,
    ROUND(AVG(b.errors), 2)                                  AS barca_errors_leading_to_shot,
    ROUND(AVG(b.red_cards), 3)                               AS barca_red_cards
FROM matches m
JOIN barca b USING (match_id)
JOIN opp   o USING (match_id)
GROUP BY ALL
ORDER BY opposition;

-- name: by_manager
-- Same split per Barça manager (only managers with at least 4 elite games are shown).
WITH per_game AS (
    SELECT m.manager, m.date, m.elite, m.result, b.xg - o.xg AS xg_diff, b.ppda, b.def_action_height
    FROM matches m
    JOIN barca b USING (match_id)
    JOIN opp   o USING (match_id)
)
SELECT
    manager,
    COUNT(*) FILTER (WHERE elite)                                        AS elite_games,
    ROUND(AVG(xg_diff) FILTER (WHERE NOT elite), 2)                      AS xg_diff_rest,
    ROUND(AVG(xg_diff) FILTER (WHERE elite), 2)                          AS xg_diff_elite,
    ROUND(AVG(xg_diff) FILTER (WHERE elite) - AVG(xg_diff) FILTER (WHERE NOT elite), 2) AS elite_drop,
    ROUND(AVG(ppda) FILTER (WHERE NOT elite), 2)                         AS ppda_rest,
    ROUND(AVG(ppda) FILTER (WHERE elite), 2)                             AS ppda_elite
FROM per_game
GROUP BY manager
HAVING COUNT(*) FILTER (WHERE elite) >= 4
ORDER BY MIN(date);

-- name: season_trend
-- xG difference per game by season, elite vs rest, with the gap between them.
SELECT
    m.season,
    ROUND(AVG(b.xg - o.xg) FILTER (WHERE NOT m.elite), 2)  AS xg_diff_rest,
    ROUND(AVG(b.xg - o.xg) FILTER (WHERE m.elite), 2)      AS xg_diff_elite,
    COUNT(*) FILTER (WHERE m.elite)                        AS elite_games
FROM matches m
JOIN barca b USING (match_id)
JOIN opp   o USING (match_id)
GROUP BY m.season
ORDER BY m.season;

-- name: worst_elite_games
-- The ten elite games where Barça were most out-created, ranked with a window function.
SELECT * FROM (
    SELECT
        m.date, m.season, m.manager,
        CASE WHEN m.barca_home THEN 'vs ' ELSE '@ ' END || m.opponent  AS fixture,
        m.barca_goals || '-' || m.opp_goals                           AS score,
        ROUND(b.xg, 2) AS xg_for, ROUND(o.xg, 2) AS xg_against,
        RANK() OVER (ORDER BY b.xg - o.xg) AS rank_worst
    FROM matches m
    JOIN barca b USING (match_id)
    JOIN opp   o USING (match_id)
    WHERE m.elite
)
WHERE rank_worst <= 10
ORDER BY rank_worst;
