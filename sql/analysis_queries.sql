-- name: Top 10 scorers in a season (Premier League example)
SELECT
    player_name,
    season,
    competition_name,
    goals,
    assists,
    minutes_played,
    goals_per_90
FROM player_season_stats
WHERE competition_name = 'premier-league'
  AND season = 2023
  AND minutes_played >= 900
ORDER BY goals DESC, goals_per_90 DESC
LIMIT 10;

-- name: Head-to-head record between two teams
SELECT
    date,
    home_team,
    away_team,
    home_score,
    away_score,
    tournament,
    result_label
FROM international_match_outcomes
WHERE (home_team = 'Argentina' AND away_team = 'France')
   OR (home_team = 'France' AND away_team = 'Argentina')
ORDER BY date DESC;

-- name: Team win rate over last 10 international matches
WITH team_matches AS (
    SELECT
        date,
        home_team AS team,
        CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS win_flag
    FROM results
    WHERE home_team = 'Brazil'
    UNION ALL
    SELECT
        date,
        away_team AS team,
        CASE WHEN away_score > home_score THEN 1 ELSE 0 END AS win_flag
    FROM results
    WHERE away_team = 'Brazil'
),
ranked AS (
    SELECT
        date,
        win_flag,
        ROW_NUMBER() OVER (ORDER BY date DESC) AS rn
    FROM team_matches
)
SELECT
    COUNT(*) AS matches,
    ROUND(AVG(win_flag), 3) AS win_rate
FROM ranked
WHERE rn <= 10;

-- name: Average goals per match by tournament (min 100 games)
SELECT
    tournament,
    COUNT(*) AS matches,
    ROUND(AVG(home_score + away_score), 2) AS avg_total_goals,
    ROUND(AVG(CASE WHEN home_score = away_score THEN 1.0 ELSE 0.0 END), 3) AS draw_rate
FROM results
GROUP BY tournament
HAVING COUNT(*) >= 100
ORDER BY avg_total_goals DESC
LIMIT 15;

-- name: Top assisters with minimum minutes
SELECT
    player_name,
    season,
    competition_name,
    assists,
    goals,
    minutes_played,
    assists_per_90
FROM player_season_stats
WHERE minutes_played >= 1200
ORDER BY assists_per_90 DESC, assists DESC
LIMIT 15;

-- name: FIFA World Cup matches by host country
SELECT
    country,
    COUNT(*) AS matches,
    MIN(date) AS first_match,
    MAX(date) AS last_match
FROM results
WHERE tournament = 'FIFA World Cup'
GROUP BY country
ORDER BY matches DESC
LIMIT 20;

-- name: Monthly international match volume since 2010
SELECT
    strftime('%Y-%m', date) AS month,
    COUNT(*) AS matches
FROM results
WHERE date >= '2010-01-01'
GROUP BY strftime('%Y-%m', date)
ORDER BY month DESC
LIMIT 24;

-- name: Most valuable players with production (market value join)
SELECT
    p.name AS player_name,
    p.position,
    p.country_of_citizenship AS nationality,
    p.market_value_in_eur AS market_value,
    SUM(s.goals) AS total_goals,
    SUM(s.assists) AS total_assists,
    SUM(s.minutes_played) AS total_minutes
FROM players p
JOIN player_season_stats s ON s.player_id = p.player_id
WHERE p.market_value_in_eur IS NOT NULL
GROUP BY
    p.player_id,
    player_name,
    p.position,
    nationality,
    market_value
HAVING total_minutes >= 2000
ORDER BY market_value DESC
LIMIT 20;
