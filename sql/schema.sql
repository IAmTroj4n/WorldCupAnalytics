CREATE INDEX IF NOT EXISTS idx_results_date ON results(date);
CREATE INDEX IF NOT EXISTS idx_results_home_team ON results(home_team);
CREATE INDEX IF NOT EXISTS idx_results_away_team ON results(away_team);
CREATE INDEX IF NOT EXISTS idx_results_tournament ON results(tournament);

CREATE INDEX IF NOT EXISTS idx_appearances_player ON appearances(player_id);
CREATE INDEX IF NOT EXISTS idx_appearances_game ON appearances(game_id);
CREATE INDEX IF NOT EXISTS idx_games_competition ON games(competition_id);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);

DROP VIEW IF EXISTS player_season_stats;
CREATE VIEW player_season_stats AS
SELECT
    a.player_id,
    COALESCE(a.player_name, p.name, p.first_name || ' ' || p.last_name) AS player_name,
    g.season,
    c.name AS competition_name,
    p.position,
    p.country_of_citizenship AS nationality,
    cl.name AS club_name,
    SUM(a.goals) AS goals,
    SUM(a.assists) AS assists,
    SUM(a.yellow_cards) AS yellow_cards,
    SUM(a.red_cards) AS red_cards,
    SUM(a.minutes_played) AS minutes_played,
    COUNT(DISTINCT a.game_id) AS games_played,
    ROUND(CASE WHEN SUM(a.minutes_played) > 0 THEN SUM(a.goals) * 90.0 / SUM(a.minutes_played) ELSE 0 END, 3) AS goals_per_90,
    ROUND(CASE WHEN SUM(a.minutes_played) > 0 THEN SUM(a.assists) * 90.0 / SUM(a.minutes_played) ELSE 0 END, 3) AS assists_per_90,
    ROUND(CASE WHEN SUM(a.minutes_played) > 0 THEN (SUM(a.goals) + SUM(a.assists)) * 90.0 / SUM(a.minutes_played) ELSE 0 END, 3) AS ga_per_90
FROM appearances a
JOIN games g ON g.game_id = a.game_id
LEFT JOIN competitions c ON c.competition_id = g.competition_id
LEFT JOIN players p ON p.player_id = a.player_id
LEFT JOIN clubs cl ON cl.club_id = p.current_club_id
GROUP BY
    a.player_id,
    player_name,
    g.season,
    competition_name,
    p.position,
    nationality,
    club_name;

DROP VIEW IF EXISTS international_match_outcomes;
CREATE VIEW international_match_outcomes AS
SELECT
    date,
    home_team,
    away_team,
    home_score,
    away_score,
    tournament,
    city,
    country,
    neutral,
    CASE
        WHEN home_score > away_score THEN 'H'
        WHEN home_score < away_score THEN 'A'
        ELSE 'D'
    END AS result_code,
    CASE
        WHEN home_score > away_score THEN 'Home Win'
        WHEN home_score < away_score THEN 'Away Win'
        ELSE 'Draw'
    END AS result_label
FROM results;
