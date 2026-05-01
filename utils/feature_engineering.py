from __future__ import annotations

import numpy as np
import pandas as pd


TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.85,
    "Copa America": 0.85,
    "African Cup of Nations": 0.8,
    "AFC Asian Cup": 0.8,
    "CONCACAF Gold Cup": 0.78,
    "UEFA Nations League": 0.75,
    "Friendly": 0.3,
}


def _result_code(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _team_match_long(results: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "date": results["date"],
            "team": results["home_team"],
            "opponent": results["away_team"],
            "is_home": 1,
            "goals_for": results["home_score"],
            "goals_against": results["away_score"],
            "neutral": results["neutral"].astype(int),
        }
    )
    away = pd.DataFrame(
        {
            "date": results["date"],
            "team": results["away_team"],
            "opponent": results["home_team"],
            "is_home": 0,
            "goals_for": results["away_score"],
            "goals_against": results["home_score"],
            "neutral": results["neutral"].astype(int),
        }
    )
    long_df = pd.concat([home, away], ignore_index=True).sort_values("date")
    long_df["win"] = (long_df["goals_for"] > long_df["goals_against"]).astype(int)
    long_df["draw"] = (long_df["goals_for"] == long_df["goals_against"]).astype(int)
    long_df["loss"] = (long_df["goals_for"] < long_df["goals_against"]).astype(int)
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    long_df["points"] = long_df["win"] * 3 + long_df["draw"]
    return long_df


def _rolling_team_features(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.sort_values(["team", "date"]).copy()
    grouped = team_df.groupby("team", group_keys=False)
    for window in [5, 10, 20, 50]:
        team_df[f"win_rate_{window}"] = grouped["win"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        team_df[f"draw_rate_{window}"] = grouped["draw"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        team_df[f"goals_for_avg_{window}"] = grouped["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        team_df[f"goals_against_avg_{window}"] = grouped["goals_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        team_df[f"goal_diff_avg_{window}"] = grouped["goal_diff"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        team_df[f"goals_total_avg_{window}"] = grouped.apply(
            lambda g: (g["goals_for"] + g["goals_against"]).shift(1).rolling(window, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
        team_df[f"goal_diff_std_{window}"] = grouped["goal_diff"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
        team_df[f"goals_for_std_{window}"] = grouped["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
    team_df["form_weighted_5"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).apply(
            lambda arr: np.average(arr, weights=np.linspace(1, 2, len(arr))) if len(arr) else np.nan,
            raw=True,
        )
    )
    team_df["home_win_rate_20"] = grouped.apply(
        lambda g: (g["win"] * g["is_home"]).shift(1).rolling(20, min_periods=1).sum()
        / g["is_home"].shift(1).rolling(20, min_periods=1).sum().replace(0, np.nan)
    ).reset_index(level=0, drop=True)
    return team_df


def _elo_like_ratings(results: pd.DataFrame, k: float = 24.0, base: float = 1500.0) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    rows = []
    for r in results.sort_values("date").itertuples(index=False):
        home = getattr(r, "home_team")
        away = getattr(r, "away_team")
        hs = float(getattr(r, "home_score"))
        a_s = float(getattr(r, "away_score"))
        rh = ratings.get(home, base)
        ra = ratings.get(away, base)
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))
        score_h = 1.0 if hs > a_s else 0.0 if hs < a_s else 0.5
        score_a = 1 - score_h
        rh_new = rh + k * (score_h - exp_h)
        ra_new = ra + k * (score_a - (1 - exp_h))
        rows.append({"date": getattr(r, "date"), "home_team": home, "away_team": away, "home_elo": rh, "away_elo": ra})
        ratings[home] = rh_new
        ratings[away] = ra_new
    return pd.DataFrame(rows)


def build_match_features(results: pd.DataFrame) -> pd.DataFrame:
    df = results.sort_values("date").copy()
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["H", "A"],
        default="D",
    )
    team_long = _team_match_long(df)
    team_feat = _rolling_team_features(team_long)
    elo_df = _elo_like_ratings(df)
    match_df = df.merge(elo_df, on=["date", "home_team", "away_team"], how="left")

    home_feat = team_feat.rename(columns={"team": "home_team"}).drop(columns=["opponent"])
    away_feat = team_feat.rename(columns={"team": "away_team"}).drop(columns=["opponent"])
    home_cols = ["date", "home_team"] + [c for c in home_feat.columns if c not in {"date", "home_team", "is_home", "neutral"}]
    away_cols = ["date", "away_team"] + [c for c in away_feat.columns if c not in {"date", "away_team", "is_home", "neutral"}]
    match_df = match_df.merge(home_feat[home_cols], on=["date", "home_team"], how="left", suffixes=("", "_home"))
    match_df = match_df.merge(away_feat[away_cols], on=["date", "away_team"], how="left", suffixes=("_home", "_away"))

    pair_key = np.where(
        match_df["home_team"] < match_df["away_team"],
        match_df["home_team"] + "__" + match_df["away_team"],
        match_df["away_team"] + "__" + match_df["home_team"],
    )
    match_df["pair_key"] = pair_key
    # Rolling windows in pandas require numeric series; build explicit indicators first.
    match_df["result_home_win"] = (match_df["result"] == "H").astype(int)
    match_df["result_draw"] = (match_df["result"] == "D").astype(int)
    match_df["result_away_win"] = (match_df["result"] == "A").astype(int)
    grouped_pair = match_df.groupby("pair_key", group_keys=False)
    match_df["h2h_home_win_rate"] = grouped_pair["result_home_win"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    )
    match_df["h2h_draw_rate"] = grouped_pair["result_draw"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    )
    match_df["h2h_away_win_rate"] = grouped_pair["result_away_win"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    )

    match_df["tournament_weight"] = match_df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(0.55)
    match_df["neutral"] = match_df["neutral"].astype(int)
    match_df["elo_diff"] = match_df["home_elo"] - match_df["away_elo"]
    match_df["elo_abs_diff"] = (match_df["elo_diff"]).abs()
    match_df["elo_diff_bucket"] = pd.cut(
        match_df["elo_abs_diff"],
        bins=[-0.1, 25, 50, 100, 200, 10_000],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # Parity features: how close the teams are in recent form/strength.
    def _safe_absdiff(a: str, b: str, out: str) -> None:
        if a in match_df.columns and b in match_df.columns:
            match_df[out] = (match_df[a] - match_df[b]).abs()

    _safe_absdiff("win_rate_10_home", "win_rate_10_away", "parity_win_rate_10")
    _safe_absdiff("draw_rate_10_home", "draw_rate_10_away", "parity_draw_rate_10")
    _safe_absdiff("goal_diff_avg_10_home", "goal_diff_avg_10_away", "parity_goal_diff_avg_10")
    _safe_absdiff("goals_total_avg_10_home", "goals_total_avg_10_away", "parity_goals_total_avg_10")
    _safe_absdiff("home_elo", "away_elo", "parity_elo")

    match_df["target"] = match_df["result"]

    numeric_cols = match_df.select_dtypes(include=["number"]).columns
    match_df[numeric_cols] = match_df[numeric_cols].fillna(match_df[numeric_cols].median())
    return match_df


def extract_form(results: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    team_df = results[(results["home_team"] == team) | (results["away_team"] == team)].sort_values("date", ascending=False).head(n).copy()
    if team_df.empty:
        return team_df
    team_df["team_result"] = team_df.apply(
        lambda r: "W"
        if (r["home_team"] == team and r["home_score"] > r["away_score"]) or (r["away_team"] == team and r["away_score"] > r["home_score"])
        else "L"
        if (r["home_team"] == team and r["home_score"] < r["away_score"]) or (r["away_team"] == team and r["away_score"] < r["home_score"])
        else "D",
        axis=1,
    )
    return team_df[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "team_result"]]


def h2h_history(results: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    h2h = results[
        ((results["home_team"] == home_team) & (results["away_team"] == away_team))
        | ((results["home_team"] == away_team) & (results["away_team"] == home_team))
    ].sort_values("date", ascending=False)
    return h2h
