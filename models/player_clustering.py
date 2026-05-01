from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class PlayerClusteringResult:
    player_stats: pd.DataFrame
    clustered: pd.DataFrame
    elbow: pd.DataFrame
    silhouette: pd.DataFrame
    cluster_summary: pd.DataFrame
    pca_columns: List[str]


def _safe_col(df: pd.DataFrame, options: List[str], default: str) -> str:
    for c in options:
        if c in df.columns:
            return c
    return default


@st.cache_data(show_spinner=False)
def build_player_stats(players: pd.DataFrame, appearances: pd.DataFrame, games: pd.DataFrame, clubs: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    player_id_col = _safe_col(players, ["player_id", "id"], "player_id")
    app_player_col = _safe_col(appearances, ["player_id"], "player_id")
    game_id_col = _safe_col(appearances, ["game_id"], "game_id")
    goals_col = _safe_col(appearances, ["goals"], "goals")
    assists_col = _safe_col(appearances, ["assists"], "assists")
    yellow_col = _safe_col(appearances, ["yellow_cards"], "yellow_cards")
    red_col = _safe_col(appearances, ["red_cards"], "red_cards")
    minutes_col = _safe_col(appearances, ["minutes_played", "minutes"], "minutes_played")

    game_lookup_cols = [c for c in ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id"] if c in games.columns]
    game_lookup = games[game_lookup_cols].copy() if game_lookup_cols else pd.DataFrame()
    merged = appearances.merge(game_lookup, left_on=game_id_col, right_on="game_id", how="left") if not game_lookup.empty else appearances.copy()
    if "competition_id" in merged.columns and "competition_id" in competitions.columns:
        comp_cols = [c for c in ["competition_id", "name"] if c in competitions.columns]
        merged = merged.merge(competitions[comp_cols], on="competition_id", how="left")

    # Ensure required numeric columns exist even when upstream schema differs.
    for c in [goals_col, assists_col, yellow_col, red_col, minutes_col]:
        if c not in merged.columns:
            merged[c] = 0
    if game_id_col not in merged.columns:
        merged[game_id_col] = pd.NA

    agg_cols = {
        goals_col: "sum",
        assists_col: "sum",
        yellow_col: "sum",
        red_col: "sum",
        minutes_col: "sum",
        game_id_col: "nunique",
    }
    group_cols = [app_player_col]
    if "season" in merged.columns:
        group_cols.append("season")
    if "name" in merged.columns:
        group_cols.append("name")
    pstats = merged.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()
    pstats = pstats.rename(
        columns={
            app_player_col: "player_id",
            goals_col: "goals",
            assists_col: "assists",
            yellow_col: "yellow_cards",
            red_col: "red_cards",
            minutes_col: "minutes_played",
            game_id_col: "games_played",
            "name": "competition_name",
        }
    )

    meta_cols = [c for c in [player_id_col, "name", "first_name", "last_name", "current_club_id", "position", "country_of_citizenship", "market_value_in_eur", "date_of_birth"] if c in players.columns]
    player_meta = players[meta_cols].copy()
    player_meta = player_meta.rename(columns={player_id_col: "player_id", "name": "player_name", "country_of_citizenship": "nationality", "market_value_in_eur": "market_value"})
    if "player_name" not in player_meta.columns:
        player_meta["player_name"] = ""
    if "first_name" in player_meta.columns or "last_name" in player_meta.columns:
        first = player_meta.get("first_name", "").fillna("").astype(str).str.strip()
        last = player_meta.get("last_name", "").fillna("").astype(str).str.strip()
        full = (first + " " + last).str.strip()
        player_meta["player_name"] = player_meta["player_name"].fillna("").astype(str).str.strip()
        player_meta["player_name"] = player_meta["player_name"].where(player_meta["player_name"] != "", full)
    player_meta["player_name"] = player_meta["player_name"].fillna("").astype(str)
    if "date_of_birth" in player_meta.columns:
        player_meta["date_of_birth"] = pd.to_datetime(player_meta["date_of_birth"], errors="coerce")
        player_meta["age"] = ((pd.Timestamp.today() - player_meta["date_of_birth"]).dt.days / 365.25).round(1)

    if "club_id" in clubs.columns:
        club_cols = [c for c in ["club_id", "name"] if c in clubs.columns]
        club_df = clubs[club_cols].rename(columns={"name": "club_name"})
        player_meta = player_meta.merge(club_df, left_on="current_club_id", right_on="club_id", how="left")

    final = pstats.merge(player_meta, on="player_id", how="left")
    final["goals_per_90"] = np.where(final["minutes_played"] > 0, final["goals"] * 90 / final["minutes_played"], 0)
    final["assists_per_90"] = np.where(final["minutes_played"] > 0, final["assists"] * 90 / final["minutes_played"], 0)
    final["ga_per_90"] = final["goals_per_90"] + final["assists_per_90"]
    final["cards_per_90"] = np.where(final["minutes_played"] > 0, (final["yellow_cards"] + final["red_cards"] * 2) * 90 / final["minutes_played"], 0)
    final["minutes_per_game"] = np.where(final["games_played"] > 0, final["minutes_played"] / final["games_played"], 0)
    final["market_value"] = final.get("market_value", 0).fillna(0)
    final["position"] = final.get("position", "Unknown").fillna("Unknown")
    final["player_name"] = final.get("player_name", "").fillna("").astype(str).str.strip()
    if "player_id" in final.columns:
        missing_names = final["player_name"] == ""
        final.loc[missing_names, "player_name"] = "Player #" + final.loc[missing_names, "player_id"].astype(str)
    return final


@st.cache_data(show_spinner=False)
def elbow_and_silhouette(df: pd.DataFrame, features: List[str], k_min: int = 2, k_max: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[features].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    elbow_rows, sil_rows = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        elbow_rows.append({"k": k, "inertia": float(km.inertia_)})
        if len(np.unique(labels)) > 1:
            sil_rows.append({"k": k, "silhouette": float(silhouette_score(Xs, labels))})
        else:
            sil_rows.append({"k": k, "silhouette": np.nan})
    return pd.DataFrame(elbow_rows), pd.DataFrame(sil_rows)


def _cluster_label(row: pd.Series) -> str:
    if row["goals_per_90"] > 0.45:
        return "Elite Goalscorers"
    if row["assists_per_90"] > 0.25:
        return "Creative Playmakers"
    if row["cards_per_90"] > 0.35:
        return "Defensive Anchors"
    return "Squad Rotators"


@st.cache_data(show_spinner=False)
def run_player_clustering(player_df: pd.DataFrame, k: int) -> PlayerClusteringResult:
    features = ["goals_per_90", "assists_per_90", "cards_per_90", "minutes_per_game", "market_value"]
    work = player_df.copy()
    X = work[features].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    work["cluster"] = km.fit_predict(Xs)

    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(Xs)
    work["pc1"], work["pc2"], work["pc3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]
    summary = work.groupby("cluster")[features].mean().reset_index()
    summary["style_label"] = summary.apply(_cluster_label, axis=1)

    elbow, sil = elbow_and_silhouette(work, features, 2, min(10, max(2, len(work) - 1)))
    return PlayerClusteringResult(
        player_stats=player_df,
        clustered=work,
        elbow=elbow,
        silhouette=sil,
        cluster_summary=summary,
        pca_columns=["pc1", "pc2", "pc3"],
    )
