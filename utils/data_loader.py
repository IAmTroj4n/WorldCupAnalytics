from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from utils.helpers import get_data_dir


MATCH_FILES = ["results.csv", "goalscorers.csv", "shootouts.csv"]
PLAYER_FILES = ["players.csv", "appearances.csv", "games.csv", "clubs.csv", "competitions.csv"]


def _validate_files(required_files: List[str], data_dir: Path) -> Tuple[bool, List[str]]:
    missing = [f for f in required_files if not (data_dir / f).exists()]
    return len(missing) == 0, missing


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    path = get_data_dir() / "results.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["neutral"] = df["neutral"].astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_goalscorers() -> pd.DataFrame:
    path = get_data_dir() / "goalscorers.csv"
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_shootouts() -> pd.DataFrame:
    path = get_data_dir() / "shootouts.csv"
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    return pd.read_csv(get_data_dir() / "players.csv")


@st.cache_data(show_spinner=False)
def load_appearances() -> pd.DataFrame:
    return pd.read_csv(get_data_dir() / "appearances.csv")


@st.cache_data(show_spinner=False)
def load_games() -> pd.DataFrame:
    df = pd.read_csv(get_data_dir() / "games.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_clubs() -> pd.DataFrame:
    return pd.read_csv(get_data_dir() / "clubs.csv")


@st.cache_data(show_spinner=False)
def load_competitions() -> pd.DataFrame:
    return pd.read_csv(get_data_dir() / "competitions.csv")


def check_match_data_availability() -> Tuple[bool, List[str]]:
    return _validate_files(MATCH_FILES, get_data_dir())


def check_player_data_availability() -> Tuple[bool, List[str]]:
    return _validate_files(PLAYER_FILES, get_data_dir())


@st.cache_data(show_spinner=False)
def load_all_match_data() -> Dict[str, pd.DataFrame]:
    return {
        "results": load_results(),
        "goalscorers": load_goalscorers(),
        "shootouts": load_shootouts(),
    }


@st.cache_data(show_spinner=False)
def load_all_player_data() -> Dict[str, pd.DataFrame]:
    return {
        "players": load_players(),
        "appearances": load_appearances(),
        "games": load_games(),
        "clubs": load_clubs(),
        "competitions": load_competitions(),
    }
