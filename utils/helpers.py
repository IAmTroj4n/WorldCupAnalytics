from __future__ import annotations

from pathlib import Path
from typing import Iterable

import streamlit as st


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_artifacts_dir() -> Path:
    return get_project_root() / "artifacts"


def apply_custom_css() -> None:
    css_path = get_project_root() / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_top_navbar() -> None:
    st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
    row = st.columns([2.2, 1, 1.3, 1.35], gap="small")
    with row[0]:
        st.markdown('<div class="brand">FIFA 2026 Predictor</div>', unsafe_allow_html=True)
    with row[1]:
        st.page_link("app.py", label="Home")
    with row[2]:
        st.page_link("pages/1_🏆_Match_Predictor.py", label="Match Predictor")
    with row[3]:
        st.page_link("pages/2_⚽_Player_Dashboard.py", label="Player Dashboard")
    st.markdown("</div>", unsafe_allow_html=True)


def render_missing_data_message(title: str, missing_files: Iterable[str]) -> None:
    files = "".join([f"<li><code>{name}</code></li>" for name in sorted(missing_files)])
    st.markdown(
        f"""
        <div class="fifa-card fifa-warning">
            <h3>{title}</h3>
            <p>Missing required CSV files in <code>data/</code>:</p>
            <ul>{files}</ul>
            <p>
                Download datasets from:
                <br/>
                <a href="https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017" target="_blank">International Results Dataset</a>
                <br/>
                <a href="https://www.kaggle.com/datasets/davidcariboo/player-scores" target="_blank">Transfermarkt Player Scores Dataset</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_badges(value: str) -> str:
    mapping = {"H": "badge-win", "D": "badge-draw", "A": "badge-loss"}
    cls = mapping.get(value, "badge-draw")
    return f'<span class="result-badge {cls}">{value}</span>'
