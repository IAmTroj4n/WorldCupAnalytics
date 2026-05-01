from __future__ import annotations

import streamlit as st

from utils.data_loader import (
    check_match_data_availability,
    check_player_data_availability,
    load_all_match_data,
    load_all_player_data,
)
from utils.helpers import apply_custom_css, render_top_navbar

st.set_page_config(page_title="FIFA 2026 Analytics Hub", layout="wide")
apply_custom_css()
render_top_navbar()

st.markdown(
    """
    <div class="hero">
        <h1>FIFA World Cup 2026 Analytics Hub</h1>
        <p>Predict match outcomes and explore player performance with interactive data science dashboards.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    st.markdown(
        """
        <div class="fifa-card">
            <h3>Match Predictor</h3>
            <p>Uses the latest pre-trained ML artifact for instant FIFA 2026 outcome predictions.</p>
            <p><b>Navigate:</b> Pages → Match Predictor</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class="fifa-card">
            <h3>Player Dashboard</h3>
            <p>Analyze player stats, compare profiles, and cluster playing styles using KMeans + PCA.</p>
            <p><b>Navigate:</b> Pages → Player Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

mc_ok, _ = check_match_data_availability()
pc_ok, _ = check_player_data_availability()

m1, m2, m3 = st.columns(3)
if mc_ok:
    match_data = load_all_match_data()["results"]
    m1.metric("Total Matches", f"{len(match_data):,}")
    m2.metric("Date Range", f"{match_data['date'].min().date()} → {match_data['date'].max().date()}")
else:
    m1.metric("Total Matches", "Data Missing")
    m2.metric("Date Range", "N/A")

if pc_ok:
    players = load_all_player_data()["players"]
    m3.metric("Total Players", f"{len(players):,}")
else:
    m3.metric("Total Players", "Data Missing")

