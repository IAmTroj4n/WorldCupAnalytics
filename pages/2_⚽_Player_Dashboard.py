from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from models.player_clustering import build_player_stats, run_player_clustering
from utils.data_loader import check_player_data_availability, load_all_player_data
from utils.helpers import apply_custom_css, render_missing_data_message, render_top_navbar

st.set_page_config(page_title="Player Performance Dashboard", layout="wide")
apply_custom_css()
render_top_navbar()

st.title("Player Performance Dashboard")
st.caption("Interactive player analytics, clustering, and side-by-side comparisons.")

ok, missing = check_player_data_availability()
if not ok:
    render_missing_data_message("Player Dashboard data is unavailable", missing)
    st.stop()

with st.spinner("Loading player data (first visit can take a minute)..."):
    data = load_all_player_data()
    player_stats = build_player_stats(
        data["players"], data["appearances"], data["games"], data["clubs"], data["competitions"]
    )

st.sidebar.header("Filters")
competition_options = sorted(player_stats["competition_name"].dropna().astype(str).unique().tolist()) if "competition_name" in player_stats.columns else []
season_options = sorted(player_stats["season"].dropna().astype(str).unique().tolist()) if "season" in player_stats.columns else []
position_options = sorted(player_stats["position"].dropna().astype(str).unique().tolist()) if "position" in player_stats.columns else []

comp = st.sidebar.selectbox("Competition", ["All"] + competition_options, index=0)
season = st.sidebar.selectbox("Season", ["All"] + season_options, index=0)
positions = st.sidebar.multiselect("Positions", position_options, default=position_options[: min(4, len(position_options))])
min_minutes = st.sidebar.slider("Minimum minutes", 0, int(max(100, player_stats["minutes_played"].max())), 600, 30)
k_clusters = st.sidebar.slider("Number of clusters (k)", 2, 10, 4)

filtered = player_stats.copy()
if comp != "All":
    filtered = filtered[filtered["competition_name"].astype(str) == comp]
if season != "All":
    filtered = filtered[filtered["season"].astype(str) == season]
if positions:
    filtered = filtered[filtered["position"].astype(str).isin(positions)]
filtered = filtered[filtered["minutes_played"] >= min_minutes]

if filtered.empty:
    st.warning("No players match current filters. Try relaxing filters.")
    st.stop()

with st.spinner("Clustering players..."):
    clustered = run_player_clustering(filtered, k_clusters)
df = clustered.clustered

RADAR_STATS = [
    ("goals_per_90", "Goals / 90"),
    ("assists_per_90", "Assists / 90"),
    ("ga_per_90", "G+A / 90"),
    ("minutes_per_game", "Minutes / Game"),
    ("cards_per_90", "Cards / 90"),
]


def _percentile(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float((s <= value).mean() * 100.0)


def player_radar_percentiles(frame: pd.DataFrame, player_row: pd.Series) -> tuple[list[str], list[float]]:
    labels, values = [], []
    for col, label in RADAR_STATS:
        labels.append(label)
        values.append(_percentile(frame[col], float(player_row.get(col, 0) or 0)))
    return labels, values


tab1, tab2, tab3, tab4 = st.tabs(["Player Search & Profile", "Visualizations", "Player Clustering", "Comparisons"])

with tab1:
    choices = [p for p in df["player_name"].dropna().astype(str).tolist() if p.strip()]
    choices = sorted(pd.Series(choices).unique().tolist())
    default_player = choices[0] if choices else None
    selected = st.selectbox("Select player", choices, index=0) if choices else default_player
    if selected:
        p = df[df["player_name"] == selected].sort_values("minutes_played", ascending=False).head(1)
        if not p.empty:
            p = p.iloc[0]
            # Persist selection so other tabs can react (e.g., cluster-filtered charts).
            try:
                st.session_state["selected_player_name"] = str(p.get("player_name", selected))
                st.session_state["selected_player_cluster"] = int(p.get("cluster"))
            except Exception:
                st.session_state["selected_player_name"] = str(selected)
                st.session_state["selected_player_cluster"] = None
            st.markdown(
                f"""
                <div class="fifa-card">
                    <h3>{p.get("player_name","Unknown")}</h3>
                    <p>{p.get("position","Unknown")} | {p.get("nationality","Unknown")} | Age: {p.get("age","N/A")}</p>
                    <p>Club: {p.get("club_name","Unknown")} | Market Value: €{p.get("market_value",0):,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            radar_labels, radar_vals = player_radar_percentiles(df, p)
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(r=radar_vals, theta=radar_labels, fill="toself", name=selected))
            radar.update_layout(
                title="Player Radar (Percentiles within current filters)",
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%", showline=False)),
                showlegend=False,
            )
            st.plotly_chart(radar, use_container_width=True)

            timeline = df[df["player_name"] == selected].sort_values("season")
            if "season" in timeline.columns:
                tfig = px.line(timeline, x="season", y=["goals", "assists"], markers=True, title="Goals/Assists Timeline")
                st.plotly_chart(tfig, use_container_width=True)

with tab2:
    selected_cluster = st.session_state.get("selected_player_cluster", None)
    selected_player_name = st.session_state.get("selected_player_name", None)
    cluster_only = st.toggle(
        "Show only selected player's cluster",
        value=True if selected_cluster is not None else False,
        help="Filters charts to players in the same cluster as the selected player (Tab 1).",
    )
    view = df
    if cluster_only and selected_cluster is not None and "cluster" in df.columns:
        view = df[df["cluster"] == selected_cluster]
        if selected_player_name:
            st.caption(f"Viewing cluster {selected_cluster} (selected: {selected_player_name}).")

    scatter = px.scatter(
        view,
        x="goals",
        y="assists",
        color="position" if "position" in view.columns else None,
        size="market_value",
        hover_data=["player_name", "club_name", "minutes_played", "ga_per_90"],
        title="Goals vs Assists (size=market value)",
    )
    st.plotly_chart(scatter, use_container_width=True)

    heat_cols = ["goals", "assists", "minutes_played", "games_played", "goals_per_90", "assists_per_90", "cards_per_90", "market_value"]
    corr = df[heat_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    dist = px.histogram(df, x="goals", nbins=30, marginal="box", title="Goals Distribution")
    st.plotly_chart(dist, use_container_width=True)

with tab3:
    st.plotly_chart(px.line(clustered.elbow, x="k", y="inertia", markers=True, title="Elbow Method"), use_container_width=True)
    st.plotly_chart(px.line(clustered.silhouette, x="k", y="silhouette", markers=True, title="Silhouette Score"), use_container_width=True)
    st.plotly_chart(
        px.scatter(df, x="pc1", y="pc2", color=df["cluster"].astype(str), hover_data=["player_name", "position", "goals_per_90"], title="2D PCA Clusters"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.scatter_3d(df, x="pc1", y="pc2", z="pc3", color=df["cluster"].astype(str), hover_data=["player_name", "position"], title="3D PCA Clusters"),
        use_container_width=True,
    )
    st.dataframe(clustered.cluster_summary, use_container_width=True)
    comp = df.groupby(["cluster", "position"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(comp, x="cluster", y="count", color="position", barmode="stack", title="Cluster Position Composition"), use_container_width=True)

with tab4:
    players_for_compare = st.multiselect("Select 2-5 players", sorted(df["player_name"].dropna().unique().tolist()), max_selections=5)
    if len(players_for_compare) >= 2:
        compare = df[df["player_name"].isin(players_for_compare)].copy()
        fig = go.Figure()
        for pname in players_for_compare:
            row = compare[compare["player_name"] == pname].sort_values("minutes_played", ascending=False).head(1)
            if row.empty:
                continue
            radar_labels, vals = player_radar_percentiles(df, row.iloc[0])
            fig.add_trace(go.Scatterpolar(r=vals, theta=radar_labels, fill="toself", name=pname))
        fig.update_layout(
            title="Player Comparison Radar (Percentiles within current filters)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%", showline=False)),
        )
        st.plotly_chart(fig, use_container_width=True)
        show_cols = ["player_name", "position", "club_name", "goals", "assists", "goals_per_90", "assists_per_90", "ga_per_90", "cards_per_90", "minutes_played", "market_value"]
        st.dataframe(compare[show_cols].sort_values("ga_per_90", ascending=False), use_container_width=True)
    else:
        st.info("Select at least 2 players to compare.")
