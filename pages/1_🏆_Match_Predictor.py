from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from models.match_predictor import build_fixture_feature_row, predict_fixture
from utils.data_loader import check_match_data_availability, load_all_match_data
from utils.feature_engineering import extract_form, h2h_history
from utils.helpers import apply_custom_css, render_missing_data_message, render_top_navbar
from utils.model_registry import load_latest_model_bundle

st.set_page_config(page_title="FIFA 2026 Match Predictor", layout="wide")
apply_custom_css()
render_top_navbar()

WC_2026_TEAMS = [
    "United States",
    "Mexico",
    "Canada",
    "Argentina",
    "Brazil",
    "France",
    "England",
    "Spain",
    "Germany",
    "Portugal",
    "Netherlands",
    "Belgium",
    "Croatia",
    "Italy",
    "Uruguay",
    "Colombia",
    "Japan",
    "South Korea",
    "Australia",
    "Iran",
    "Saudi Arabia",
    "Qatar",
    "Morocco",
    "Senegal",
    "Tunisia",
    "Algeria",
    "Egypt",
    "Nigeria",
    "Ghana",
    "Cameroon",
    "Ecuador",
    "Chile",
    "Peru",
    "Paraguay",
    "Poland",
    "Switzerland",
    "Serbia",
    "Denmark",
    "Sweden",
    "Norway",
    "Turkey",
    "Ukraine",
    "Austria",
    "Czech Republic",
    "Wales",
    "Scotland",
    "Costa Rica",
    "Panama",
]


def outcome_label(code: str) -> str:
    return {"H": "Home Win", "D": "Draw", "A": "Away Win"}.get(code, code)


st.caption("Predict Win / Draw / Loss with historical international football data.")

ok, missing = check_match_data_availability()
if not ok:
    render_missing_data_message("Match Predictor data is unavailable", missing)
    st.stop()

data = load_all_match_data()
results = data["results"]
teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
tournaments = ["FIFA World Cup"]
default_tournament = "FIFA World Cup"

# Inference-only mode (no training on page load).
try:
    bundle, model_info = load_latest_model_bundle()
except FileNotFoundError:
    st.error(
        "No trained model artifact found. Run `python -m pipelines.train_match_model --model \"XGBoost\"` first."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=max(0, teams.index("Argentina")) if "Argentina" in teams else 0)
with col2:
    away_team = st.selectbox("Away Team", teams, index=max(0, teams.index("France")) if "France" in teams else 1)

c1, c2 = st.columns([2, 1])
with c1:
    tournament = st.selectbox("Tournament", tournaments, index=0)
with c2:
    neutral = st.toggle("Neutral Venue", value=True)

if home_team == away_team:
    st.warning("Please pick two different teams.")
    st.stop()

if st.button("Predict", type="primary", use_container_width=True):
    row = build_fixture_feature_row(bundle.frame, home_team, away_team, tournament, neutral)
    pred, prob_map = predict_fixture(bundle, row)
    confidence = max(prob_map.values()) * 100

    st.markdown(
        f"""
        <div class="fifa-card prediction-card">
            <h2>{home_team} vs {away_team}</h2>
            <p class="prediction-main">{outcome_label(pred)}</p>
            <p class="prediction-sub">Confidence: {confidence:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob_df = pd.DataFrame(
        {
            "Outcome": [outcome_label(x) for x in ["H", "D", "A"]],
            "Probability": [prob_map.get("H", 0), prob_map.get("D", 0), prob_map.get("A", 0)],
        }
    )
    fig_prob = px.bar(prob_df, x="Outcome", y="Probability", color="Outcome", title="Outcome Probabilities", text_auto=".1%")
    fig_prob.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_prob, use_container_width=True)

h2h = h2h_history(results, home_team, away_team)
st.subheader("Head-to-Head History")
if h2h.empty:
    st.info("No previous meetings found between these teams.")
else:
    h2h_show = h2h[["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]].head(20)
    st.dataframe(h2h_show, use_container_width=True)
    h2h["outcome"] = np.where(h2h["home_score"] > h2h["away_score"], "Home Team Won", np.where(h2h["home_score"] < h2h["away_score"], "Away Team Won", "Draw"))
    pie = px.pie(h2h, names="outcome", title="H2H W/D/L Split")
    st.plotly_chart(pie, use_container_width=True)

st.subheader("Recent Form")
f1, f2 = st.columns(2)
for team, container in [(home_team, f1), (away_team, f2)]:
    with container:
        st.markdown(f"#### {team}")
        form = extract_form(results, team, n=10)
        if form.empty:
            st.info("No recent matches.")
        else:
            badges = "".join(
                [
                    f'<span class="result-badge {"badge-win" if x=="W" else "badge-loss" if x=="L" else "badge-draw"}">{x}</span>'
                    for x in form["team_result"].tolist()
                ]
            )
            st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
            st.dataframe(form[["date", "home_team", "away_team", "home_score", "away_score", "team_result"]], use_container_width=True)

with st.expander("Model Performance"):
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f'{bundle.metrics["accuracy"]:.3f}')
    c2.metric("F1 Macro", f'{bundle.metrics["f1_macro"]:.3f}')
    md1, md2, md3 = st.columns(3)
    md1.metric("Last Trained (UTC)", model_info["trained_at"])
    md2.metric("Data Last Updated (UTC)", model_info["data_last_updated"])
    md3.metric("Model Version", model_info["version"])
    if "cv_f1_macro_best" in bundle.metrics:
        st.caption(f"Best CV Macro-F1: {bundle.metrics['cv_f1_macro_best']:.3f}")
    if "draw_threshold" in bundle.metrics:
        st.caption(
            f"Draw threshold: {bundle.metrics['draw_threshold']:.2f} "
            f"(macro-F1 with threshold: {bundle.metrics.get('macro_f1_with_draw_threshold', 0):.3f})"
        )
    if "best_params" in bundle.metrics:
        with st.expander("Best XGBoost Params"):
            st.json(bundle.metrics["best_params"])
    cm = pd.DataFrame(bundle.metrics["confusion_matrix"], index=["H", "D", "A"], columns=["H", "D", "A"])
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)
    report = bundle.metrics.get("classification_report", {})
    rows = []
    label_name = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    for code in ["H", "D", "A"]:
        if code in report:
            rows.append(
                {
                    "Class": label_name[code],
                    "Precision": float(report[code].get("precision", 0.0)),
                    "Recall": float(report[code].get("recall", 0.0)),
                    "F1": float(report[code].get("f1-score", 0.0)),
                    "Support": int(report[code].get("support", 0)),
                }
            )
    if rows:
        st.markdown("**Per-class performance (focus: Macro F1 balance)**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
