from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils.feature_engineering import build_match_features


MODEL_FEATURES = [
    "neutral",
    "tournament_weight",
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_abs_diff",
    "elo_diff_bucket",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "win_rate_5_home",
    "win_rate_10_home",
    "win_rate_20_home",
    "win_rate_50_home",
    "draw_rate_10_home",
    "draw_rate_20_home",
    "goals_for_avg_10_home",
    "goals_against_avg_10_home",
    "goal_diff_avg_10_home",
    "goals_total_avg_10_home",
    "goal_diff_std_10_home",
    "goals_for_std_10_home",
    "form_weighted_5_home",
    "home_win_rate_20_home",
    "win_rate_5_away",
    "win_rate_10_away",
    "win_rate_20_away",
    "win_rate_50_away",
    "draw_rate_10_away",
    "draw_rate_20_away",
    "goals_for_avg_10_away",
    "goals_against_avg_10_away",
    "goal_diff_avg_10_away",
    "goals_total_avg_10_away",
    "goal_diff_std_10_away",
    "goals_for_std_10_away",
    "form_weighted_5_away",
    "home_win_rate_20_away",
    "parity_win_rate_10",
    "parity_draw_rate_10",
    "parity_goal_diff_avg_10",
    "parity_goals_total_avg_10",
    "parity_elo",
]

LABEL_ORDER = ["H", "D", "A"]
LABEL_TO_INT = {label: idx for idx, label in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {idx: label for label, idx in LABEL_TO_INT.items()}


@dataclass
class MatchModelBundle:
    model_name: str
    estimator: Pipeline
    features: List[str]
    labels: List[str]
    metrics: Dict[str, object]
    frame: pd.DataFrame
    draw_threshold: float | None = None


def _apply_draw_threshold(prob_map: Dict[str, float], threshold: float) -> str:
    if prob_map.get("D", 0.0) >= threshold:
        return "D"
    h = prob_map.get("H", 0.0)
    a = prob_map.get("A", 0.0)
    return "H" if h >= a else "A"


def _tune_draw_threshold(y_true: pd.Series, prob_maps: List[Dict[str, float]]) -> tuple[float, float]:
    thresholds = np.linspace(0.18, 0.42, 25)
    best_t, best_f1 = 0.28, -1.0
    for t in thresholds:
        preds = [_apply_draw_threshold(pm, float(t)) for pm in prob_maps]
        score = float(f1_score(y_true, preds, average="macro"))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, best_f1


def _build_model(name: str):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42)
    if name == "XGBoost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    return LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=None)


def _needs_scaler(name: str) -> bool:
    return name == "Logistic Regression"


def _tune_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    base = RandomForestClassifier(random_state=42, class_weight="balanced")
    params = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [None, 8, 12, 18, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(base, params, n_iter=15, cv=3, scoring="f1_macro", random_state=42, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_


def _compute_balanced_sample_weights(y_int: pd.Series) -> np.ndarray:
    counts = y_int.value_counts().to_dict()
    n_classes = max(1, len(counts))
    n_samples = len(y_int)
    class_weight = {cls: n_samples / (n_classes * cnt) for cls, cnt in counts.items() if cnt > 0}
    return y_int.map(class_weight).astype(float).values


def _tune_xgboost(X: pd.DataFrame, y_int: pd.Series, sample_weight: np.ndarray) -> tuple[XGBClassifier, dict, float]:
    base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    params = {
        "n_estimators": [300, 450, 600, 800, 1000],
        "learning_rate": [0.01, 0.02, 0.04, 0.06, 0.1],
        "max_depth": [2, 3, 4, 5, 6, 7],
        "min_child_weight": [1, 2, 3, 4, 6, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0.0, 0.05, 0.1, 0.2, 0.4],
        "reg_alpha": [0.0, 0.05, 0.1, 0.2, 0.5],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
        "max_delta_step": [0, 1, 2],
        "grow_policy": ["depthwise", "lossguide"],
        "max_leaves": [0, 16, 32, 64, 96],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=params,
        n_iter=28,
        cv=tscv,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y_int, sample_weight=sample_weight)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


@st.cache_resource(show_spinner=True)
def train_match_model(results_df: pd.DataFrame, model_name: str, retrain_flag: int = 0) -> MatchModelBundle:
    _ = retrain_flag
    frame = build_match_features(results_df)
    frame = frame.sort_values("date")
    cutoff_train = pd.Timestamp("2023-01-01")
    cutoff_valid = pd.Timestamp("2025-12-31")
    train = frame[frame["date"] < cutoff_train].copy()
    valid = frame[(frame["date"] >= cutoff_train) & (frame["date"] <= cutoff_valid)].copy()
    if valid.empty:
        valid = frame[frame["date"] >= cutoff_train].copy()
    if train.empty:
        train = frame.iloc[:-max(1, len(frame) // 5)].copy()
        valid = frame.iloc[-max(1, len(frame) // 5) :].copy()

    X_train = train[MODEL_FEATURES]
    y_train = train["target"]
    X_valid = valid[MODEL_FEATURES]
    y_valid = valid["target"]

    estimator = _build_model(model_name)
    if model_name == "Random Forest":
        estimator = _tune_random_forest(X_train, y_train)
    fit_y_train = y_train
    eval_y_valid = y_valid
    fit_sample_weight = None
    if model_name == "XGBoost":
        fit_y_train = y_train.map(LABEL_TO_INT).astype(int)
        fit_sample_weight = _compute_balanced_sample_weights(fit_y_train)
        eval_y_valid = y_valid.map(LABEL_TO_INT).astype(int)
        estimator, best_params, cv_f1 = _tune_xgboost(X_train, fit_y_train, fit_sample_weight)
    steps = []
    if _needs_scaler(model_name):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    if model_name == "XGBoost" and fit_sample_weight is not None:
        pipe.fit(X_train, fit_y_train, model__sample_weight=fit_sample_weight)
    else:
        pipe.fit(X_train, fit_y_train)
    preds = pipe.predict(X_valid)
    if model_name == "XGBoost":
        preds_eval = pd.Series(preds).map(INT_TO_LABEL)
        y_eval = y_valid
    else:
        preds_eval = preds
        y_eval = y_valid

    metrics = {
        "accuracy": float(accuracy_score(y_eval, preds_eval)),
        "f1_macro": float(f1_score(y_eval, preds_eval, average="macro")),
        "confusion_matrix": confusion_matrix(y_eval, preds_eval, labels=LABEL_ORDER),
        "classification_report": classification_report(y_eval, preds_eval, output_dict=True, zero_division=0),
    }
    draw_threshold = None
    if model_name == "XGBoost":
        # Tune a simple draw decision rule on the time-based validation split
        probs = pipe.predict_proba(X_valid)
        classes = pipe.classes_
        mapped = [INT_TO_LABEL.get(int(k), str(k)) if isinstance(k, (int, np.integer)) else str(k) for k in classes]
        prob_maps = [{str(lbl): float(p) for lbl, p in zip(mapped, row)} for row in probs]
        for pm in prob_maps:
            for lbl in LABEL_ORDER:
                pm.setdefault(lbl, 0.0)
        draw_threshold, tuned_macro = _tune_draw_threshold(y_valid, prob_maps)
        tuned_preds = [_apply_draw_threshold(pm, float(draw_threshold)) for pm in prob_maps]
        metrics["draw_threshold"] = float(draw_threshold)
        metrics["macro_f1_with_draw_threshold"] = float(tuned_macro)
        metrics["classification_report_with_draw_threshold"] = classification_report(
            y_valid, tuned_preds, output_dict=True, zero_division=0
        )
        metrics["confusion_matrix_with_draw_threshold"] = confusion_matrix(
            y_valid, tuned_preds, labels=LABEL_ORDER
        )
    if model_name == "XGBoost":
        metrics["cv_f1_macro_best"] = float(cv_f1)
        metrics["best_params"] = best_params
    return MatchModelBundle(
        model_name=model_name,
        estimator=pipe,
        features=MODEL_FEATURES,
        labels=LABEL_ORDER,
        metrics=metrics,
        frame=frame,
        draw_threshold=draw_threshold,
    )


def build_fixture_feature_row(
    frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    tournament: str,
    neutral: bool,
) -> pd.DataFrame:
    home_latest = frame[frame["home_team"] == home_team].sort_values("date").tail(1)
    away_latest = frame[frame["away_team"] == away_team].sort_values("date").tail(1)
    pair_rows = frame[
        ((frame["home_team"] == home_team) & (frame["away_team"] == away_team))
        | ((frame["home_team"] == away_team) & (frame["away_team"] == home_team))
    ].sort_values("date")

    row = {}
    for col in MODEL_FEATURES:
        if col.endswith("_home") and not home_latest.empty:
            row[col] = float(home_latest.iloc[-1][col])
        elif col.endswith("_away") and not away_latest.empty:
            row[col] = float(away_latest.iloc[-1][col])
        else:
            row[col] = float(frame[col].median()) if col in frame.columns else 0.0

    row["neutral"] = int(neutral)
    row["tournament_weight"] = frame["tournament_weight"].median()
    known_tournaments = frame[["tournament", "tournament_weight"]].dropna().drop_duplicates()
    tournament_map = dict(zip(known_tournaments["tournament"], known_tournaments["tournament_weight"]))
    row["tournament_weight"] = float(tournament_map.get(tournament, row["tournament_weight"]))
    if not home_latest.empty:
        row["home_elo"] = float(home_latest.iloc[-1]["home_elo"])
    if not away_latest.empty:
        row["away_elo"] = float(away_latest.iloc[-1]["away_elo"])
    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    if not pair_rows.empty:
        row["h2h_home_win_rate"] = float(pair_rows["h2h_home_win_rate"].iloc[-1])
        row["h2h_draw_rate"] = float(pair_rows["h2h_draw_rate"].iloc[-1])
        row["h2h_away_win_rate"] = float(pair_rows["h2h_away_win_rate"].iloc[-1])
    return pd.DataFrame([row], columns=MODEL_FEATURES)


def predict_fixture(bundle: MatchModelBundle, feature_row: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    probs = bundle.estimator.predict_proba(feature_row)[0]
    labels = bundle.estimator.classes_
    mapped_labels = [INT_TO_LABEL.get(int(k), str(k)) if isinstance(k, (int, np.integer)) else str(k) for k in labels]
    prob_map = {str(k): float(v) for k, v in zip(mapped_labels, probs)}
    for label in bundle.labels:
        prob_map.setdefault(label, 0.0)
    if bundle.draw_threshold is not None:
        pred = _apply_draw_threshold(prob_map, float(bundle.draw_threshold))
    else:
        pred = max(prob_map.items(), key=lambda item: item[1])[0]
    return pred, prob_map
