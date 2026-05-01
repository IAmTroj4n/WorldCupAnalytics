from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.match_predictor import build_fixture_feature_row, predict_fixture
from utils.model_registry import get_latest_model_info, load_latest_model_bundle


app = FastAPI(title="Worldcup Prediction API", version="1.0.0")

_MODEL_BUNDLE = None
_MODEL_VERSION = None


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    tournament: str = "FIFA World Cup"
    neutral: bool = True


def _ensure_model_loaded():
    global _MODEL_BUNDLE, _MODEL_VERSION
    info = get_latest_model_info()
    if not info:
        raise HTTPException(status_code=503, detail="Model artifact is not available. Run offline training first.")
    version = info["version"]
    if _MODEL_BUNDLE is None or _MODEL_VERSION != version:
        bundle, _ = load_latest_model_bundle()
        _MODEL_BUNDLE = bundle
        _MODEL_VERSION = version
    return _MODEL_BUNDLE, info


@app.get("/health")
def health() -> Dict[str, str]:
    info = get_latest_model_info()
    return {"status": "ok", "model_version": info["version"] if info else "unavailable"}


@app.post("/predict")
def predict(payload: PredictRequest):
    bundle, info = _ensure_model_loaded()
    feature_row = build_fixture_feature_row(
        bundle.frame,
        payload.home_team,
        payload.away_team,
        payload.tournament,
        payload.neutral,
    )
    outcome, probs = predict_fixture(bundle, feature_row)
    return {
        "prediction": outcome,
        "probabilities": probs,
        "model_version": info["version"],
        "trained_at": info["trained_at"],
        "accuracy": info["accuracy"],
        "f1_macro": info["f1_macro"],
    }
