from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from utils.helpers import get_artifacts_dir


REGISTRY_PATH = "match_model_registry.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _registry_file() -> Path:
    return get_artifacts_dir() / REGISTRY_PATH


def _ensure_artifact_dir() -> Path:
    path = get_artifacts_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_model_info() -> Optional[Dict[str, Any]]:
    reg = _registry_file()
    if not reg.exists():
        return None
    return json.loads(reg.read_text(encoding="utf-8"))


def load_latest_model_bundle() -> Tuple[Any, Dict[str, Any]]:
    info = get_latest_model_info()
    if not info:
        raise FileNotFoundError("No trained model metadata found.")
    model_path = Path(info["artifact_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact is missing: {model_path}")
    bundle = joblib.load(model_path)
    return bundle, info


def save_model_bundle(bundle: Any, model_name: str, accuracy: float, f1_macro: float, data_last_updated: str) -> Dict[str, Any]:
    artifact_dir = _ensure_artifact_dir()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    temp_path = artifact_dir / f"match_model_{version}.tmp.joblib"
    final_path = artifact_dir / f"match_model_{version}.joblib"
    reg_path = _registry_file()
    temp_reg = artifact_dir / f"{REGISTRY_PATH}.tmp"

    joblib.dump(bundle, temp_path)
    temp_path.replace(final_path)

    payload = {
        "version": version,
        "model_name": model_name,
        "artifact_path": str(final_path),
        "trained_at": _utc_now_iso(),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "data_last_updated": data_last_updated,
    }

    temp_reg.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_reg.replace(reg_path)
    return payload
