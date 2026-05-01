from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from models.match_predictor import train_match_model
from utils.data_loader import load_results
from utils.model_registry import save_model_bundle


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline trainer for match prediction model.")
    parser.add_argument(
        "--model",
        dest="model_name",
        choices=["XGBoost", "Random Forest", "Gradient Boosting", "Logistic Regression"],
        default="XGBoost",
        help="Estimator family to train and persist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    logging.info("Loading latest results dataset")
    results = load_results()
    ts = Path("data/results.csv").stat().st_mtime
    data_last_updated = datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()
    logging.info("Training model: %s", args.model_name)
    bundle = train_match_model(results, args.model_name, retrain_flag=1)
    metrics = bundle.metrics
    saved = save_model_bundle(
        bundle=bundle,
        model_name=args.model_name,
        accuracy=float(metrics["accuracy"]),
        f1_macro=float(metrics["f1_macro"]),
        data_last_updated=data_last_updated,
    )
    logging.info("Model saved successfully")
    logging.info("Version: %s | Accuracy: %.4f | F1: %.4f", saved["version"], saved["accuracy"], saved["f1_macro"])
    report = metrics.get("classification_report", {})
    for code, label in [("H", "Home Win"), ("D", "Draw"), ("A", "Away Win")]:
        if code in report:
            logging.info(
                "%s F1: %.4f | Precision: %.4f | Recall: %.4f",
                label,
                float(report[code].get("f1-score", 0.0)),
                float(report[code].get("precision", 0.0)),
                float(report[code].get("recall", 0.0)),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
