from __future__ import annotations

import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

from pipelines.train_match_model import main as train_once


def _scheduled_train() -> None:
    logging.info("Scheduled retraining started")
    try:
        train_once()
        logging.info("Scheduled retraining finished")
    except Exception as exc:  # noqa: BLE001
        # Keep previous model active when retraining fails.
        logging.exception("Scheduled retraining failed: %s", exc)


def run_scheduler() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    scheduler = BackgroundScheduler()
    scheduler.add_job(_scheduled_train, trigger="interval", hours=6, id="train_match_model")
    scheduler.start()
    logging.info("Scheduler running: retraining every 6 hours")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()


if __name__ == "__main__":
    run_scheduler()
