from __future__ import annotations

import argparse
import logging

from utils.sqlite_db import build_sqlite_db, db_exists, get_db_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load CSV datasets into SQLite.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if db already exists.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if db_exists() and not args.force:
        logging.info("Database already exists: %s", get_db_path())
        return 0
    logging.info("Building SQLite database...")
    path = build_sqlite_db(force=args.force)
    logging.info("Done: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
