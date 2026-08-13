from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from utils.data_loader import check_match_data_availability, check_player_data_availability
from utils.helpers import get_data_dir, get_db_path, get_sql_dir

CSV_TABLES = {
    "results": "results.csv",
    "goalscorers": "goalscorers.csv",
    "shootouts": "shootouts.csv",
    "players": "players.csv",
    "appearances": "appearances.csv",
    "games": "games.csv",
    "clubs": "clubs.csv",
    "competitions": "competitions.csv",
}

CHUNK_SIZE = 100_000


def db_exists() -> bool:
    path = get_db_path()
    return path.exists() and path.stat().st_size > 0


def get_connection(read_only: bool = False) -> sqlite3.Connection:
    path = get_db_path()
    if read_only:
        uri = f"file:{path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _is_read_only_sql(sql: str) -> bool:
    cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.MULTILINE | re.DOTALL).strip()
    if not cleaned:
        return False
    first = cleaned.split(";", 1)[0].strip().lower()
    return first.startswith("select") or first.startswith("with") or first.startswith("pragma")


def run_query(sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    if not _is_read_only_sql(sql):
        raise ValueError("Only read-only SELECT / WITH / PRAGMA queries are allowed.")
    conn = get_connection(read_only=True)
    try:
        return pd.read_sql_query(sql, conn, params=params or ())
    finally:
        conn.close()


def _load_csv(conn: sqlite3.Connection, table: str, csv_name: str) -> None:
    csv_path = get_data_dir() / csv_name
    if not csv_path.exists():
        return
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    if table == "appearances":
        first = True
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            chunk.to_sql(table, conn, if_exists="append" if not first else "replace", index=False)
            first = False
    else:
        df = pd.read_csv(csv_path)
        df.to_sql(table, conn, if_exists="replace", index=False)


def _apply_sql_file(conn: sqlite3.Connection, filename: str) -> None:
    path = get_sql_dir() / filename
    if path.exists():
        conn.executescript(path.read_text(encoding="utf-8"))


def build_sqlite_db(force: bool = False) -> Path:
    match_ok, _ = check_match_data_availability()
    player_ok, _ = check_player_data_availability()
    if not match_ok and not player_ok:
        raise FileNotFoundError("No CSV files found in data/. Download datasets before building the database.")

    db_path = get_db_path()
    if db_exists() and not force:
        return db_path

    if db_path.exists():
        db_path.unlink()

    conn = get_connection()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        for table, csv_name in CSV_TABLES.items():
            _load_csv(conn, table, csv_name)
        _apply_sql_file(conn, "schema.sql")
        conn.commit()
    finally:
        conn.close()
    return db_path


def list_analysis_queries() -> list[dict[str, str]]:
    path = get_sql_dir() / "analysis_queries.sql"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    blocks: list[dict[str, str]] = []
    current_title = ""
    current_sql: list[str] = []
    for line in text.splitlines():
        if line.startswith("-- name:"):
            if current_title and current_sql:
                blocks.append({"title": current_title, "sql": "\n".join(current_sql).strip()})
            current_title = line.replace("-- name:", "", 1).strip()
            current_sql = []
        elif line.startswith("-- ") and not line.startswith("-- name:"):
            continue
        elif current_title:
            current_sql.append(line)
    if current_title and current_sql:
        blocks.append({"title": current_title, "sql": "\n".join(current_sql).strip()})
    return blocks
