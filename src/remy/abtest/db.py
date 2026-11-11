"""SQLite storage for survey responses (stdlib sqlite3, one file)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id TEXT UNIQUE NOT NULL,   -- random UUID kept in the browser cookie
    external_id    TEXT,                   -- optional ?pid= from the invite link
    created_at     TEXT NOT NULL,
    model_left     TEXT NOT NULL,          -- shown as "List 1"
    model_right    TEXT NOT NULL,          -- shown as "List 2"
    stage          TEXT NOT NULL,          -- quiz -> compare -> done
    completed_at   TEXT,
    config_json    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quiz_ratings (
    participant_id TEXT NOT NULL,
    position       INTEGER NOT NULL,
    recipe_id      INTEGER NOT NULL,
    rating         INTEGER,                -- NULL until the quiz is submitted
    PRIMARY KEY (participant_id, position)
);
CREATE TABLE IF NOT EXISTS shown_items (
    participant_id TEXT NOT NULL,
    model          TEXT NOT NULL,
    side           TEXT NOT NULL,          -- left | right
    rank           INTEGER NOT NULL,       -- 1 = model's top recommendation
    recipe_id      INTEGER NOT NULL,
    rating         INTEGER,                -- NULL until the comparison is submitted
    PRIMARY KEY (participant_id, side, rank)
);
CREATE TABLE IF NOT EXISTS list_ratings (
    participant_id TEXT NOT NULL,
    model          TEXT NOT NULL,
    side           TEXT NOT NULL,
    overall_rating INTEGER NOT NULL,
    PRIMARY KEY (participant_id, side)
);
CREATE TABLE IF NOT EXISTS preferences (
    participant_id  TEXT PRIMARY KEY,
    preferred_model TEXT                   -- NULL = no preference
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def create_participant(self, participant_id, external_id, model_left, model_right,
                           quiz_ids, config_json) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO participants (participant_id, external_id, created_at, model_left,"
                " model_right, stage, config_json) VALUES (?, ?, ?, ?, ?, 'quiz', ?)",
                (participant_id, external_id, _now(), model_left, model_right, config_json),
            )
            conn.executemany(
                "INSERT INTO quiz_ratings (participant_id, position, recipe_id) VALUES (?, ?, ?)",
                [(participant_id, i, r) for i, r in enumerate(quiz_ids)],
            )

    def participant(self, participant_id) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM participants WHERE participant_id = ?", (participant_id,)
            ).fetchone()
        return dict(row) if row else None

    def quiz_ids(self, participant_id) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT recipe_id FROM quiz_ratings WHERE participant_id = ? ORDER BY position",
                (participant_id,),
            ).fetchall()
        return [r["recipe_id"] for r in rows]

    def save_quiz(self, participant_id, ratings: dict[int, int], shown: list[tuple]) -> None:
        """Store quiz ratings and the two recommendation lists, then move to 'compare'.

        shown: (model, side, rank, recipe_id) tuples.
        """
        with self._connect() as conn:
            conn.executemany(
                "UPDATE quiz_ratings SET rating = ? WHERE participant_id = ? AND recipe_id = ?",
                [(rating, participant_id, recipe_id) for recipe_id, rating in ratings.items()],
            )
            conn.executemany(
                "INSERT INTO shown_items (participant_id, model, side, rank, recipe_id)"
                " VALUES (?, ?, ?, ?, ?)",
                [(participant_id, *row) for row in shown],
            )
            conn.execute(
                "UPDATE participants SET stage = 'compare' WHERE participant_id = ?",
                (participant_id,),
            )

    def shown(self, participant_id) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model, side, rank, recipe_id FROM shown_items"
                " WHERE participant_id = ? ORDER BY side, rank",
                (participant_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_compare(self, participant_id, item_ratings: dict[tuple[str, int], int],
                     list_ratings: dict[str, tuple[str, int]], preferred_model: str | None,
                     ask_preference: bool) -> None:
        """item_ratings: (side, rank) -> rating; list_ratings: side -> (model, rating)."""
        with self._connect() as conn:
            conn.executemany(
                "UPDATE shown_items SET rating = ? WHERE participant_id = ? AND side = ? AND rank = ?",
                [(rating, participant_id, side, rank) for (side, rank), rating in item_ratings.items()],
            )
            conn.executemany(
                "INSERT INTO list_ratings (participant_id, model, side, overall_rating)"
                " VALUES (?, ?, ?, ?)",
                [(participant_id, model, side, rating) for side, (model, rating) in list_ratings.items()],
            )
            if ask_preference:
                conn.execute(
                    "INSERT INTO preferences (participant_id, preferred_model) VALUES (?, ?)",
                    (participant_id, preferred_model),
                )
            conn.execute(
                "UPDATE participants SET stage = 'done', completed_at = ? WHERE participant_id = ?",
                (_now(), participant_id),
            )


def load_responses(path: str | Path) -> dict[str, pd.DataFrame]:
    """Every table as a DataFrame, for analysis notebooks."""
    with sqlite3.connect(path) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {
            t: pd.read_sql_query(f"SELECT * FROM {t}", conn)
            for t in tables if not t.startswith("sqlite_")
        }
