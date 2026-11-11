"""Survey state and the JSONL response log.

In-progress surveys live in memory (a server restart drops unfinished ones).
Each completed survey is appended to the log as one JSON line with everything
needed for analysis, including which model produced each list.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResponseLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, dict] = {}  # user_id -> in-progress survey
        self._lock = Lock()

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_responses(path: str | Path) -> pd.DataFrame:
    """One row per rated recommendation, with the survey-level fields repeated.

    The raw nested records are just: pd.read_json(path, lines=True)
    """
    records = pd.read_json(path, lines=True).to_dict("records")
    rows = []
    for r in records:
        for lst in r["lists"]:
            for item in lst["items"]:
                rows.append({
                    "user_id": r["user_id"],
                    "external_id": r["external_id"],
                    "completed_at": r["completed_at"],
                    "train_cutoff": r["train_cutoff"],
                    "model": lst["model"],
                    "side": lst["side"],
                    "list_rating": lst["overall_rating"],
                    "preferred_model": r["preferred_model"],
                    **item,
                })
    return pd.DataFrame(rows)
