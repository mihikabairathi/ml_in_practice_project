"""Survey state and the JSONL response logs.

In-progress surveys live in memory (a server restart drops unfinished ones).
Each server start appends to its own file, responses_<YYYYmmdd-HHMMSS>.jsonl,
one JSON line per completed survey with everything needed for analysis,
including which model produced each list.
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
    def __init__(self, directory: str | Path, started_at: datetime | None = None):
        """One file per server start. It is created on the first completed survey."""
        self.started_at = (started_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"responses_{self.started_at:%Y%m%d-%H%M%S}.jsonl"
        self.sessions: dict[str, dict] = {}  # user_id -> in-progress survey
        self._lock = Lock()

    def append(self, record: dict) -> None:
        record = {**record, "server_started_at": self.started_at.isoformat(timespec="seconds")}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def response_files(path: str | Path) -> list[Path]:
    """A single .jsonl file, or every .jsonl file in a folder (sorted, i.e. by start time)."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(f"No response files at {path}")
    return files


def read_surveys(path: str | Path) -> pd.DataFrame:
    """One row per completed survey (the raw JSON records) from a file or a folder
    of files, with a `file` column naming the deployment file it came from."""
    frames = [pd.read_json(f, lines=True).assign(file=f.name)
              for f in response_files(path) if f.stat().st_size]
    if not frames:
        raise ValueError(f"No completed surveys in {path}")
    return pd.concat(frames, ignore_index=True)


def load_responses(path: str | Path) -> pd.DataFrame:
    """One row per rated recommendation, with the survey-level fields repeated.
    `path` is a responses file or a folder of them (e.g. data/processed/abtest/responses)."""
    rows = []
    for r in read_surveys(path).to_dict("records"):
        for lst in r["lists"]:
            for item in lst["items"]:
                rows.append({
                    "user_id": r["user_id"],
                    "external_id": r["external_id"],
                    "completed_at": r["completed_at"],
                    "train_cutoff": r["train_cutoff"],
                    "file": r["file"],
                    "model": lst["model"],
                    "side": lst["side"],
                    "list_rating": lst["overall_rating"],
                    "preferred_model": r["preferred_model"],
                    **item,
                })
    return pd.DataFrame(rows)
