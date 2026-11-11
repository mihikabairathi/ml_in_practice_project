"""Load and validate the survey config (config.yaml)."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from remy.paths import PROJECT_ROOT

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


class SurveySettings(BaseModel):
    n_quiz: int = Field(10, ge=1)
    n_recs: int = Field(6, ge=1)
    quiz_pool_size: int = Field(300, ge=1)
    show_community_stats: bool = False
    ask_preference: bool = True


class ModelSpec(BaseModel):
    type: str
    params: dict[str, Any] = {}


class Config(BaseModel):
    train_cutoff: dt.date
    responses_path: Path
    model_cache_dir: Path
    survey: SurveySettings = SurveySettings()
    models: dict[str, ModelSpec]

    @field_validator("models")
    @classmethod
    def _two_models(cls, models):
        if set(models) != {"a", "b"}:
            raise ValueError("models must have exactly the keys 'a' and 'b'")
        from remy.abtest.models import MODELS

        for spec in models.values():
            if spec.type not in MODELS:
                raise ValueError(f"Unknown model type {spec.type!r}; choose from {sorted(MODELS)}")
        return models

    @field_validator("responses_path", "model_cache_dir")
    @classmethod
    def _from_root(cls, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    def model_label(self, key: str) -> str:
        """Readable, unique name stored in the database for model slot a or b."""
        a, b = self.models["a"].type, self.models["b"].type
        return self.models[key].type if a != b else f"{self.models[key].type}_{key}"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("REMY_ABTEST_CONFIG", DEFAULT_CONFIG))
    return Config.model_validate(yaml.safe_load(path.read_text()))
