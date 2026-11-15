"""Read fixed experiment specifications without fitting any models."""

from __future__ import annotations

import inspect
from pathlib import Path
import tomllib

MODEL_TYPES = ("popularity", "knn", "lightgbm", "embedding", "embedding_trained", "xgboost")
SUPERVISED_TYPES = {"lightgbm", "embedding_trained", "xgboost"}
EMBEDDING_TYPES = {"embedding", "embedding_trained", "xgboost"}


def model_class(kind):
    # Lazy imports keep popularity/KNN runs independent of embedding files and XGBoost.
    if kind == "popularity":
        from remy.model.popularity import Popularity
        return Popularity
    if kind == "knn":
        from remy.model.knn import KNNRecommender
        return KNNRecommender
    if kind == "lightgbm":
        from remy.model.lightgbm import LightGBM
        return LightGBM
    if kind == "embedding":
        from remy.model.embedding import EmbeddingRecommender
        return EmbeddingRecommender
    if kind == "embedding_trained":
        from remy.model.embedding_trained import TrainedEmbeddingRecommender
        return TrainedEmbeddingRecommender
    if kind == "xgboost":
        from remy.model.xgboost import XGBoost
        return XGBoost
    raise ValueError(f"Unknown model type {kind!r}; choose from {MODEL_TYPES}")


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path):
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    if config.get("schema_version") != 1:
        raise ValueError("Experiment config must have schema_version = 1")
    unexpected = set(config) - {"schema_version", "description", "evaluation", "embeddings", "models"}
    if unexpected:
        raise ValueError(f"Unknown config fields: {sorted(unexpected)}")
    evaluation = config.get("evaluation", {})
    unexpected = set(evaluation) - {"split", "candidate_k", "save_top_k"}
    if unexpected:
        raise ValueError(f"Unknown evaluation fields: {sorted(unexpected)}")
    if evaluation.get("split", "val") not in {"val", "test"}:
        raise ValueError("evaluation.split must be 'val' or 'test'")
    for name in ("candidate_k", "save_top_k"):
        positive_integer(evaluation.get(name, 1000 if name == "candidate_k" else 100), name)
    if set(config.get("embeddings", {})) - {"directory"}:
        raise ValueError("embeddings only accepts a directory field")
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Config must contain at least one named model")
    for name, spec in models.items():
        if not name or not all(char.isalnum() or char in "_-" for char in name):
            raise ValueError(f"Model name {name!r} must contain only letters, numbers, '_' or '-'")
        unexpected = set(spec) - {"type", "label", "source", "notes", "params", "reference_val"}
        if unexpected:
            raise ValueError(f"{name}: unknown fields {sorted(unexpected)}")
        if spec.get("type") not in MODEL_TYPES:
            raise ValueError(f"{name}: unknown model type {spec.get('type')!r}")
        params = spec.get("params", {})
        if not isinstance(params, dict) or "embeddings" in params:
            raise ValueError(f"{name}: params must be a table; use embeddings.directory for arrays")
    return config


def make_configured_model(spec, embeddings=None):
    params = dict(spec.get("params", {}))
    if "rating_weights" in params:
        # TOML table keys are strings; interaction ratings are integers.
        params["rating_weights"] = {int(key): value for key, value in params["rating_weights"].items()}
    if spec["type"] in EMBEDDING_TYPES and embeddings is not None:
        params["embeddings"] = embeddings
    cls = model_class(spec["type"])
    try:
        inspect.signature(cls).bind(**params)
    except TypeError as error:
        raise ValueError(f"{spec['type']}: {error}") from error
    return cls(**params)
