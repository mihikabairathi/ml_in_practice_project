"""Run comparable rolling-quarter recommendation experiments.

Examples (after ``python scripts/make_splits.py``)::

    python scripts/run_experiments.py --models popularity knn --split val
    python scripts/run_experiments.py --models lightgbm --split test --max-quarters 2
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
from itertools import islice
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd

from remy.evaluate import METRICS, evaluate_quarter
from remy.experiment_config import (
    EMBEDDING_TYPES, MODEL_TYPES, SUPERVISED_TYPES, load_config, make_configured_model,
)
from remy.paths import PROJECT_ROOT
from remy.plots import save_plots
from remy.splits import SPLIT_DIR, RollingSplits


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Fit and evaluate recommenders on shared rolling quarterly splits.",
        epilog="Example: python scripts/run_experiments.py --models popularity knn --split val",
        allow_abbrev=False,
    )
    models = parser.add_argument_group("models and evaluation")
    models.add_argument("--config", type=Path, help="TOML file of fixed model configurations")
    models.add_argument("--describe-config", action="store_true", help="Print selected settings without loading data or fitting")
    models.add_argument(
        "--models",
        nargs="+",
        help="Config model names to select, or model types without --config (default: all config models, otherwise popularity)",
    )
    models.add_argument(
        "--split", choices=("val", "test"),
        help="Rolling evaluation period (default: val)",
    )
    models.add_argument(
        "--max-quarters", type=int,
        help="Score only the first N quarters, for a shorter run",
    )
    models.add_argument("--max-users", type=int, help="Score a common random sample of N positive-outcome users per quarter; training is unchanged")
    models.add_argument("--sample-seed", type=int, default=10718, help="Evaluation user-sampling seed")

    paths = parser.add_argument_group("data and outputs")
    paths.add_argument("--split-dir", type=Path, default=SPLIT_DIR, help="Input directory from make_splits.py")
    paths.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts", help="Parent directory for timestamped runs")
    paths.add_argument("--run-name", help="Name the new run folder (default: timestamp, split, and models)")
    paths.add_argument("--output", type=Path, help="Write the results CSV at this path instead of creating a run folder")
    paths.add_argument("--no-plots", action="store_true", help="Skip saving PNG charts")
    paths.add_argument("--embedding-dir", type=Path, help="Override saved embedding directory (paths in config resolve from the project root)")
    paths.add_argument("--save-top-k", type=int, help="Save this many completed recommendations per user (default: config setting or 100)")

    knn = parser.add_argument_group("KNN settings")
    knn.add_argument("--k-neighbors", type=int, default=50, help="Neighbors retained per recipe (default: 50)")
    knn.add_argument("--min-rating", type=float, default=5.0, help="Minimum rating treated as positive (default: 5)")

    candidates = parser.add_argument_group("Candidate shortlist settings")
    candidates.add_argument(
        "--n-candidates", type=int, default=1000,
        help="Popular recipes shortlisted for KNN and LightGBM (default: 1000)",
    )

    lgbm = parser.add_argument_group("LightGBM settings")
    lgbm.add_argument(
        "--negative-samples-per-positive", type=int, default=10,
        help="Sampled negatives per training positive (default: 10, matching rolling_lightgbm.ipynb)",
    )
    lgbm.add_argument("--boost-rounds", type=int, default=200, help="LightGBM boosting rounds (default: 200)")
    lgbm.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate (default: 0.05)")
    lgbm.add_argument("--num-leaves", type=int, default=31, help="Maximum LightGBM leaves per tree (default: 31)")
    lgbm.add_argument("--min-data-in-leaf", type=int, default=100, help="Minimum rows per LightGBM leaf (default: 100)")
    lgbm.add_argument("--seed", type=int, default=10718, help="Random seed for LightGBM sampling (default: 10718)")

    args = parser.parse_args(argv)
    if args.config:
        # Frozen settings must never silently lose to model-tuning CLI arguments.
        tuning_flags = {
            "--k-neighbors", "--min-rating", "--n-candidates", "--negative-samples-per-positive",
            "--boost-rounds", "--learning-rate", "--num-leaves", "--min-data-in-leaf", "--seed",
        }
        supplied = {argument.split("=", 1)[0] for argument in argv}
        if supplied & tuning_flags:
            parser.error("With --config, edit model params in the config instead of supplying tuning flags")
        try:
            args.frozen_config = load_config(args.config)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
        evaluation = args.frozen_config.get("evaluation", {})
        args.models = args.models or list(args.frozen_config["models"])
        unknown = set(args.models) - set(args.frozen_config["models"])
        if unknown:
            parser.error(f"Unknown config model names: {sorted(unknown)}")
        args.model_specs = {name: args.frozen_config["models"][name] for name in dict.fromkeys(args.models)}
        args.split = args.split or evaluation.get("split", "val")
        args.n_candidates = evaluation.get("candidate_k", 1000)
        args.save_top_k = args.save_top_k if args.save_top_k is not None else evaluation.get("save_top_k", 100)
        if args.embedding_dir is None:
            args.embedding_dir = Path(args.frozen_config.get("embeddings", {}).get(
                "directory", "data/processed/embeddings/minilm_joint"))
            if not args.embedding_dir.is_absolute():
                args.embedding_dir = PROJECT_ROOT / args.embedding_dir
    else:
        args.frozen_config = None
        args.models = args.models or ["popularity"]
        if set(args.models) - set(MODEL_TYPES):
            parser.error(f"Choose model types from {MODEL_TYPES}")
        args.split = args.split or "val"
        args.save_top_k = args.save_top_k if args.save_top_k is not None else 100
        args.model_specs = {name: legacy_spec(name, args) for name in dict.fromkeys(args.models)}
    args.models = list(args.model_specs)
    args.argv = argv
    positive_args = (
        "max_quarters", "k_neighbors", "negative_samples_per_positive",
        "n_candidates", "boost_rounds", "num_leaves", "min_data_in_leaf", "max_users", "save_top_k",
    )
    for name in positive_args:
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def legacy_spec(name, args):
    """Preserve the original CLI's defaults when no frozen config is supplied."""
    if name == "popularity":
        params = {}
    elif name == "knn":
        params = dict(k_neighbors=args.k_neighbors, min_rating=args.min_rating, n_candidates=args.n_candidates)
    elif name == "lightgbm":
        params = dict(
            negatives_per_positive=args.negative_samples_per_positive,
            num_boost_round=args.boost_rounds,
            n_candidates=args.n_candidates,
            seed=args.seed,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
        )
    elif name == "embedding":
        params = dict(method="centered", n_candidates=args.n_candidates)
    elif name == "embedding_trained":
        params = dict(variant="softmax", architecture="nonlinear", width=128,
                      epochs=15, learning_rate=0.001, batch_size=512,
                      n_candidates=args.n_candidates, seed=args.seed)
    elif name == "xgboost":
        params = dict(seed=args.seed, n_candidates=args.n_candidates)
    else:
        raise ValueError(f"Unknown model: {name}")
    return {"type": name, "params": params}


def make_model(name, args, embeddings=None):
    return make_configured_model(args.model_specs[name], embeddings)


def completed_top_k(user_id, recs, global_ranking, seen, k):
    """Use exactly the evaluator's candidate-first, unseen-popularity completion."""
    candidates = list(recs.get(user_id, ()))
    blocked = set(seen.get(user_id, ())) | set(candidates)
    return candidates[:k] + list(islice(
        (recipe for recipe in global_ranking if recipe not in blocked), max(0, k - len(candidates))
    ))


def validate_recommendations(recs, user_ids, catalog, seen):
    unknown_users = set(recs) - set(user_ids)
    if unknown_users:
        raise ValueError(f"Recommendations contain unrequested users: {unknown_users}")
    available = set(catalog)
    for user_id, recipes in recs.items():
        unique = set(recipes)
        if len(unique) != len(recipes) or not unique <= available or unique & seen.get(user_id, set()):
            raise ValueError(f"Invalid recommendations for user {user_id}: duplicate, unavailable or previously seen recipe")


def run_metadata(args):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    files = [Path(__file__), PROJECT_ROOT / "src/remy/experiment_config.py",
             PROJECT_ROOT / "src/remy/evaluate.py", PROJECT_ROOT / "src/remy/splits.py"]
    files.extend((PROJECT_ROOT / "src/remy/model").glob("*.py"))
    config["source_sha256"] = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    config["versions"] = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch")}
    if any(spec["type"] == "xgboost" for spec in args.model_specs.values()):
        config["versions"]["xgboost"] = version("xgboost")
    try:
        config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        config["git_commit"] = None
    return config


def prepare_output(args):
    if args.output is None:
        model_names = "-".join(args.models)
        run_name = args.run_name or f"{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}_{args.split}_{model_names}"
        output = args.artifacts_dir / run_name / "results.csv"
        output.parent.mkdir(parents=True, exist_ok=False)
    else:
        output = args.output
        if output.exists() or (output.parent / "run_config.json").exists():
            raise FileExistsError(f"Use a fresh output directory to preserve prediction provenance: {output.parent}")
        output.parent.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(args)
    metadata["execution"] = "one fresh process per model" if len(args.models) > 1 else "single model"
    (output.parent / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def finish_results(results, output, args):
    if not args.no_plots:
        plot_paths = save_plots(results, output)
        print("Saved charts: " + ", ".join(map(str, plot_paths)))
    summary = results.groupby("model")[METRICS].mean().round(4)
    print(f"\nMean across quarters:\n{summary.to_string()}")
    print(f"\nSaved {output}")
    return results


def run_isolated(args):
    """Never fit Torch, LightGBM and XGBoost together in one native runtime.

    In particular, their macOS wheels can load incompatible OpenMP libraries.
    Each worker uses the ordinary single-model CLI; the parent merges artifacts.
    """
    import shutil

    # Check missing embedding files before spending time on any model.
    for spec in args.model_specs.values():
        if spec["type"] in EMBEDDING_TYPES:
            from remy.model.embedding import EMBEDDING_DIR
            directory = args.embedding_dir or EMBEDDING_DIR
            required = ("id.npy", "vectors.npz") if spec.get("params", {}).get("variant") == "tfidf" else ("id.npy", "name.npy", "ingredients.npy", "vectors.npy")
            if any(not (directory / file).is_file() for file in required):
                raise FileNotFoundError("Saved recipe embeddings are missing. Run python scripts/embed_recipes.py first, or use --embedding-dir.")
    output = prepare_output(args)
    parts = []
    for name in args.models:
        worker_output = output.parent / "workers" / name / "results.csv"
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        log_path = worker_output.parent / "run.log"
        command = [sys.executable, str(Path(__file__).resolve()), *args.argv,
                   "--models", name, "--output", str(worker_output.resolve()), "--no-plots"]
        print(f"Running {name} in a fresh process; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if worker_output.exists():
            parts.append(pd.read_csv(worker_output, parse_dates=["start"]))
            pd.concat(parts, ignore_index=True).to_csv(output, index=False)
        predictions = worker_output.parent / "recommendations"
        if predictions.exists():
            shutil.copytree(predictions, output.parent / "recommendations", dirs_exist_ok=True)
        if completed.returncode:
            raise RuntimeError(f"{name} worker exited with code {completed.returncode}. "
                               f"See {log_path}; completed scores and predictions are preserved in {output.parent}.")
        if not worker_output.exists():
            raise RuntimeError(f"{name} worker produced no scores; see {log_path}")
        print(f"Finished {name}", flush=True)
    return finish_results(pd.concat(parts, ignore_index=True), output, args)


def run(args):
    if args.describe_config:
        print(json.dumps({"split": args.split, "candidate_k": args.n_candidates,
                          "save_top_k": args.save_top_k, "embedding_dir": str(args.embedding_dir),
                          "models": args.model_specs}, indent=2))
        return None
    if len(args.models) > 1:
        return run_isolated(args)
    splits = RollingSplits(args.split_dir)
    quarters = splits.evaluation_quarters(args.split)
    if args.max_quarters is not None:
        quarters = quarters[: args.max_quarters]

    # Resolve prerequisites before doing any fits, and reuse frozen arrays across models.
    embedding_cache = {}
    for spec in args.model_specs.values():
        if spec["type"] not in EMBEDDING_TYPES:
            continue
        variant = spec.get("params", {}).get("variant", "joint")
        kind = variant if variant in {"tfidf", "separate_fields"} else "joint"
        if kind not in embedding_cache:
            from remy.model.embedding import load_embeddings
            try:
                embedding_cache[kind] = load_embeddings(kind, directory=args.embedding_dir)
            except FileNotFoundError as error:
                raise FileNotFoundError("Saved recipe embeddings are missing. Run python scripts/embed_recipes.py first, or use --embedding-dir.") from error
            ids = embedding_cache[kind]["id"]
            if len(ids) != len(set(ids)) or not set(splits.recipes.id) <= set(ids):
                raise ValueError("Embedding IDs must be unique and cover every recipe in the split catalog")

    output = prepare_output(args)

    rows = []
    for quarter in quarters:
        user_ids = quarter.outcomes.loc[quarter.outcomes.rating.eq(5), "user_id"].unique()
        if args.max_users is not None and len(user_ids) > args.max_users:
            import numpy as np
            user_ids = np.random.default_rng(args.sample_seed).choice(sorted(user_ids), args.max_users, replace=False)
            quarter = replace(quarter, outcomes=quarter.outcomes[quarter.outcomes.user_id.isin(user_ids)])
        training_quarters = (splits.training_quarters(quarter.start)
                             if any(spec["type"] in SUPERVISED_TYPES for spec in args.model_specs.values()) else ())
        for name, spec in args.model_specs.items():
            kind = spec.get("params", {}).get("variant", "joint")
            kind = kind if kind in {"tfidf", "separate_fields"} else "joint"
            model = make_model(name, args, embedding_cache.get(kind))
            kwargs = {"recipes": quarter.recipes, "training_quarters": training_quarters if spec["type"] in SUPERVISED_TYPES else ()}
            if spec["type"] in {"lightgbm", "xgboost"}:
                kwargs["cutoff"] = quarter.start
            model.fit(quarter.history, **kwargs)
            # The evaluator can use popularity's global ranking directly,
            # filtering seen recipes without building per-user recommendation lists.
            if spec["type"] == "popularity":
                recommendations = {}
            elif spec["type"] in {"embedding", "embedding_trained"}:
                recommendations = model.recommend_many(user_ids, k=None)
            else:
                recommendations = model.recommend_batch(user_ids, k=None)
            validate_recommendations(recommendations, user_ids, quarter.recipes.id, model.seen_)
            row = evaluate_quarter(
                quarter,
                model.ranking_,
                model.seen_,
                recommendations,
                candidate_k=args.n_candidates,
            )
            row["model"] = name
            row["split"] = args.split
            rows.append(row)
            predictions = pd.DataFrame({
                "user_id": user_ids,
                "recipe_ids": [completed_top_k(user, recommendations, model.ranking_, model.seen_, args.save_top_k)
                               for user in user_ids],
            })
            prediction_dir = output.parent / "recommendations" / str(quarter.start.date())
            prediction_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(prediction_dir / f"{name}.parquet", index=False)
            # Checkpoint both scores and served lists so later metrics never require refitting.
            pd.DataFrame(rows).to_csv(output, index=False)
            print(
                f"{name:12s} {quarter.start.date()}  "
                f"NDCG={row['NDCG']:.3f}  HR@10={row['HR@10']:.3f}",
                flush=True,
            )

    if not rows:
        raise ValueError(f"No {args.split} evaluation quarters in {args.split_dir}")

    results = pd.DataFrame(rows)
    return finish_results(results, output, args)


if __name__ == "__main__":
    run(parse_args())
