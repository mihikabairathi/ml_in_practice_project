"""Run comparable rolling-quarter recommendation experiments.

Examples (after ``python scripts/make_splits.py``)::

    python scripts/run_experiments.py --models popularity knn --split val
    python scripts/run_experiments.py --models lightgbm --split test --max-quarters 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from remy.evaluate import METRICS, evaluate_quarter
from remy.model.knn import KNNRecommender
from remy.model.lightgbm import LightGBM
from remy.model.popularity import Popularity
from remy.paths import PROJECT_ROOT
from remy.plots import save_plots
from remy.splits import SPLIT_DIR, RollingSplits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit and evaluate recommenders on shared rolling quarterly splits.",
        epilog="Example: python scripts/run_experiments.py --models popularity knn --split val",
    )
    models = parser.add_argument_group("models and evaluation")
    models.add_argument(
        "--models",
        nargs="+",
        choices=("popularity", "knn", "lightgbm"),
        default=["popularity"],
        help="Models to compare (default: popularity)",
    )
    models.add_argument(
        "--split", choices=("val", "test"), default="val",
        help="Rolling evaluation period (default: val)",
    )
    models.add_argument(
        "--max-quarters", type=int,
        help="Score only the first N quarters, for a shorter run",
    )

    paths = parser.add_argument_group("data and outputs")
    paths.add_argument("--split-dir", type=Path, default=SPLIT_DIR, help="Input directory from make_splits.py")
    paths.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts", help="Parent directory for timestamped runs")
    paths.add_argument("--run-name", help="Name the new run folder (default: timestamp, split, and models)")
    paths.add_argument("--output", type=Path, help="Write the results CSV at this path instead of creating a run folder")
    paths.add_argument("--no-plots", action="store_true", help="Skip saving PNG charts")

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
    positive_args = (
        "max_quarters", "k_neighbors", "negative_samples_per_positive",
        "n_candidates", "boost_rounds", "num_leaves", "min_data_in_leaf",
    )
    for name in positive_args:
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def make_model(name, args):
    if name == "popularity":
        return Popularity()
    if name == "knn":
        return KNNRecommender(
            k_neighbors=args.k_neighbors,
            min_rating=args.min_rating,
            n_candidates=args.n_candidates,
        )
    if name == "lightgbm":
        return LightGBM(
            negatives_per_positive=args.negative_samples_per_positive,
            num_boost_round=args.boost_rounds,
            n_candidates=args.n_candidates,
            seed=args.seed,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
        )
    raise ValueError(f"Unknown model: {name}")


def run(args):
    splits = RollingSplits(args.split_dir)
    quarters = splits.evaluation_quarters(args.split)
    if args.max_quarters is not None:
        quarters = quarters[: args.max_quarters]

    rows = []
    for quarter in quarters:
        for name in dict.fromkeys(args.models):
            model = make_model(name, args)
            training_quarters = (
                splits.training_quarters(quarter.start) if name == "lightgbm" else ()
            )
            model.fit(
                quarter.history,
                recipes=quarter.recipes,
                training_quarters=training_quarters,
            )
            positives = quarter.outcomes.loc[quarter.outcomes["rating"] == 5]
            user_ids = positives["user_id"].unique()
            # The evaluator can use popularity's global ranking directly,
            # filtering seen recipes without building per-user recommendation lists.
            recommendations = (
                None if name == "popularity"
                else model.recommend_batch(user_ids, k=None)
            )
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
            print(
                f"{name:12s} {quarter.start.date()}  "
                f"NDCG={row['NDCG']:.4f}  HR@10={row['HR@10']:.4f}",
                flush=True,
            )

    if not rows:
        raise ValueError(f"No {args.split} evaluation quarters in {args.split_dir}")

    results = pd.DataFrame(rows)
    if args.output is None:
        model_names = "-".join(dict.fromkeys(args.models))
        run_name = args.run_name or (
            f"{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}_{args.split}_{model_names}"
        )
        output = args.artifacts_dir / run_name / "results.csv"
        output.parent.mkdir(parents=True, exist_ok=False)
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output, index=False)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (output.parent / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not args.no_plots:
        plot_paths = save_plots(results, output)
        print("Saved charts: " + ", ".join(map(str, plot_paths)))

    summary = results.groupby("model")[METRICS].mean().round(4)
    print(f"\nMean across quarters:\n{summary.to_string()}")
    print(f"\nSaved {output}")
    return results


if __name__ == "__main__":
    run(parse_args())
