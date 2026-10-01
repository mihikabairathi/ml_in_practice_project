"""Run comparable rolling-quarter recommendation experiments.

Examples (after ``python scripts/make_splits.py``)::

    python scripts/run_experiments.py --models popularity item_knn --split val
    python scripts/run_experiments.py --models lightgbm --split test --max-quarters 2
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from remy.model.item_knn import ItemKNNRecommender
from remy.model.popularity import Popularity
from remy.paths import PROJECT_ROOT
from remy.splits import RollingSplits, SPLIT_DIR

METRICS = ["NDCG", "HR@10", "HR@100", "Recall@10", "Recall@100"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation models on rolling quarterly splits.",
        epilog="Example: python scripts/run_experiments.py --models popularity item_knn --split val",
    )
    parser.add_argument("--models", nargs="+", choices=("popularity", "item_knn", "lightgbm"), default=["popularity"], help="Models to compare (default: popularity)")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Evaluation period (default: val)")
    parser.add_argument("--max-quarters", type=int, help="Score only the first N quarters, for a quick run")
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR, help="Directory from make_splits.py")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts", help="Root directory for experiment runs")
    parser.add_argument("--run-name", help="Run folder name (default: timestamp, split, and models)")
    parser.add_argument("--output", type=Path, help="Override the results CSV path and run folder")
    parser.add_argument("--k-neighbors", type=int, default=50, help="Item KNN neighbors per item")
    parser.add_argument("--min-rating", type=float, default=5.0, help="Item KNN positive rating threshold (default: 5)")
    parser.add_argument("--train-quarters", type=int, default=8, help="Most recent training quarters for LightGBM")
    parser.add_argument("--max-train-users", type=int, default=3000, help="Training users per quarter for LightGBM")
    parser.add_argument("--boost-rounds", type=int, default=200, help="LightGBM boosting rounds")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving the PNG charts")
    args = parser.parse_args(argv)
    for name in ("max_quarters", "train_quarters", "max_train_users", "boost_rounds"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.k_neighbors < 1:
        parser.error("--k-neighbors must be positive")
    return args


def make_model(name, args):
    if name == "popularity":
        return Popularity()
    if name == "item_knn":
        return ItemKNNRecommender(args.k_neighbors, args.min_rating)
    from remy.model.lightgbm_ranker import LightGBMRanker

    return LightGBMRanker(
        n_train_quarters=args.train_quarters,
        max_train_users=args.max_train_users,
        num_boost_round=args.boost_rounds,
    )


def evaluate_quarter(model, quarter, name):
    """Score the complete ranking: model candidates, then unseen popularity."""
    positives = quarter.outcomes.loc[quarter.outcomes["rating"] == 5]
    if positives.empty:
        raise ValueError(f"No scorable users at {quarter.start.date()}")
    users = positives["user_id"].unique()
    batch = model.recommend_many(users) if name == "lightgbm" else None
    global_rank = {recipe: rank for rank, recipe in enumerate(model.ranking_, start=1)}
    scores = []
    candidate_hits = 0
    for user_id, rows in positives.groupby("user_id"):
        relevant = set(rows["recipe_id"])
        if name == "lightgbm":
            candidates = batch[user_id]
        elif name == "item_knn":
            candidates = model.recommend_candidates(user_id)
        else:
            candidates = []
        candidate_rank = {recipe: rank for rank, recipe in enumerate(candidates, start=1)}
        candidate_hits += sum(recipe in candidate_rank for recipe in rows["recipe_id"])
        blocked = set(model.seen_.get(user_id, ())) | set(candidates)
        blocked_ranks = np.sort([global_rank[recipe] for recipe in blocked if recipe in global_rank])
        ranks = []
        for recipe in relevant:
            if recipe in candidate_rank:
                ranks.append(candidate_rank[recipe])
            else:
                popularity_rank = global_rank[recipe]
                ranks.append(len(candidates) + popularity_rank - np.searchsorted(blocked_ranks, popularity_rank))
        ranks = np.asarray(ranks)
        dcg = np.sum(1 / np.log2(ranks + 1))
        ideal = np.sum(1 / np.log2(np.arange(2, len(relevant) + 2)))
        row = {"NDCG": dcg / ideal}
        for k in (10, 100):
            hits = int(np.sum(ranks <= k))
            row[f"HR@{k}"] = float(hits > 0)
            row[f"Recall@{k}"] = hits / len(relevant)
        scores.append(row)
    result = {
        "model": name,
        "split": None,  # populated by the runner
        "start": quarter.start,
        "eligible_users": len(quarter.users),
        "scored_users": len(scores),
        "positives": len(positives),
        "candidate_recall": candidate_hits / len(positives) if name != "popularity" else np.nan,
    }
    result.update(pd.DataFrame(scores).mean().to_dict())
    return result


def save_plots(results: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Save quarter trends and a model comparison beside the results CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    models = list(results["model"].unique())
    colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(models)}
    titles = [
        ("NDCG", "Full-ranking NDCG"),
        ("HR@10", "Hit rate @10"),
        ("Recall@10", "Recall @10"),
    ]
    fourth = ("candidate_recall", "Candidate recall") if results["candidate_recall"].notna().any() else ("scored_users", "Scored users")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    for ax, (metric, title) in zip(axes.flat, [*titles, fourth]):
        for name in models:
            frame = results.loc[results["model"] == name].sort_values("start")
            if frame[metric].notna().any():
                ax.plot(frame["start"], frame[metric], marker="o", linewidth=2, label=name, color=colors[name])
        ax.set_title(title)
        ax.grid(alpha=0.2)
        if metric != "scored_users":
            peak = results[metric].max(skipna=True)
            ax.set_ylim(0, min(1, max(0.01, peak * 1.2)))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.suptitle(f"Rolling {results['split'].iloc[0]} performance", fontsize=16)
    trend_path = output.with_name(f"{output.stem}_quarters.png")
    fig.savefig(trend_path, dpi=160)
    plt.close(fig)

    means = results.groupby("model")[METRICS].mean().reindex(models)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(METRICS))
    width = 0.8 / len(models)
    for i, name in enumerate(models):
        ax.bar(x - 0.4 + width * (i + 0.5), means.loc[name], width, label=name, color=colors[name])
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0, min(1, max(0.01, means.max().max() * 1.2)))
    ax.set_ylabel("Mean across quarters")
    ax.set_title(f"Model comparison · {results['split'].iloc[0]}")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    summary_path = output.with_name(f"{output.stem}_summary.png")
    fig.savefig(summary_path, dpi=160)
    plt.close(fig)
    return trend_path, summary_path


def run(args):
    splits = RollingSplits(args.split_dir)
    quarters = splits.evaluation_quarters(args.split)
    if args.max_quarters is not None:
        quarters = quarters[:args.max_quarters]
    rows = []
    for quarter in quarters:
        for name in dict.fromkeys(args.models):
            model = make_model(name, args)
            fit_kwargs = {"recipes": quarter.recipes}
            if name == "lightgbm":
                fit_kwargs.update(training_quarters=splits.training_quarters(quarter.start), asof=quarter.start)
            model.fit(quarter.history, **fit_kwargs)
            row = evaluate_quarter(model, quarter, name)
            row["split"] = args.split
            rows.append(row)
            print(f"{name:12s} {quarter.start.date()}  NDCG={row['NDCG']:.4f}  HR@10={row['HR@10']:.4f}", flush=True)
    if not rows:
        raise ValueError(f"No {args.split} evaluation quarters in {args.split_dir}")
    results = pd.DataFrame(rows)
    if args.output is None:
        run_name = args.run_name or f"{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}_{args.split}_{'-'.join(dict.fromkeys(args.models))}"
        output = args.artifacts_dir / run_name / "results.csv"
        output.parent.mkdir(parents=True, exist_ok=False)
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output.parent / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if not args.no_plots:
        plot_paths = save_plots(results, output)
        print(f"Saved charts: {plot_paths[0]}, {plot_paths[1]}")
    print(f"\nMean across quarters:\n{results.groupby('model')[METRICS].mean().round(4).to_string()}")
    print(f"\nSaved {output}")
    return results


if __name__ == "__main__":
    run(parse_args())
