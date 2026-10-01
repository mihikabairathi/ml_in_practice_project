"""Run comparable rolling-quarter recommendation experiments.

Examples (after ``python scripts/make_splits.py``)::

    python scripts/run_experiments.py --models popularity item_knn --split val
    python scripts/run_experiments.py --models lightgbm --split test --max-quarters 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from remy.model.item_knn import ItemKNNRecommender
from remy.model.popularity import Popularity
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
    parser.add_argument("--output", type=Path, help="Results CSV (default: split-dir/experiments_<split>.csv)")
    parser.add_argument("--k-neighbors", type=int, default=50, help="Item KNN neighbors per item")
    parser.add_argument("--min-rating", type=float, default=4.0, help="Item KNN positive rating threshold")
    parser.add_argument("--train-quarters", type=int, default=8, help="Most recent training quarters for LightGBM")
    parser.add_argument("--max-train-users", type=int, default=3000, help="Training users per quarter for LightGBM")
    parser.add_argument("--boost-rounds", type=int, default=200, help="LightGBM boosting rounds")
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
    output = args.output or args.split_dir / f"experiments_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"\nMean across quarters:\n{results.groupby('model')[METRICS].mean().round(4).to_string()}")
    print(f"\nSaved {output}")
    return results


if __name__ == "__main__":
    run(parse_args())
