"""Tune recipe rerankers on rolling validation and score finalists on test."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from remy.evaluate import METRICS, evaluate_quarter
from remy.model.lightgbm_features import ALL_ROUND1, LightGBM
from remy.model.random_forest import RandomForest
from remy.model.xgboost import XGBoost
from remy.splits import MIN_REVIEWS, RollingSplits

CONFIGS = {
    "lightgbm": (LightGBM, {}),
    "xgb_depth3": (XGBoost, dict(max_depth=3, n_estimators=120, learning_rate=0.08)),
    "xgb_depth6": (XGBoost, dict(max_depth=6, n_estimators=120, learning_rate=0.08)),
    "rf_depth8": (RandomForest, dict(max_depth=8, n_estimators=32, min_samples_leaf=40)),
    "rf_depth14": (RandomForest, dict(max_depth=14, n_estimators=32, min_samples_leaf=20)),
}


def run_quarter(splits, quarter, names, threads):
    context = LightGBM(feature_groups=ALL_ROUND1, num_threads=threads)
    X, y, groups = context._prepare_training(
        quarter.history, quarter.recipes, splits.training_quarters(quarter.start), quarter.start
    )
    print(f"{quarter.start.date()} prepared {X.shape} with {int(y.sum())} positives", flush=True)
    models = {}
    for name in names:
        cls, params = CONFIGS[name]
        model = context if cls is LightGBM else cls(num_threads=threads, **params)
        if model is not context:
            # The feature columns and row encoding are shared across estimators.
            model.dense_cols_ = context.dense_cols_
        start = perf_counter()
        model._fit_estimator(X, y, groups)
        models[name] = model
        print(f"  {name} fit in {perf_counter() - start:.1f}s", flush=True)
    del X, y, groups

    recommendations = {name: {} for name in names}
    users = quarter.outcomes.loc[quarter.outcomes.rating == 5, "user_id"].unique()
    # Omitted users receive the evaluator's popularity fallback.
    users = [u for u in users if len(context.seen_[u]) >= MIN_REVIEWS]
    for offset in range(0, len(users), 32):
        batch = users[offset:offset + 32]
        candidates = [
            np.asarray([r for r in context.candidates_ if r not in context.seen_[u]], dtype=int)
            for u in batch
        ]
        lengths = [len(ids) for ids in candidates]
        if not sum(lengths):
            continue
        features = context._features(np.repeat(batch, lengths), np.concatenate(candidates), context.state_)
        boundaries = np.cumsum(lengths)[:-1]
        for name, model in models.items():
            for user, ids, scores in zip(batch, candidates, np.split(model._score(features), boundaries)):
                recommendations[name][user] = ids[np.lexsort((ids, -scores))].tolist()

    for name, recs in recommendations.items():
        result = evaluate_quarter(quarter, context.ranking_, context.seen_, recs, candidate_k=context.n_candidates)
        print(f"  {name} NDCG={result['NDCG']:.6f}", flush=True)
        yield dict(model=name, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth quarter in each split.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stride < 1 or args.threads < 1:
        parser.error("--stride and --threads must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    splits = RollingSplits()
    rows = []
    names = list(CONFIGS)
    for split in ("val", "test"):
        for quarter in splits.evaluation_quarters(split)[::args.stride]:
            rows.extend(dict(split=split, **row) for row in run_quarter(splits, quarter, names, args.threads))
            pd.DataFrame(rows).to_csv(args.out / "quarterly.csv", index=False)
        summary = pd.DataFrame(rows).groupby(["split", "model"], sort=False)[METRICS].mean()
        summary.to_csv(args.out / "summary.csv")
        if split == "val":
            ndcg = summary.loc["val", "NDCG"]
            names = ["lightgbm"] + [
                ndcg.loc[[name for name in CONFIGS if name.startswith(prefix)]].idxmax()
                for prefix in ("xgb_", "rf_")
            ]
    print(summary.to_string())


if __name__ == "__main__":
    main()
