"""Evaluate the zero-shot LLM recommender on the rolling val and test quarters.

Needs a GPU and vLLM (pip install -e ".[llm]"). Example, inside a SLURM job::

    python scripts/run_llm_baseline.py --run-dir artifacts/llm_qwen3-8b

Popularity is scored on the same quarters as a reference. Results are written
after every quarter, and rerunning with the same --run-dir skips quarters that
are already done, so a preempted job can simply be resubmitted. Outputs:

    <run-dir>/results.csv             one row per (split, quarter, model)
    <run-dir>/summary.csv             mean metrics per split and model
    <run-dir>/responses/*.jsonl       raw LLM answers per user, for inspection
    <run-dir>/<split>_results_*.png   the usual charts from remy.plots

The LLM returns only 10 recipes per user; the evaluator ranks everything after
those by popularity. So @10 metrics (and NDCG@10, added here) describe the
LLM, while HR@100, Recall@100 and full NDCG mostly describe that fallback.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from remy.evaluate import METRICS, evaluate_quarter
from remy.model.llm import LLMRecommender
from remy.model.popularity import Popularity
from remy.paths import PROJECT_ROOT
from remy.plots import save_plots
from remy.splits import SPLIT_DIR, RollingSplits

K = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "llm_baseline",
                        help="Output folder; reusing it resumes an interrupted run")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B", help="Hugging Face model id")
    parser.add_argument("--n-candidates", type=int, default=200, help="Recipes shown per prompt")
    parser.add_argument("--n-history", type=int, default=30, help="Past liked recipes shown per prompt")
    parser.add_argument("--max-quarters", type=int, help="Only the first N quarters of each split")
    parser.add_argument("--max-users", type=int,
                        help="Score a random sample of N users per quarter (quick tests)")
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    return parser.parse_args(argv)


def sample_users(quarter, max_users, seed=0):
    """Keep only a random sample of the quarter's scorable users (for quick tests)."""
    positives = quarter.outcomes[quarter.outcomes["rating"] == 5]
    users = positives["user_id"].unique()
    if max_users is None or len(users) <= max_users:
        return quarter
    keep = np.random.default_rng(seed).choice(users, size=max_users, replace=False)
    outcomes = quarter.outcomes[quarter.outcomes["user_id"].isin(keep)]
    return dataclasses.replace(quarter, outcomes=outcomes)


def score(quarter, model, recommendations):
    """The shared metrics, plus NDCG@10 (the shared NDCG cut at the LLM's list length)."""
    row = evaluate_quarter(quarter, model.ranking_, model.seen_, recommendations)
    at_k = evaluate_quarter(quarter, model.ranking_, model.seen_, recommendations,
                            ks=(K,), candidate_k=K)
    row[f"NDCG@{K}"] = at_k["NDCG@Candidates"]
    return row


def run(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "responses").mkdir(exist_ok=True)
    results_path = args.run_dir / "results.csv"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    done = {(row["split"], str(row["start"])[:10]) for row in rows if row["model"] == "llm"}
    splits = RollingSplits(args.split_dir)

    for split in args.splits:
        quarters = splits.evaluation_quarters(split)[: args.max_quarters]
        for quarter in quarters:
            start = str(quarter.start.date())
            if (split, start) in done:
                print(f"skip {split} {start} (already in {results_path})", flush=True)
                continue
            quarter = sample_users(quarter, args.max_users)
            user_ids = quarter.outcomes.loc[quarter.outcomes["rating"] == 5, "user_id"].unique()

            popularity = Popularity().fit(quarter.history, recipes=quarter.recipes)
            pop_row = score(quarter, popularity, None)

            llm = LLMRecommender(args.model_name, args.n_candidates, args.n_history, K)
            llm.fit(quarter.history, recipes=quarter.recipes)
            started = pd.Timestamp.now()
            recommendations = llm.recommend_batch(user_ids)
            seconds = (pd.Timestamp.now() - started).total_seconds()
            llm_row = score(quarter, llm, recommendations)
            # How often the LLM itself supplied all 10 ids (vs. popularity filling gaps).
            n_valid = np.array([r["n_valid"] for r in llm.responses_])
            llm_row.update(llm_seconds=seconds, llm_mean_valid_ids=n_valid.mean(),
                           llm_share_all_valid=(n_valid == K).mean())

            with open(args.run_dir / "responses" / f"{split}_{start}.jsonl", "w") as f:
                for response in llm.responses_:
                    f.write(json.dumps(response) + "\n")
            for model_name, row in [("popularity", pop_row), ("llm", llm_row)]:
                rows.append({"split": split, "model": model_name, **row})
            pd.DataFrame(rows).to_csv(results_path, index=False)
            print(f"{split} {start}  users={len(user_ids)}  {seconds:.0f}s  "
                  f"HR@10 llm={llm_row['HR@10']:.3f} pop={pop_row['HR@10']:.3f}  "
                  f"all-10-valid={llm_row['llm_share_all_valid']:.2f}", flush=True)

    results = pd.DataFrame(rows)
    results["start"] = pd.to_datetime(results["start"])
    summary_cols = ["HR@10", "Recall@10", f"NDCG@{K}"] + [m for m in METRICS if not m.endswith("@10")]
    summary = results.groupby(["split", "model"])[summary_cols].mean().round(4)
    summary.to_csv(args.run_dir / "summary.csv")
    for split, split_results in results.groupby("split"):
        save_plots(split_results, args.run_dir / f"{split}_results.csv")
    print(f"\nMean across quarters:\n{summary.to_string()}\n\nSaved {args.run_dir}")


if __name__ == "__main__":
    run(parse_args())
