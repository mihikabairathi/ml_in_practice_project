"""Run comparable rolling-quarter recommendation experiments.

Examples (after ``python scripts/make_splits.py``)::

    python scripts/run_experiments.py --models popularity item_knn --split val
    python scripts/run_experiments.py --models lightgbm --split test --max-quarters 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from remy.evaluate import METRICS, evaluate_quarter
from remy.model.item_knn import ItemKNNRecommender
from remy.model.lightgbm_new import LightGBMNew
from remy.model.popularity import Popularity
from remy.paths import PROJECT_ROOT
from remy.splits import SPLIT_DIR, RollingSplits

BACKGROUND = "#EAEAEA"
MINT, PEACH, SAND = "#94D8A1", "#E1C6BA", "#E4D6AB"
MODEL_COLORS = {"popularity": MINT, "lightgbm": PEACH, "item_knn": SAND}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit and evaluate recommenders on shared rolling quarterly splits.",
        epilog="Example: python scripts/run_experiments.py --models popularity item_knn --split val",
    )
    models = parser.add_argument_group("models and evaluation")
    models.add_argument(
        "--models",
        nargs="+",
        choices=("popularity", "item_knn", "lightgbm"),
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

    knn = parser.add_argument_group("item KNN settings")
    knn.add_argument("--k-neighbors", type=int, default=50, help="Neighbors retained per recipe (default: 50)")
    knn.add_argument("--min-rating", type=float, default=5.0, help="Minimum rating treated as positive (default: 5)")

    lgbm = parser.add_argument_group("LightGBM settings")
    lgbm.add_argument(
        "--negative-samples-per-positive", type=int, default=10,
        help="Sampled negatives per training positive (default: 10, matching rolling_lightgbm.ipynb)",
    )
    lgbm.add_argument(
        "--lgbm-candidates", type=int, default=1000,
        help="Popular recipes shortlisted for LightGBM to rank per user (default: 1000)",
    )
    lgbm.add_argument("--boost-rounds", type=int, default=200, help="LightGBM boosting rounds (default: 200)")
    lgbm.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate (default: 0.05)")
    lgbm.add_argument("--num-leaves", type=int, default=31, help="Maximum LightGBM leaves per tree (default: 31)")
    lgbm.add_argument("--min-data-in-leaf", type=int, default=100, help="Minimum rows per LightGBM leaf (default: 100)")
    lgbm.add_argument("--seed", type=int, default=10718, help="Random seed for LightGBM sampling (default: 10718)")

    args = parser.parse_args(argv)
    positive_args = (
        "max_quarters", "k_neighbors", "negative_samples_per_positive",
        "lgbm_candidates", "boost_rounds", "num_leaves", "min_data_in_leaf",
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
    if name == "item_knn":
        return ItemKNNRecommender(args.k_neighbors, args.min_rating)
    if name == "lightgbm":
        return LightGBMNew(
            negatives_per_positive=args.negative_samples_per_positive,
            num_boost_round=args.boost_rounds,
            n_candidates=args.lgbm_candidates,
            seed=args.seed,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
        )
    raise ValueError(f"Unknown model: {name}")


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="white", linewidth=1.1)
    ax.tick_params(axis="both", length=0, pad=7, colors="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.yaxis.label.set_color("#333333")
    ax.xaxis.label.set_color("#333333")
    ax.title.set_color("#333333")


def save_plots(results: pd.DataFrame, output: Path) -> list[Path]:
    """Save styled quarterly trends and clustered metric comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from cycler import cycler
    from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

    models = list(results["model"].drop_duplicates())
    colors = {name: MODEL_COLORS[name] for name in models}
    palette = [MINT, PEACH, SAND]
    saved = []

    with plt.rc_context({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.prop_cycle": cycler(color=palette),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelpad": 9,
        "legend.frameon": False,
        "legend.fontsize": 10,
    }):
        trend_metrics = [
            ("NDCG", "Full-ranking NDCG", True),
            ("HR@10", "Hit rate @10", True),
            ("Recall@10", "Recall @10", True),
        ]
        if results["candidate_recall"].notna().any():
            trend_metrics.append(("candidate_recall", "Candidate recall", True))
        else:
            trend_metrics.append(("scored_users", "Scored users", False))

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, layout="constrained")
        for ax, (metric, title, is_rate) in zip(axes.flat, trend_metrics):
            _style_axes(ax)
            for name in models:
                frame = results.loc[results["model"] == name].sort_values("start")
                if frame[metric].notna().any():
                    ax.plot(
                        frame["start"], frame[metric], marker="o", linewidth=2.2,
                        label=name, color=colors[name],
                    )
            ax.set_title(title)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=not is_rate))
            if is_rate:
                peak = results[metric].max(skipna=True)
                ax.set_ylim(0, min(1, max(0.01, peak * 1.35)))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            else:
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.set_xlabel("Quarter start" if ax.get_subplotspec().is_last_row() else "")
        axes[0, 0].legend(loc="upper right")
        fig.suptitle(f"Rolling {results['split'].iloc[0]} performance", fontsize=16)
        trend_path = output.with_name(f"{output.stem}_quarters.png")
        fig.savefig(trend_path, dpi=160)
        plt.close(fig)
        saved.append(trend_path)

        means = results.groupby("model")[METRICS].mean().reindex(models)

        def comparison_chart(metrics, title, filename, figsize):
            scores = means.loc[:, metrics]
            positions = np.arange(len(metrics))
            width = 0.8 / len(models)
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")
            for index, name in enumerate(models):
                offset = (index - (len(models) - 1) / 2) * width
                bars = ax.bar(
                    positions + offset,
                    scores.loc[name].to_numpy(),
                    width,
                    label=name,
                    color=colors[name],
                    edgecolor="#777777",
                    linewidth=0.6,
                )
                ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=9)
            _style_axes(ax)
            ax.set_title(title)
            ax.set_xlabel("Metric")
            ax.set_xticks(positions, metrics)
            ax.set_xlim(-0.6, len(metrics) - 0.4)
            ax.set_ylim(0, max(0.01, scores.to_numpy().max() * 1.35))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
            ax.legend(loc="upper right")
            chart_path = output.with_name(filename)
            fig.savefig(chart_path, dpi=160)
            plt.close(fig)
            saved.append(chart_path)

        comparison_chart(
            ["HR@10", "HR@100", "Recall@10", "Recall@100"],
            f"Hit rate and recall · {results['split'].iloc[0]}",
            f"{output.stem}_hit_recall.png",
            (10, 5),
        )
        comparison_chart(
            ["NDCG"],
            f"NDCG · {results['split'].iloc[0]}",
            f"{output.stem}_ndcg.png",
            (6, 5),
        )

    return saved


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
            recommendations = model.recommend_many(user_ids)
            row = evaluate_quarter(
                quarter, model.ranking_, model.seen_, recommendations
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

    print(f"\nMean across quarters:\n{results.groupby('model')[METRICS + ['candidate_recall']].mean().round(4).to_string()}")
    print(f"\nSaved {output}")
    return results


if __name__ == "__main__":
    run(parse_args())
