"""Charts for saved rolling recommendation experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from remy.evaluate import METRICS

BACKGROUND = "#EAEAEA"
MINT, PEACH, SAND = "#94D8A1", "#E1C6BA", "#E4D6AB"
BLUE, LAVENDER = "#B8CEDB", "#C9BEDA"
MODEL_COLORS = {
    "popularity": MINT,
    "lightgbm": PEACH,
    "knn": SAND,
    "embedding": BLUE,
    "embedding_trained": LAVENDER,
}
MODEL_LABELS = {
    "popularity": "Popularity",
    "lightgbm": "LGBM",
    "knn": "KNN",
    "embedding": "Embedding",
    "embedding_trained": "Trained embedding",
}


def _style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(False, which="both", axis="both")
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
    palette = [MINT, PEACH, SAND, BLUE, LAVENDER]
    saved = []

    with plt.rc_context({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.prop_cycle": cycler(color=palette),
        "axes.grid": False,
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
        trend_metrics.append(("scored_users", "Scored users", False))

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, layout="constrained")
        for ax, (metric, title, is_rate) in zip(axes.flat, trend_metrics):
            _style_axes(ax)
            for name in models:
                frame = results.loc[results["model"] == name].sort_values("start")
                if frame[metric].notna().any():
                    ax.plot(
                        frame["start"], frame[metric], marker="o", linewidth=2.2,
                        label=MODEL_LABELS.get(name, name), color=colors[name],
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
        y_max = max(0.01, means.to_numpy().max() * 1.35)

        def comparison_panel(ax, metrics):
            scores = means.loc[:, metrics]
            positions = np.arange(len(metrics))
            width = 0.7 / len(models)
            for index, name in enumerate(models):
                offset = (index - (len(models) - 1) / 2) * width
                bars = ax.bar(
                    positions + offset,
                    scores.loc[name].to_numpy(),
                    width,
                    label=MODEL_LABELS.get(name, name),
                    color=colors[name],
                    edgecolor="#777777",
                    linewidth=0.6,
                )
                ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=9)
            _style_axes(ax)
            ax.set_xlabel("Metric")
            ax.set_ylabel("Score")
            ax.set_xticks(positions, metrics)
            ax.set_xlim(-0.5, len(metrics) - 0.5)
            ax.set_ylim(0, y_max)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))

        fig, (metrics_ax, ndcg_ax) = plt.subplots(
            1,
            2,
            figsize=(14, 5),
            gridspec_kw={"width_ratios": [4, 2]},
            layout="constrained",
        )
        comparison_panel(metrics_ax, ["HR@10", "HR@100", "Recall@10", "Recall@100"])
        comparison_panel(ndcg_ax, ["NDCG", "NDCG@Candidates"])
        ndcg_ax.legend(loc="upper right")
        title = " vs. ".join(MODEL_LABELS.get(name, name) for name in models)
        fig.suptitle(title, fontsize=20, fontfamily="serif", color="#123744")
        metrics_path = output.with_name(f"{output.stem}_metrics.png")
        fig.savefig(metrics_path, dpi=160)
        plt.close(fig)
        saved.append(metrics_path)

    return saved
