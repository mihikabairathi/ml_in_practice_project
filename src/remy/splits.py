"""Prepare the same historical inputs and next-quarter labels for every model."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from remy.paths import PROJECT_ROOT

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "rolling"
MIN_REVIEWS = 3


@dataclass
class Quarter:
    """A container for one history-to-future prediction task, with no model logic.

    start, end: the outcome window, including start and excluding end.
    history: everyone's reviews strictly before start, with all ratings kept.
    recipes: recipe metadata with submission dates strictly before start.
    users: eligible user_id and n_reviews (at least three reviews before start).
    outcomes: reviews during the window by eligible users, restricted to recipes
        that were available and not already reviewed by that user. All ratings
        are kept; rating == 5 selects positives for ranking evaluation.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    history: pd.DataFrame
    recipes: pd.DataFrame
    users: pd.DataFrame
    outcomes: pd.DataFrame


class RollingSplits:
    """Load the shared split files and provide quarterly training/evaluation data.

    This class keeps the time-filtering rules in one place so every model uses
    the same histories, eligible users, available recipes, and future labels.
    It prepares data only; the caller fits and evaluates their own model.

    Construction:
        splits = RollingSplits()  # after running scripts/make_splits.py
        An optional directory selects another folder containing the three files.

    Stored tables (pandas DataFrames):
        interactions: review events, including the early history before 2003.
        recipes: recipe metadata, including IDs and submission dates.
        windows: quarterly start/end dates and train/val/test roles.
        These tables span the whole experiment; do not pass them directly to fit.

    Two public methods:
        training_quarters(cutoff): earlier tasks whose outcome windows have
            finished by cutoff. Build supervised features from each task's own
            history and labels from its outcomes. Earlier validation/test tasks
            become training tasks once their outcomes are in the past.
        evaluation_quarters(split): held-out tasks for "val" or "test", in order.
            At each task, create a fresh model and train on the available past.
            Use that task's outcomes only for scoring, never for its own fit.

    Both methods return ordinary lists of Quarter objects, so indexing, slicing,
    and repeated iteration work normally. Histories and recipe tables are slices
    of the shared data rather than separately stored copies. Model settings are
    selected on validation and kept fixed throughout the rolling test run.
    """

    def __init__(self, directory=SPLIT_DIR):
        directory = Path(directory)
        self.interactions = pd.read_parquet(directory / "interactions.parquet")
        self.interactions = self.interactions.sort_values("date").reset_index(drop=True)
        self.recipes = pd.read_parquet(directory / "recipes.parquet")
        self.recipes = self.recipes.sort_values("submitted").reset_index(drop=True)
        self.windows = pd.read_parquet(directory / "windows.parquet")
        self.windows = self.windows.sort_values("start").reset_index(drop=True)

    def training_quarters(self, cutoff) -> list[Quarter]:
        """Return completed history/label tasks, ending on or before cutoff.

        For example, training at January 2007 can use October-December 2006
        labels, paired with history before October 2006. It cannot use any
        January-March 2007 labels. The returned list can be indexed or reused.
        """
        cutoff = pd.Timestamp(cutoff)
        completed = self.windows[self.windows["end"] <= cutoff]
        quarters = []
        for window in completed.itertuples(index=False):
            quarters.append(self._make_quarter(window.start, window.end))
        return quarters

    def evaluation_quarters(self, split) -> list[Quarter]:
        """Return the chronological held-out quarters for 'val' or 'test'."""
        if split not in {"val", "test"}:
            raise ValueError("split must be 'val' or 'test'")
        selected = self.windows[self.windows["split"] == split]
        quarters = []
        for window in selected.itertuples(index=False):
            quarters.append(self._make_quarter(window.start, window.end))
        return quarters

    def _make_quarter(self, start, end):
        """Apply the shared history, eligibility, and candidate rules to a window."""
        # The tables are sorted by date. Slice at the cutoff rather than copying
        # the full history/catalog for every Quarter in the returned lists.
        history_end = self.interactions["date"].searchsorted(start)
        outcome_end = self.interactions["date"].searchsorted(end)
        recipes_end = self.recipes["submitted"].searchsorted(start)
        history = self.interactions.iloc[:history_end]
        recipes = self.recipes.iloc[:recipes_end]
        counts = history.groupby("user_id").size().rename("n_reviews")
        users = counts[counts >= MIN_REVIEWS].reset_index()
        outcomes = self.interactions.iloc[history_end:outcome_end]
        outcomes = outcomes[
            outcomes["user_id"].isin(users["user_id"])
            & outcomes["recipe_id"].isin(recipes["id"])
        ]
        # Compare user/recipe pairs to exclude recipes already reviewed by the
        # same user, even if a later version of the dataset has repeat reviews.
        seen = pd.MultiIndex.from_frame(history[["user_id", "recipe_id"]])
        pairs = pd.MultiIndex.from_frame(outcomes[["user_id", "recipe_id"]])
        outcomes = outcomes[~pairs.isin(seen)]
        return Quarter(start, end, history, recipes, users, outcomes)
