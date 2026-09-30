"""Save the common quarterly recommendation tasks in data/processed/rolling/.

Run after download_data.py: python scripts/make_splits.py

At each quarter start, inputs are all earlier reviews and recipe metadata;
outcomes are the following calendar quarter. Users need >=3 earlier reviews
of ANY rating, including unrated (0). Outcome recipes must already exist and
must not have been reviewed by that user. Keep all ratings; 5 stars defines
a positive for ranking evaluation, not eligibility.

Training tasks start on 2003-01-01. The sparse opening years remain historical
context: by January 2003 there are 23,934 reviews and 1,252 eligible users,
with 455-759 evaluable users per quarter that year (versus 67-352 in 2002).
The team's first evaluation checkpoint is 2007-01-01: validation covers
2007-2010 and test covers 2011-2015, before the increasingly cold-user tail.
Fixed calendar quarters preserve a consistent prediction horizon despite
varying traffic. These are explainable calendar choices, not optimized dates.

These are NOT three permanently separate interaction tables. At a checkpoint,
supervised training uses earlier quarterly tasks whose outcome windows have
finished, with each task's own historical inputs. Refit at every checkpoint;
select settings on validation and freeze settings throughout test. Completed
validation/test outcomes become training data at subsequent checkpoints.
Keep all past reviews for history/popularity, even from ineligible users.

Write interactions and recipe metadata once, plus windows.parquet containing
the 52 boundaries, roles, and counts. remy.splits reconstructs the tasks;
notebooks/rolling_popularity.ipynb shows the loop. Files are gitignored.
"""

import pandas as pd

from remy.paths import PROJECT_ROOT
from remy.splits import RollingSplits, SPLIT_DIR


def main():
    raw = PROJECT_ROOT / "data" / "raw"
    interactions = pd.read_parquet(
        raw / "interactions.parquet", columns=["user_id", "recipe_id", "date", "rating"]
    )
    recipes = pd.read_parquet(raw / "recipes.parquet")
    if interactions.isna().any().any() or recipes[["id", "submitted"]].isna().any().any():
        raise ValueError("Missing interaction values or recipe IDs/submission dates")
    if not recipes["id"].is_unique or not interactions["recipe_id"].isin(recipes["id"]).all():
        raise ValueError("Recipe IDs must be unique and cover every interaction")

    stop = pd.Timestamp("2016-01-01")
    interactions = interactions[interactions["date"] < stop].sort_values(
        ["date", "user_id", "recipe_id"]
    ).reset_index(drop=True)
    recipes = recipes[recipes["submitted"] < stop].sort_values("id").reset_index(drop=True)
    boundaries = pd.date_range("2003-01-01", stop, freq="QS")
    windows = pd.DataFrame({"start": boundaries[:-1], "end": boundaries[1:]})
    windows["split"] = "train"
    windows.loc[windows["start"] >= "2007-01-01", "split"] = "val"
    windows.loc[windows["start"] >= "2011-01-01", "split"] = "test"

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(SPLIT_DIR / "interactions.parquet", index=False)
    recipes.to_parquet(SPLIT_DIR / "recipes.parquet", index=False)
    windows.to_parquet(SPLIT_DIR / "windows.parquet", index=False)

    splits = RollingSplits()
    summaries = []
    for quarter in splits.training_quarters(stop):
        positives = quarter.outcomes[quarter.outcomes["rating"] == 5]
        counts = {
            "history_reviews": len(quarter.history),
            "eligible_users": len(quarter.users),
            "outcome_reviews": len(quarter.outcomes),
            "positive_reviews": len(positives),
            "positive_users": positives["user_id"].nunique(),
        }
        summaries.append(counts)
    windows = pd.concat([windows, pd.DataFrame(summaries)], axis=1)

    windows.to_parquet(SPLIT_DIR / "windows.parquet", index=False)
    print(windows.groupby("split", sort=False)[
        ["outcome_reviews", "positive_reviews", "positive_users"]
    ].sum().rename(columns={"positive_users": "positive_user_quarters"}).to_string())
    print(f"Saved {len(windows)} quarterly tasks to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
