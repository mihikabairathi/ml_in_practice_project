"""Popularity by past review count, with available/previously unseen candidates."""

import pandas as pd


class Popularity:
    """Recommend recipes by past review count, with no learned user preferences.

    fit stores scores_ (review counts), ranking_ (recipe IDs in score order),
    and seen_ (previously reviewed recipe IDs per user). recommend uses the
    same ranking for everyone but skips recipes that user has already reviewed.
    """

    def fit(
        self, interactions: pd.DataFrame, *, recipes=None, training_quarters=()
    ) -> "Popularity":
        """Count all historical reviews, regardless of rating; reset on every fit.

        Pass the current Quarter's history and available recipe metadata.
        Recipes with no reviews get score zero; ties use ascending recipe ID.
        Without recipes, use only IDs observed in interactions, preserving the
        original fit(interactions) interface. Supervised models loop over
        training_quarters to build historical input/label pairs; counting
        popularity does not need labels, so this model ignores that argument.
        """
        counts = interactions["recipe_id"].value_counts()
        if recipes is not None:
            counts = counts.reindex(recipes["id"], fill_value=0)
        self.scores_ = counts.sort_index().sort_values(ascending=False, kind="stable")
        self.ranking_ = self.scores_.index.tolist()
        self.seen_ = interactions.groupby("user_id")["recipe_id"].agg(set).to_dict()
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return up to k available, unseen recipe IDs, most popular first."""
        if k < 0:
            raise ValueError("k must be nonnegative")
        if not hasattr(self, "ranking_"):
            raise RuntimeError("Call fit before recommend")
        result = []
        seen = self.seen_.get(user_id, set())
        for recipe_id in self.ranking_:
            if len(result) == k:
                break
            if recipe_id not in seen:
                result.append(recipe_id)
        return result
