from collections.abc import Iterable

import pandas as pd

class Popularity:
    """Recommend recipes by past review count, with no learned user preferences"""

    def fit(self, interactions: pd.DataFrame, recipes=None, training_quarters=()) -> "Popularity":
        """
        Count all historical reviews, regardless of rating.
        Supervised models use training_quarters to build historical input/label pairs.
        This model ignores that argument since we are not training on anything.
        """

        # count recipe reviews, optionally filter to historical recipes, recipes without reviews get a score of zero
        counts = interactions["recipe_id"].value_counts()
        if recipes is not None:
            counts = counts.reindex(recipes["id"], fill_value=0)

        self.scores_ = counts.sort_index().sort_values(ascending=False, kind="stable") # breaks ties by recipe ID
        self.ranking_ = self.scores_.index.tolist()
        self.seen_ = interactions.groupby("user_id")["recipe_id"].agg(set).to_dict()
        return self

    def recommend(self, user_id: int, k: int | None = 10) -> list[int]:
        """Return up to k available, unseen recipe IDs, most popular first."""

        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        if not hasattr(self, "ranking_"):
            raise RuntimeError("Call fit before recommend")
        
        result = []
        seen = self.seen_.get(user_id, set())
        for recipe_id in self.ranking_:
            if k is not None and len(result) == k:
                break
            if recipe_id not in seen:
                result.append(recipe_id)

        return result

    def recommend_batch(
        self, user_ids: Iterable[int], k: int | None = None
    ) -> dict[int, list[int]]:
        """Return each user's unseen recipes in popularity order."""
        if not hasattr(self, "ranking_"):
            raise RuntimeError("Call fit before recommend_batch")
        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        return {user_id: self.recommend(user_id, k=k) for user_id in user_ids}
