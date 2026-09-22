"""Example model interface only; recommendation logic is not implemented."""

import pandas as pd


class Popularity:
    """Recommend the globally most popular recipes to every user."""

    def fit(self, interactions: pd.DataFrame) -> "Popularity":
        """Learn a recipe ranking from training interactions.

        Input: DataFrame with user_id, recipe_id, and rating columns.
        Output: self, so callers can write model = Popularity().fit(train).

        TODO: Decide what counts as popularity, count interactions per recipe,
        and store the sorted recipe IDs on this instance.
        """
        raise NotImplementedError

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return up to k original recipe IDs, most popular first.

        Input: a user ID and the requested number of recommendations.
        Output example: [123, 456, 789] for k=3.
        This baseline ignores user_id and returns the same ranking to everyone.

        TODO: Return the first k recipe IDs from the learned ranking.
        """
        raise NotImplementedError
