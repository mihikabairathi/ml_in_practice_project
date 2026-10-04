import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from remy.splits import MIN_REVIEWS

USER_FEATURES = ["user_n_reviews", "user_mean_rating", "user_pos_rate"]


class LightGBMNew:
    """
    Simple LightGBM baseline using user statistics and binary recipe tags.

    Training examples come from completed historical quarters. For each user with
    a new 5-star outcome in a quarter, those recipes are positive examples and we
    sample random recipes that were available and unseen at the quarter start as
    negative examples.

    At recommendation time the fitted model scores every available recipe the user has not previously reviewed. 
    Recipe popularity is kept as a fallback for users with fewer than MIN_REVIEWS historical reviews.
    """

    def __init__(self, negatives_per_positive=5, num_boost_round=200, n_candidates=1000, seed=10718):
        self.negatives_per_positive = negatives_per_positive
        self.num_boost_round = num_boost_round
        self.n_candidates = n_candidates
        self.seed = seed
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "verbosity": -1,
            "seed": seed
        }

    def fit(self, interactions: pd.DataFrame, recipes=None, training_quarters=()) -> "LightGBMNew":
        """
        Fit from completed historical quarters and store the current state.

        interactions and recipes describe what is known at the current prediction
        cutoff. Each training quarter's history creates user features, and its
        outcomes determine positive examples.
        """
        if recipes is None:
            raise ValueError("recipes with a 'tags' column are required")
        elif not training_quarters:
            raise ValueError("At least one completed training quarter is required")

        # One tag vocabulary for the current catalog (works for historical ones too)
        tags = recipes["tags"].map(self.as_tag_list)
        self.tag_encoder_ = MultiLabelBinarizer(sparse_output=True)
        self.recipe_tags_ = self.tag_encoder_.fit_transform(tags).astype(np.float32).tocsr()
        self.recipe_tag_row_ = pd.Series(np.arange(len(recipes)), index=recipes["id"])
        self.feature_names_ = USER_FEATURES + [f"tag_{tag}" for tag in self.tag_encoder_.classes_]

        # Current state used at recommendation time
        self.user_features_ = self._user_features(interactions)
        self.seen_ = interactions.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self.available_recipes_ = recipes["id"].tolist()

        # Popularity is fallback behavior
        counts = interactions["recipe_id"].value_counts().reindex(recipes["id"], fill_value=0)
        counts = counts.sort_index().sort_values(ascending=False, kind="stable")
        self.ranking_ = counts.index.tolist()

        # Ranking over all recipes takes too long, so we limit the candidates to the top n_candidates by popularity
        self.candidates_ = self.ranking_[:self.n_candidates]

        X_parts, y_parts = [], []
        rng = np.random.default_rng(self.seed)
        for quarter in list(training_quarters):
            positives = quarter.outcomes.loc[quarter.outcomes["rating"] == 5, ["user_id", "recipe_id"]].drop_duplicates()
            if positives.empty:
                continue

            q_users = self._user_features(quarter.history)
            seen = quarter.history.groupby("user_id")["recipe_id"].agg(set).to_dict()
            available = np.asarray(quarter.recipes["id"].tolist())

            rows, labels = [], []
            for user_id, user_pos in positives.groupby("user_id"):
                pos_ids = user_pos["recipe_id"].tolist()
                rows.extend((user_id, recipe_id) for recipe_id in pos_ids)
                labels.extend([1] * len(pos_ids))

                blocked = seen.get(user_id, set()) | set(pos_ids)
                n_negative = self.negatives_per_positive * len(pos_ids)
                neg_ids = set()

                while len(neg_ids) < n_negative:
                    recipe_id = rng.choice(available)
                    if recipe_id not in blocked:
                        neg_ids.add(recipe_id)

                rows.extend((user_id, recipe_id) for recipe_id in neg_ids)
                labels.extend([0] * len(neg_ids))

            if not rows:
                continue

            examples = pd.DataFrame(rows, columns=["user_id", "recipe_id"])
            X_parts.append(self._features(examples, q_users))
            y_parts.append(np.asarray(labels, dtype=np.int8))

        if not X_parts:
            raise ValueError("Training quarters contain no interactions")

        X_train = sparse.vstack(X_parts, format="csr")
        y_train = np.concatenate(y_parts)

        if np.unique(y_train).size < 2:
            raise ValueError("Training data must contain both positive and negative examples")

        train = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names_,)
        self.model_ = lgb.train(self.params, train, num_boost_round=self.num_boost_round)

        return self

    def recommend_many(self, user_ids, k=None) -> dict[int, list[int]]:
        """
        Rank the shortlisted unseen recipes for each requested user.
        For users with fewer than MIN_REVIEWS historical reviews, return the
        popularity fallback instead of running LightGBM.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before recommend")
        return {user_id: self._recommend_one(user_id, k) for user_id in user_ids}

    def _recommend_one(self, user_id: int, k=None) -> list[int]:
        seen = self.seen_.get(user_id, set())
        unseen = [recipe_id for recipe_id in self.candidates_ if recipe_id not in seen]

        if user_id not in self.user_features_.index or len(seen) < MIN_REVIEWS:
            ranked = [recipe_id for recipe_id in self.ranking_ if recipe_id not in seen]
        else:
            pairs = pd.DataFrame({"user_id": user_id, "recipe_id": unseen})
            scores = self.model_.predict(self._features(pairs, self.user_features_))
            ranked = [unseen[i] for i in np.lexsort((np.asarray(unseen), -scores))]
        return ranked if k is None else ranked[:k]

    @staticmethod
    def _user_features(history):
        by_user = history.groupby("user_id")
        rated = history[history["rating"] > 0]

        users = pd.DataFrame({
            "user_n_reviews": by_user.size(),
            "user_mean_rating": rated.groupby("user_id")["rating"].mean(),
            "user_pos_rate": (rated["rating"] == 5).groupby(rated["user_id"]).mean()
        })
        users[["user_mean_rating", "user_pos_rate"]] = users[["user_mean_rating", "user_pos_rate"]].fillna(0.0)
        return users

    def _features(self, pairs: pd.DataFrame, users: pd.DataFrame) -> sparse.csr_matrix:
        user_x = users.reindex(pairs["user_id"])[USER_FEATURES].to_numpy(np.float32)
        recipe_rows = self.recipe_tag_row_.reindex(pairs["recipe_id"])

        return sparse.hstack(
            [sparse.csr_matrix(user_x), self.recipe_tags_[recipe_rows.to_numpy(dtype=int)]], 
            format="csr"
        )

    @staticmethod
    def as_tag_list(value):
        """Normalize the parquet tag value to a list without changing tag content."""
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        return list(value) if not isinstance(value, str) else [value]