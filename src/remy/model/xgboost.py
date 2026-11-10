"""XGBoost recipe reranking with popularity, taste, and MiniLM features."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from remy.model.embedding import EMBEDDING_DIR
from remy.model.lightgbm_features import ALL_GROUPS, BASELINE_GROUPS, GROUP_COLS, LightGBM
from remy.model.popularity import Popularity
from remy.splits import MIN_REVIEWS

POPULARITY_GROUPS = ("recipe_pop", "pop_season", "pop_trend", "pop_5star")
TASTE_GROUPS = ("user_taste", "inter_tag")
XGBOOST_GROUPS = BASELINE_GROUPS + POPULARITY_GROUPS + TASTE_GROUPS


class _RecipeContent:
    fields = ("name", "ingredients")
    columns = ("name_profile_cosine", "ingredients_profile_cosine", "recent_best_cosine")

    def __init__(self):
        ids = np.load(EMBEDDING_DIR / "id.npy")
        self.rows = pd.Series(np.arange(len(ids), dtype=np.int32), index=ids)
        self.raw = [np.load(EMBEDDING_DIR / f"{field}.npy", mmap_mode="r") for field in self.fields]
        self.normalized = [normalize(np.asarray(vectors, dtype=np.float32)) for vectors in self.raw]

    def profile(self, history, available):
        catalog = self.rows.reindex(available).dropna().to_numpy(dtype=int)
        liked = history.loc[history.rating.ge(4) & history.recipe_id.isin(self.rows.index)]
        liked = liked.sort_values("date", kind="stable").drop_duplicates(["user_id", "recipe_id"], keep="last")
        users = pd.Index(liked.user_id.unique())
        user_rows = users.get_indexer(liked.user_id)
        recipe_rows = self.rows.loc[liked.recipe_id].to_numpy()
        weights = np.where(liked.rating.eq(5), 1.0, 0.5).astype(np.float32)
        matrix = sparse.csr_matrix((weights, (user_rows, recipe_rows)), shape=(len(users), len(self.rows)))
        totals = np.asarray(matrix.sum(axis=1))
        means = [vectors[catalog].mean(axis=0) for vectors in self.raw]
        profiles = [
            normalize(matrix @ vectors - totals * mean) if len(users) else np.empty((0, len(mean)))
            for vectors, mean in zip(self.normalized, means)
        ]
        recent = np.full((len(users), 3), -1, dtype=np.int32)
        last = liked.iloc[::-1].groupby("user_id", sort=False).cumcount().iloc[::-1].to_numpy()
        keep = last < 3
        recent[user_rows[keep], last[keep]] = recipe_rows[keep]
        return users, means, profiles, recent

    @staticmethod
    def _unit(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, np.finfo(np.float32).tiny)

    def pair_features(self, user_ids, recipe_ids, state):
        users, means, profiles, recent = state
        user_rows = users.get_indexer(user_ids)
        recipe_rows = self.rows.reindex(recipe_ids).fillna(-1).to_numpy(dtype=int)
        output = np.full((len(user_ids), len(self.columns)), np.nan, dtype=np.float32)
        valid = np.flatnonzero((user_rows >= 0) & (recipe_rows >= 0))
        for offset in range(0, len(valid), 8192):
            positions = valid[offset:offset + 8192]
            u = user_rows[positions]
            r = recipe_rows[positions]
            recent_rows = recent[u]
            similarities = np.zeros(recent_rows.shape, dtype=np.float32)
            for col, (vectors, mean, profile) in enumerate(zip(self.normalized, means, profiles)):
                candidate = self._unit(vectors[r] - mean)
                output[positions, col] = np.einsum("ij,ij->i", candidate, profile[u])
                for index in range(recent_rows.shape[1]):
                    previous = self._unit(vectors[recent_rows[:, index]] - mean)
                    similarities[:, index] += 0.5 * np.einsum("ij,ij->i", candidate, previous)
            similarities[recent_rows < 0] = -np.inf
            output[positions, 2] = similarities.max(axis=1)
        return output



class _TreeReranker(LightGBM):
    """Reuse the recipe features and historical sampling from LightGBM."""

    def __init__(self, feature_groups, seed, num_threads):
        super().__init__(feature_groups=feature_groups, seed=seed, num_threads=num_threads)
        self.num_threads = num_threads

    def fit(self, interactions, recipes=None, training_quarters=(), cutoff=None):
        quarters = list(training_quarters)
        if recipes is None or not quarters:
            raise ValueError("Recipes and completed training quarters are required")
        self.__dict__.pop("model_", None)
        recipes = recipes.drop_duplicates("id").reset_index(drop=True)
        self.date_col_ = self._find_date_col(interactions)
        self.cutoff_ = pd.Timestamp(cutoff) if cutoff is not None else (
            pd.to_datetime(interactions[self.date_col_]).max() + pd.Timedelta(days=1)
        )

        self.recipe_ids_ = recipes.id.to_numpy()
        self.recipe_row_ = pd.Series(np.arange(len(recipes)), index=recipes.id)
        self._catalog_set = set(self.recipe_ids_)
        tags = recipes.tags.map(self.as_tag_list)
        encoder = MultiLabelBinarizer(sparse_output=True)
        self.recipe_tags_ = encoder.fit_transform(tags).astype(np.float32).tocsr()
        self.recipe_tags_norm_ = normalize(self.recipe_tags_)
        self.recipe_n_tags_ = np.maximum(np.asarray(self.recipe_tags_.sum(axis=1)).ravel(), 1)
        self.recipe_static_ = self._static(recipes, tags)
        self.dense_cols_ = [c for group in ALL_GROUPS if group in self.feature_groups for c in GROUP_COLS[group]]
        self.use_tags_ = "tags" in self.feature_groups
        if not hasattr(self, "content_"):
            self.content_ = _RecipeContent()
        self.submitted_ = pd.to_datetime(recipes.set_index("id").submitted)
        self.numeric_width_ = len(self.dense_cols_) + len(self.content_.columns)

        popularity = Popularity().fit(interactions, recipes)
        self.ranking_, self.seen_ = popularity.ranking_, popularity.seen_
        self.candidates_ = self.ranking_[:self.n_candidates]
        self.state_ = self._state(interactions, self.cutoff_)
        self.user_features_ = self.state_["user"]

        rng = np.random.default_rng(self.seed)
        examples = [part for quarter in quarters if (part := self._quarter_examples(quarter, rng)) is not None]
        if not examples:
            raise ValueError("Training quarters contain no examples")
        features, labels, _ = zip(*examples)
        y = np.concatenate(labels)
        if np.unique(y).size < 2:
            raise ValueError("Training data must contain both classes")
        X = sparse.vstack(features, format="csr")
        del examples, features, labels
        self.model_ = self._train(X, y)
        return self

    def _state(self, history, cutoff):
        state = super()._state(history, cutoff)
        available = self.submitted_.index[self.submitted_.lt(cutoff)]
        state["content"] = self.content_.profile(history, available)
        return state

    def _features(self, users, recipes, state):
        features = super()._features(users, recipes, state)
        extra = sparse.csr_matrix(self.content_.pair_features(users, recipes, state["content"]))
        width = len(self.dense_cols_)
        return sparse.hstack([features[:, :width], extra, features[:, width:]], format="csr")

    def recommend(self, user_id, k=10):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before recommend")
        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        seen = self.seen_.get(user_id, set())
        if len(seen) < MIN_REVIEWS:
            ranked = [recipe for recipe in self.ranking_ if recipe not in seen]
        else:
            recipes = np.asarray([recipe for recipe in self.candidates_ if recipe not in seen])
            ranked = []
            if len(recipes):
                features = self._features([user_id] * len(recipes), recipes, self.state_)
                ranked = recipes[np.lexsort((recipes, -self._score(features)))].tolist()
        return ranked if k is None else ranked[:k]


class XGBoost(_TreeReranker):
    """The depth-five configuration selected by rolling validation Recall@10."""

    def __init__(self, seed=10718, num_threads=4):
        super().__init__(XGBOOST_GROUPS, seed, num_threads)

    def _matrix(self, features):
        # Preserve numeric zero separately from NaN in XGBoost's sparse input.
        width = self.numeric_width_
        values = features[:, :width].toarray()
        columns = np.tile(np.arange(width, dtype=np.int32), len(values))
        offsets = np.arange(len(values) + 1, dtype=np.int64) * width
        numeric = sparse.csr_matrix((values.ravel(), columns, offsets), shape=values.shape)
        return sparse.hstack([numeric, features[:, width:]], format="csr")

    def _train(self, X, y):
        import xgboost as xgb

        matrix = xgb.QuantileDMatrix(self._matrix(X), label=y, nthread=self.num_threads)
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "nthread": self.num_threads,
            "seed": self.seed,
            "max_depth": 5,
            "eta": 0.0822948425999282,
            "min_child_weight": 5.7853773583089464,
            "subsample": 0.8853370472924392,
            "colsample_bytree": 0.6526173563026589,
            "lambda": 1.0971159919250149,
        }
        return xgb.train(params, matrix, num_boost_round=120)

    def _score(self, features):
        return self.model_.inplace_predict(self._matrix(features))
