"""Item-to-item collaborative filtering from positive historical reviews."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


class KNNRecommender:
    """Cosine item neighbors, summed over a user's liked recipes.

    Only ratings at or above ``min_rating`` form the user-item matrix. All
    historical reviews count as seen. At recommendation time, KNN and LightGBM
    use the same top ``n_candidates`` popularity shortlist; KNN ranks that list
    with collaborative-filter scores. ``k_neighbors`` controls the item graph,
    not the number of recommendations. ``training_quarters`` is accepted for
    the shared model interface; this baseline learns directly from current history.
    """

    def __init__(
        self,
        k_neighbors: int = 50,
        min_rating: float = 5.0,
        n_candidates: int = 1000,
    ):
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        if n_candidates < 1:
            raise ValueError("n_candidates must be positive")
        self.k_neighbors = k_neighbors
        self.min_rating = min_rating
        self.n_candidates = n_candidates

    def fit(
        self, interactions: pd.DataFrame, recipes=None, training_quarters=()
    ) -> "KNNRecommender":
        history = interactions
        catalog = (
            pd.Index(recipes["id"].unique()) if recipes is not None
            else pd.Index(history["recipe_id"].unique())
        )
        catalog = catalog.sort_values()
        history = history[history["recipe_id"].isin(catalog)]

        self.scores_ = (
            history["recipe_id"].value_counts().reindex(catalog, fill_value=0)
            .sort_index().sort_values(ascending=False, kind="stable")
        )
        self.ranking_ = self.scores_.index.tolist()
        self.candidates_ = self.ranking_[:self.n_candidates]
        self.seen_ = history.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self.idx_to_item_ = catalog.to_numpy()
        self.item_to_idx_ = {item: i for i, item in enumerate(self.idx_to_item_)}

        positive = history.loc[history["rating"] >= self.min_rating, ["user_id", "recipe_id"]].drop_duplicates()
        users = pd.Index(positive["user_id"].unique())
        self.user_to_idx_ = {user: i for i, user in enumerate(users)}
        rows = positive["user_id"].map(self.user_to_idx_).to_numpy(dtype=np.int64)
        cols = positive["recipe_id"].map(self.item_to_idx_).to_numpy(dtype=np.int64)
        self.R_ = sparse.csr_matrix(
            (np.ones(len(positive), dtype=np.float32), (rows, cols)),
            shape=(len(users), len(catalog)),
        )
        if len(positive):
            normalized = normalize(self.R_.tocsc(), norm="l2", axis=0)
            similarity = (normalized.T @ normalized).tocsr()
            similarity.setdiag(0)
            similarity.eliminate_zeros()
            self.S_ = self._top_k_sparse(similarity, self.k_neighbors)
        else:
            self.S_ = sparse.csr_matrix((len(catalog), len(catalog)), dtype=np.float32)
        return self

    @staticmethod
    def _top_k_sparse(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
        """Keep each item's highest-scoring neighbors, breaking ties by item ID."""
        data, indices, indptr = [], [], [0]
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            values = matrix.data[start:end]
            columns = matrix.indices[start:end]
            order = np.lexsort((columns, -values))[:k]
            data.extend(values[order])
            indices.extend(columns[order])
            indptr.append(len(data))
        return sparse.csr_matrix(
            (data, indices, indptr), shape=matrix.shape, dtype=matrix.dtype
        )

    def _candidate_ranking(self, user_id: int) -> list[int]:
        """Rank the shared popularity shortlist by collaborative-filter score."""
        if not hasattr(self, "S_"):
            raise RuntimeError("Call fit before recommend")
        seen = self.seen_.get(user_id, set())
        candidates = [
            recipe_id for recipe_id in self.candidates_ if recipe_id not in seen
        ]
        if user_id not in self.user_to_idx_:
            return candidates

        scores = (self.R_[self.user_to_idx_[user_id]] @ self.S_).tocsr()
        score_by_item = dict(zip(scores.indices, scores.data))
        candidate_scores = np.fromiter(
            (
                score_by_item.get(self.item_to_idx_[recipe_id], 0.0)
                for recipe_id in candidates
            ),
            dtype=np.float32,
            count=len(candidates),
        )
        # The candidate list starts in popularity order, so stable sorting
        # keeps popularity as the tie-break for recipes with equal CF scores.
        order = np.argsort(-candidate_scores, kind="stable")
        return [candidates[i] for i in order]

    def recommend(self, user_id: int, k: int | None = 10) -> list[int]:
        """Return ranked unseen items, using popularity after CF candidates.

        With ``k=None``, return the full shared popularity shortlist, ranked
        by collaborative-filter scores. The evaluator appends the remaining
        items in popularity order.
        """
        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        if k == 0:
            return []
        candidates = self._candidate_ranking(user_id)
        if k is None:
            return candidates
        result = candidates[:k]
        chosen = set(result)
        seen = self.seen_.get(user_id, set())
        for recipe_id in self.ranking_:
            if len(result) == k:
                break
            if recipe_id not in seen and recipe_id not in chosen:
                result.append(recipe_id)
        return result

    def recommend_batch(
        self, user_ids: Iterable[int], k: int | None = None
    ) -> dict[int, list[int]]:
        """Return candidate rankings for multiple users with one API call."""
        if not hasattr(self, "S_"):
            raise RuntimeError("Call fit before recommend_batch")
        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        return {user_id: self.recommend(user_id, k=k) for user_id in user_ids}
