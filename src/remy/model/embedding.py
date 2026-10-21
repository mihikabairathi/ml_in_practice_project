import json

import numpy as np
import pandas as pd
from scipy import sparse

from remy.paths import PROJECT_ROOT
from remy.splits import MIN_REVIEWS

EMBEDDING_DIR = PROJECT_ROOT / 'data/processed/embeddings/qwen'
TEXT_FEATURES = ['name', 'ingredients', 'tags', 'description']
FEATURE_WEIGHTS = dict(name=1., ingredients=1., tags=1., description=1., numeric=.5)
RATING_WEIGHTS = {5: 1., 4: .5, 3: 0., 2: 0., 1: 0., 0: 0.}


def load_embeddings(directory=EMBEDDING_DIR):
    metadata = json.loads((directory / 'metadata.json').read_text())
    arrays = {f: np.load(directory / f'{f}.npy', mmap_mode='r')
              for f in ['id', 'numeric'] + TEXT_FEATURES}
    assert len(np.unique(arrays['id'])) == len(arrays['id']) == metadata['rows']
    assert all(arrays[f].shape == (metadata['rows'], metadata['dimension']) for f in TEXT_FEATURES)
    assert arrays['numeric'].shape == (metadata['rows'], 10)
    return arrays


class EmbeddingRecommender:
    """Rerank popular candidates using averaged preferences or individual historical matches."""

    def __init__(self, embeddings, rating_weights=None, feature_weights=None,
                 n_candidates=10000, aggregation='centroid', history_k=3):
        self.embeddings = embeddings
        self.rating_weights = RATING_WEIGHTS if rating_weights is None else rating_weights
        self.feature_weights = FEATURE_WEIGHTS if feature_weights is None else feature_weights
        self.n_candidates = n_candidates
        self.aggregation = aggregation
        self.history_k = history_k
        if aggregation not in {'centroid', 'top_history'} or history_k < 1 or n_candidates < 1:
            raise ValueError('Invalid profile or candidate settings')
        if set(self.feature_weights) != set(FEATURE_WEIGHTS):
            raise ValueError('Provide a weight for every feature')
        if min(self.feature_weights.values()) < 0 or sum(self.feature_weights.values()) <= 0:
            raise ValueError('Feature weights must be nonnegative, with at least one positive')
        if any(weight < 0 for weight in self.rating_weights.values()):
            raise ValueError('Rating weights must be nonnegative')
        ids = embeddings['id']
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Recipe IDs must be unique')
        self.recipe_row_ = pd.Series(np.arange(len(ids)), index=ids)

    def fit(self, interactions, recipes=None, training_quarters=()):
        if recipes is None or recipes.empty:
            raise ValueError('Available recipes are required')
        counts = interactions.recipe_id.value_counts().reindex(recipes.id, fill_value=0)
        self.ranking_ = counts.sort_index().sort_values(ascending=False, kind='stable').index.tolist()
        self.seen_ = interactions.groupby('user_id').recipe_id.agg(set).to_dict()
        self.candidates_ = np.asarray(self.ranking_[:self.n_candidates])
        candidate_rows = self.recipe_row_.loc[self.candidates_].to_numpy()

        # Fit the same numeric transformation for users and candidates at this cutoff.
        numeric = np.log1p(self.embeddings['numeric'])
        available = numeric[self.recipe_row_.loc[recipes.id].to_numpy()]
        low, high = np.percentile(available, [1, 99], axis=0)
        clipped = np.clip(available, low, high)
        scale = clipped.std(axis=0)
        numeric = (np.clip(numeric, low, high) - clipped.mean(axis=0)) / np.where(scale > 0, scale, 1)
        if not np.isfinite(numeric).all():
            raise ValueError('Non-finite numerical features')

        weights = interactions.rating.map(self.rating_weights).fillna(0.)
        counts = interactions.groupby('user_id').size()
        liked = interactions[(weights > 0) & interactions.user_id.isin(counts[counts >= MIN_REVIEWS].index)]
        self.user_row_ = pd.Series(np.arange(liked.user_id.nunique()), index=np.sort(liked.user_id.unique()))
        W = sparse.csr_matrix((weights.loc[liked.index].to_numpy(np.float32),
                              (self.user_row_.loc[liked.user_id].to_numpy(),
                               self.recipe_row_.loc[liked.recipe_id].to_numpy())),
                             shape=(len(self.user_row_), len(self.recipe_row_)))
        self.user_vectors_, self.candidate_vectors_, self.candidate_present_ = {}, {}, {}
        for feature in TEXT_FEATURES:
            if self.feature_weights[feature] == 0:
                continue
            self.user_vectors_[feature] = self.normalize(W @ self.embeddings[feature])
            self.candidate_vectors_[feature] = self.normalize(self.embeddings[feature][candidate_rows].astype(np.float32))
            self.candidate_present_[feature] = self.candidate_vectors_[feature].any(axis=1)
        self.user_vectors_['numeric'] = np.asarray(W @ numeric) / W.sum(axis=1).A
        self.candidate_vectors_['numeric'] = numeric[candidate_rows]
        if self.aggregation == 'top_history':
            history = liked[['user_id', 'recipe_id']].copy()
            history['weight'] = weights.loc[liked.index]
            self.history_ = {u: (self.recipe_row_.loc[g.recipe_id].to_numpy(), g.weight.to_numpy())
                             for u, g in history.groupby('user_id')}
        return self

    def recommend_many(self, user_ids, k=None):
        if not hasattr(self, 'user_vectors_'):
            raise RuntimeError('Call fit before recommend_many')
        if k is not None and k < 0:
            raise ValueError('k must be nonnegative')
        return {u: self._recommend_one(u, k) for u in user_ids if u in self.user_row_.index}

    def _recommend_one(self, user_id, k):
        scores = self._scores(self.user_row_.loc[user_id])
        order = np.lexsort((self.candidates_, -scores))
        seen = self.seen_.get(user_id, set())
        ranked = [int(self.candidates_[i]) for i in order
                  if np.isfinite(scores[i]) and self.candidates_[i] not in seen]
        return ranked if k is None else ranked[:k]

    def _scores(self, row):
        if self.aggregation == 'top_history':
            return self._history_scores(row)
        total = np.zeros(len(self.candidates_), dtype=np.float32)
        denominator = np.zeros_like(total)
        for feature in TEXT_FEATURES:
            weight = self.feature_weights[feature]
            if weight == 0:
                continue
            user = self.user_vectors_[feature][row]
            present = self.candidate_present_[feature] & user.any()
            total += weight * (self.candidate_vectors_[feature] @ user) * present
            denominator += weight * present
        if self.feature_weights['numeric']:
            total += self.feature_weights['numeric'] * self._numeric_scores(row)
            denominator += self.feature_weights['numeric']
        return np.divide(total, denominator, out=np.full_like(total, np.nan), where=denominator > 0)

    def _numeric_scores(self, row):
        diff = self.candidate_vectors_['numeric'] - self.user_vectors_['numeric'][row]
        return np.exp(-np.sqrt((diff**2).mean(axis=1)))

    def _history_scores(self, row):
        history_rows, rating_weights = self.history_[self.user_row_.index[row]]
        history = {f: self.normalize(self.embeddings[f][history_rows].astype(np.float32))
                   for f in TEXT_FEATURES if self.feature_weights[f] > 0}
        numeric = self._numeric_scores(row) if self.feature_weights['numeric'] else None
        scores = np.full(len(self.candidates_), np.nan, dtype=np.float32)
        for start in range(0, len(scores), 256):
            end = min(start + 256, len(scores))
            total = np.zeros((end-start, len(history_rows)), dtype=np.float32)
            denominator = np.zeros_like(total)
            for feature, vectors in history.items():
                candidates = self.candidate_vectors_[feature][start:end]
                present = candidates.any(axis=1)[:, None] & vectors.any(axis=1)[None, :]
                total += self.feature_weights[feature] * (candidates @ vectors.T) * present
                denominator += self.feature_weights[feature] * present
            if numeric is not None:
                total += self.feature_weights['numeric'] * numeric[start:end, None]
                denominator += self.feature_weights['numeric']
            pairs = np.divide(total, denominator, out=np.full_like(total, -np.inf), where=denominator > 0)
            n = min(self.history_k, len(history_rows))
            best = np.argpartition(pairs, -n, axis=1)[:, -n:]
            selected = np.take_along_axis(pairs, best, axis=1)
            present = np.isfinite(selected)
            weights = rating_weights[best] * present
            numerator = (np.where(present, selected, 0.) * weights).sum(axis=1)
            scores[start:end] = np.divide(numerator, weights.sum(axis=1),
                                          out=np.full(end-start, np.nan), where=weights.sum(axis=1) > 0)
        return scores

    @staticmethod
    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
