"""Random Forest recipe reranking with dense popularity and MiniLM features."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from remy.model.xgboost import XGBOOST_GROUPS, _TreeReranker


class RandomForest(_TreeReranker):
    def __init__(self, seed=10718, num_threads=4):
        groups = tuple(group for group in XGBOOST_GROUPS if group != "tags")
        super().__init__(groups, seed, num_threads)

    @staticmethod
    def _matrix(features):
        return np.nan_to_num(features.toarray(), copy=False, nan=-1e9)

    def _train(self, X, y):
        return RandomForestClassifier(
            n_estimators=32,
            max_depth=12,
            min_samples_leaf=40,
            max_features=0.5,
            max_samples=0.12,
            n_jobs=self.num_threads,
            random_state=self.seed,
        ).fit(self._matrix(X), y)

    def _score(self, features):
        return self.model_.predict_proba(self._matrix(features))[:, 1]
