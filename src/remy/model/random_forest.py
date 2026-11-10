"""Random forest classifier for the feature-rich recipe reranker."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from remy.model.lightgbm_features import ALL_ROUND1, LightGBM


class RandomForest(LightGBM):
    def __init__(self, *, max_depth=12, n_estimators=40, min_samples_leaf=20,
                 max_features="sqrt", max_samples=0.2, num_threads=4,
                 feature_groups=ALL_ROUND1, **kwargs):
        super().__init__(feature_groups=feature_groups, num_threads=num_threads, **kwargs)
        if self.objective != "binary":
            raise ValueError("Random forest reranker requires the binary objective")
        self.estimator_params = dict(
            max_depth=max_depth, n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            max_samples=max_samples,
            n_jobs=num_threads, random_state=self.seed,
        )

    @staticmethod
    def _matrix(X):
        # sklearn's sparse forest input requires finite values. Use a sentinel
        # outside the range of the shared numeric recipe features.
        X = X.copy()
        np.nan_to_num(X.data, copy=False, nan=-1e9)
        return X

    def _fit_estimator(self, X_train, y_train, group_sizes):
        self.model_ = RandomForestClassifier(**self.estimator_params)
        self.model_.fit(self._matrix(X_train).tocsc(), y_train)

    def _score(self, X):
        return self.model_.predict_proba(self._matrix(X))[:, 1]
