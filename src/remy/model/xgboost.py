"""XGBoost classifier for the feature-rich recipe reranker."""

import numpy as np
from scipy import sparse

from remy.model.lightgbm_features import ALL_ROUND1, LightGBM


class XGBoost(LightGBM):
    def __init__(self, *, max_depth=4, n_estimators=150, learning_rate=0.08,
                 num_threads=4, feature_groups=ALL_ROUND1, **kwargs):
        super().__init__(feature_groups=feature_groups, num_threads=num_threads, **kwargs)
        if self.objective != "binary":
            raise ValueError("XGBoost reranker requires the binary objective")
        self.estimator_params = dict(
            max_depth=max_depth, n_estimators=n_estimators,
            learning_rate=learning_rate, n_jobs=num_threads,
            random_state=self.seed, tree_method="hist", eval_metric="logloss",
        )

    def _matrix(self, X):
        # XGBoost treats absent sparse entries as missing. Store numeric zeros
        # explicitly so that zero reviews, for example, differs from NaN.
        width = len(self.dense_cols_)
        if not width:
            return X
        values = X[:, :width].toarray()
        cols = np.tile(np.arange(width, dtype=np.int32), X.shape[0])
        offsets = np.arange(X.shape[0] + 1, dtype=np.int64) * width
        numeric = sparse.csr_matrix((values.ravel(), cols, offsets), shape=values.shape)
        return sparse.hstack([numeric, X[:, width:]], format="csr")

    def _fit_estimator(self, X_train, y_train, group_sizes):
        from xgboost import XGBClassifier

        self.model_ = XGBClassifier(**self.estimator_params)
        self.model_.fit(self._matrix(X_train), y_train)

    def _score(self, X):
        return self.model_.predict_proba(self._matrix(X))[:, 1]
