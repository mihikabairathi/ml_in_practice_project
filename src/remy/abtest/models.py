"""Standard wrapper that lets any fitted remy model serve brand-new survey users.

The wrapped models are used unchanged: each is fit once on the training data,
and at request time the wrapper adds the participant's quiz ratings to the
model's per-user state (the same state fit() builds from interactions), calls
the model's own recommend(), then removes the participant again.

To add a model: subclass SurveyModel, set ``model_class`` (and ``supervised``
if fit needs training_quarters), extend add_user/remove_user with any per-user
state beyond ``seen_``, and register it in MODELS.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from remy.abtest.data import TrainingData
from remy.model.knn import KNNRecommender
from remy.model.lightgbm import LightGBM
from remy.model.popularity import Popularity
from remy.model.xgboost import XGBoost


class SurveyModel:
    """Base wrapper. Input: ratings with the interactions schema
    (user_id, recipe_id, date, rating) for ONE new user. Output: ranked recipe ids."""

    model_class: type
    supervised: bool = False

    def __init__(self, **params):
        self.params = params
        self.model = self.model_class(**params)
        self._lock = Lock()

    def fit_once(self, data: TrainingData) -> None:
        self.model.fit(
            data.history,
            recipes=data.catalog,
            training_quarters=data.training_quarters if self.supervised else (),
        )

    def recommend(self, ratings: pd.DataFrame, k: int) -> list[int]:
        user_ids = ratings["user_id"].unique()
        if len(user_ids) != 1:
            raise ValueError("ratings must belong to exactly one user")
        user_id = user_ids[0]
        with self._lock:  # the add/remove below mutates shared model state
            if user_id in self.model.seen_:
                raise ValueError(f"user {user_id} already exists in the fitted model")
            self.add_user(ratings)
            try:
                return [int(r) for r in self.model.recommend(user_id, k=k)]
            finally:
                self.remove_user(user_id)

    # Per-model hooks. Every current model tracks reviewed recipes in seen_.
    def add_user(self, ratings: pd.DataFrame) -> None:
        self.model.seen_[ratings["user_id"].iat[0]] = set(ratings["recipe_id"])

    def remove_user(self, user_id: int) -> None:
        self.model.seen_.pop(user_id, None)

    # Fit once, then reuse from disk until the cutoff, params or model code change.
    def cache_key(self, cutoff) -> str:
        source = inspect.getsource(sys.modules[self.model_class.__module__])
        blob = json.dumps(self.params, sort_keys=True) + source
        digest = hashlib.sha1(blob.encode()).hexdigest()[:10]
        return f"{self.model_class.__name__}-{pd.Timestamp(cutoff).date()}-{digest}.pkl"

    def fit_or_load(self, data: TrainingData, cache_dir: Path) -> None:
        path = Path(cache_dir) / self.cache_key(data.cutoff)
        if path.exists():
            tqdm.write(f"{self.model_class.__name__}: loading cached model from {path}")
            self.model = pickle.loads(path.read_bytes())
            tqdm.write(f"{self.model_class.__name__}: cache loaded; no training needed")
            return
        tqdm.write(f"{self.model_class.__name__}: no cached model; fitting on "
                   f"{len(data.history):,} reviews and {len(data.training_quarters) if self.supervised else 0} "
                   "completed training quarters")
        self.fit_once(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        tqdm.write(f"{self.model_class.__name__}: saving fitted model to {path}")
        path.write_bytes(pickle.dumps(self.model))


class PopularitySurvey(SurveyModel):
    model_class = Popularity


class KNNSurvey(SurveyModel):
    """KNN also needs the user's liked recipes as a row of R_ (see KNNRecommender.fit)."""

    model_class = KNNRecommender

    def add_user(self, ratings):
        super().add_user(ratings)
        m = self.model
        liked = ratings.loc[ratings["rating"] >= m.min_rating, "recipe_id"].drop_duplicates()
        cols = [m.item_to_idx_[r] for r in liked if r in m.item_to_idx_]
        if cols:
            row = sparse.csr_matrix(
                (np.ones(len(cols), dtype=np.float32), (np.zeros(len(cols), dtype=np.int64), cols)),
                shape=(1, m.R_.shape[1]),
            )
            m.R_ = sparse.vstack([m.R_, row], format="csr")
            m.user_to_idx_[ratings["user_id"].iat[0]] = m.R_.shape[0] - 1

    def remove_user(self, user_id):
        super().remove_user(user_id)
        m = self.model
        if m.user_to_idx_.pop(user_id, None) is not None:
            m.R_ = m.R_[:-1]


HELPER_USER_ID = -999_999_999


class LightGBMSurvey(SurveyModel):
    """LightGBM keeps per-user state in ``state_``: the user feature table
    (``state_["user"]``, also ``user_features_``) and tag-taste rows in
    ``state_["freq"]``/``state_["norm"]``, indexed by ``state_["urow"]``.

    The new user's rows come from the model's own ``_state`` run on just their
    ratings. The one exception is ``user_pref_log_pop`` (mean popularity of liked
    recipes), which needs the full history, so it is taken from the fitted
    recipe popularity instead.
    """

    model_class = LightGBM
    supervised = True

    def add_user(self, ratings):
        super().add_user(ratings)
        m, st = self.model, self.model.state_
        user_id = ratings["user_id"].iat[0]
        rows = ratings.rename(columns={"date": m.date_col_})
        # _state fails when no one in its input liked anything, so add a throwaway
        # helper user with one liked recipe. Features are per user, so it can't
        # change this user's rows; it is dropped below.
        helper = rows.iloc[[0]].assign(user_id=HELPER_USER_ID, rating=5, recipe_id=m.recipe_ids_[0])
        mini = m._state(pd.concat([rows, helper], ignore_index=True), m.cutoff_)

        user = mini["user"].drop(index=HELPER_USER_ID)
        liked = ratings.loc[ratings["rating"] >= m.liked_min_rating, "recipe_id"]
        liked = [r for r in liked.drop_duplicates() if r in m._catalog_set]
        if liked:
            user["user_pref_log_pop"] = st["recipe"].loc[liked, "recipe_log_pop"].mean()
        st["user"] = m.user_features_ = pd.concat([st["user"], user])

        if user_id in mini["urow"].index:  # only users with liked recipes get a taste row
            r = mini["urow"][user_id]
            st["freq"] = sparse.vstack([st["freq"], mini["freq"][r]], format="csr")
            st["norm"] = sparse.vstack([st["norm"], mini["norm"][r]], format="csr")
            st["urow"] = pd.concat([st["urow"], pd.Series({user_id: st["freq"].shape[0] - 1})])
        return mini

    def remove_user(self, user_id):
        super().remove_user(user_id)
        m, st = self.model, self.model.state_
        st["user"] = m.user_features_ = st["user"].drop(index=user_id, errors="ignore")
        if user_id in st["urow"].index:
            st["urow"] = st["urow"].drop(index=user_id)
            st["freq"], st["norm"] = st["freq"][:-1], st["norm"][:-1]


class XGBoostSurvey(LightGBMSurvey):
    """Add XGBoost's MiniLM profiles and recent likes alongside its tag taste.

    The model's own _state builds the participant's features with the same
    available catalog as fit. Only their user rows are attached; fitted catalog
    means and all learned item-side state stay unchanged.
    """

    model_class = XGBoost

    def cache_key(self, cutoff):
        # XGBoost inherits its feature construction and historical sampling from
        # LightGBM, so changes there must invalidate the fitted-model cache too.
        source = inspect.getsource(sys.modules[LightGBM.__module__])
        digest = hashlib.sha1((super().cache_key(cutoff) + source).encode()).hexdigest()[:10]
        return f"{self.model_class.__name__}-{pd.Timestamp(cutoff).date()}-{digest}.pkl"

    def add_user(self, ratings):
        m, st = self.model, self.model.state_
        user_id = ratings["user_id"].iat[0]
        # Keep original references so cleanup also restores nested content arrays
        # exactly, and a failed add cannot leave partially attached user state.
        self._user_state_before = {
            key: st[key] for key in ("user", "freq", "norm", "urow", "content")
        }
        self._user_features_before = m.user_features_
        try:
            mini = super().add_user(ratings)
            users, means, profiles, recent = st["content"]
            new_users, _, new_profiles, new_recent = mini["content"]
            row = new_users.get_indexer([user_id])[0]
            if row >= 0:  # no liked embedded recipes means no content profile
                st["content"] = (
                    users.append(new_users[row:row + 1]),
                    means,
                    [np.concatenate([old, new[row:row + 1]], axis=0)
                     for old, new in zip(profiles, new_profiles)],
                    np.concatenate([recent, new_recent[row:row + 1]], axis=0),
                )
        except BaseException:
            self.remove_user(user_id)
            raise

    def remove_user(self, user_id):
        SurveyModel.remove_user(self, user_id)
        before = self.__dict__.pop("_user_state_before", None)
        if before is not None:
            self.model.state_.update(before)
            self.model.user_features_ = self.__dict__.pop("_user_features_before")


MODELS: dict[str, type[SurveyModel]] = {
    "popularity": PopularitySurvey,
    "knn": KNNSurvey,
    "lightgbm": LightGBMSurvey,
    "xgboost": XGBoostSurvey,
}
