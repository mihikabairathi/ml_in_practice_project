"""Survey participants use the fitted XGBoost features without leaving state behind."""

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from remy.abtest.data import TrainingData
from remy.abtest.models import LightGBMSurvey, XGBoostSurvey
from remy.model.lightgbm import LightGBM
from remy.splits import Quarter


def training_data():
    ids = np.arange(1, 25)
    recipes = pd.DataFrame({
        "id": ids, "submitted": pd.Timestamp("2000-01-01"),
        "name": [f"recipe {i}" for i in ids], "description": "A recipe.",
        "ingredients": [["rice", "salt"] for _ in ids],
        "steps": [["cook rice."] for _ in ids],
        "tags": [["desserts" if i % 2 else "soups-stews", "easy"] for i in ids],
        "minutes": ids * 5, "n_steps": ids % 5 + 1, "n_ingredients": ids % 7 + 1,
        "nutrition": [[100.0 + i, 1, 2, 3, 4, 5, 6] for i in ids],
    })
    rows = []
    for user, ratings in [(1, [5, 4, 2, 5, 4]), (2, [1, 2, 3, 2, 1]),
                          (3, [1, 4, 2, 3, 1]), (4, [4, 5, 5, 4, 5])]:
        rows.extend((user, user + j, pd.Timestamp("2002-01-01") + pd.Timedelta(days=j), rating)
                    for j, rating in enumerate(ratings))
    before = pd.DataFrame(rows, columns=["user_id", "recipe_id", "date", "rating"])
    outcomes = pd.DataFrame({"user_id": [1, 2, 3, 4], "recipe_id": [13, 14, 15, 16],
                             "date": pd.Timestamp("2003-02-01"), "rating": [5, 2, 5, 5]})
    history = pd.concat([before, outcomes], ignore_index=True)
    quarter = Quarter(pd.Timestamp("2003-01-01"), pd.Timestamp("2003-04-01"),
                      before, recipes, pd.DataFrame({"user_id": [1, 2, 3, 4], "n_reviews": 5}), outcomes)
    rng = np.random.default_rng(42)
    # One catalog recipe deliberately lacks a vector, as can happen with partial assets.
    embeddings = {"id": ids[:-1]}
    for field in ("name", "ingredients"):
        vectors = rng.normal(size=(len(ids) - 1, 8)).astype(np.float32)
        embeddings[field] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return TrainingData(pd.Timestamp("2003-04-01"), history, recipes, [quarter]), embeddings


class XGBoostSurveyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, embeddings = training_data()
        cls.fitted = XGBoostSurvey(embeddings=embeddings, num_threads=1, num_boost_round=4,
                                  min_child_weight=1, n_candidates=24)
        cls.fitted.fit_once(cls.data)

    def setUp(self):
        self.wrapper = XGBoostSurvey()
        self.wrapper.model = copy.deepcopy(self.fitted.model)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_nested_equal(a, e)
        elif isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        elif isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(actual, expected)
        elif isinstance(expected, pd.Index):
            pd.testing.assert_index_equal(actual, expected)
        elif sparse.issparse(expected):
            np.testing.assert_array_equal(actual.toarray(), expected.toarray())
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            self.assertEqual(actual, expected)

    def assert_restored(self, state, seen):
        self.assert_nested_equal(self.wrapper.model.state_, state)
        self.assert_nested_equal(self.wrapper.model.seen_, seen)
        self.assertIs(self.wrapper.model.user_features_, self.wrapper.model.state_["user"])

    def test_copied_users_match_all_features_and_recommendations(self):
        m = self.wrapper.model
        state, seen = copy.deepcopy(m.state_), copy.deepcopy(m.seen_)
        content = copy.deepcopy((m.content_.raw, m.content_.normalized))
        for user in [1, 2, 3, 4, 1, 2]:
            with self.subTest(user=user):
                ratings = self.data.history[self.data.history.user_id.eq(user)].assign(user_id=-1)
                expected_features = m._features([user] * len(m.recipe_ids_), m.recipe_ids_, m.state_).toarray()
                expected_recs = m.recommend(user, k=10)
                self.wrapper.add_user(ratings)
                try:
                    actual = m._features([-1] * len(m.recipe_ids_), m.recipe_ids_, m.state_).toarray()
                    np.testing.assert_allclose(actual, expected_features, rtol=1e-6, atol=1e-6, equal_nan=True)
                    # Catalog centering is shared fitted item state, never participant state.
                    for mean, original in zip(m.state_["content"][1], state["content"][1]):
                        np.testing.assert_array_equal(mean, original)
                    if user == 2:
                        self.assertTrue(np.isnan(actual[:, len(m.dense_cols_):len(m.dense_cols_) + 3]).all())
                    else:
                        self.assertTrue(np.isfinite(actual[:-1, len(m.dense_cols_):len(m.dense_cols_) + 3]).all())
                finally:
                    self.wrapper.remove_user(-1)
                self.assertEqual(self.wrapper.recommend(ratings, k=10), expected_recs)
                self.assert_restored(state, seen)
                self.assert_nested_equal((m.content_.raw, m.content_.normalized), content)

    def test_no_likes_still_scores_and_failed_add_is_cleaned_up(self):
        m = self.wrapper.model
        state, seen = copy.deepcopy(m.state_), copy.deepcopy(m.seen_)
        ratings = self.data.history[self.data.history.user_id.eq(2)].assign(user_id=-1)
        with patch.object(m, "_score", wraps=m._score) as score:
            self.assertEqual(len(self.wrapper.recommend(ratings, k=10)), 10)
            score.assert_called_once()
        self.assert_restored(state, seen)
        with patch.object(m, "_state", side_effect=RuntimeError("feature construction failed")):
            with self.assertRaisesRegex(RuntimeError, "feature construction failed"):
                self.wrapper.recommend(ratings, k=10)
        self.assert_restored(state, seen)
        add_user = LightGBMSurvey.add_user

        def fail_after_attaching(wrapper, rows):
            add_user(wrapper, rows)
            raise RuntimeError("content attachment failed")

        liked_ratings = self.data.history[self.data.history.user_id.eq(1)].assign(user_id=-1)
        with patch.object(LightGBMSurvey, "add_user", fail_after_attaching):
            with self.assertRaisesRegex(RuntimeError, "content attachment failed"):
                self.wrapper.recommend(liked_ratings, k=10)
        self.assert_restored(state, seen)

    def test_cache_key_tracks_inherited_feature_code(self):
        wrapper = XGBoostSurvey()
        original = wrapper.cache_key(self.data.cutoff)
        getsource = inspect.getsource

        def changed_source(module):
            source = getsource(module)
            return source + "\n# changed feature computation\n" if module.__name__ == LightGBM.__module__ else source

        with patch("remy.abtest.models.inspect.getsource", side_effect=changed_source):
            self.assertNotEqual(wrapper.cache_key(self.data.cutoff), original)

    def test_prediction_exception_removes_participant(self):
        m = self.wrapper.model
        state, seen = copy.deepcopy(m.state_), copy.deepcopy(m.seen_)
        ratings = self.data.history[self.data.history.user_id.eq(1)].assign(user_id=-1)
        with patch.object(m, "_score", side_effect=RuntimeError("scoring failed")):
            with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                self.wrapper.recommend(ratings, k=10)
        self.assert_restored(state, seen)
        self.assertEqual(self.wrapper.recommend(ratings, k=10), m.recommend(1, k=10))

    def test_complete_blind_survey_with_and_without_likes(self):
        from fastapi.testclient import TestClient
        from remy.abtest.app import COOKIE, app
        from remy.abtest.config import Config

        with tempfile.TemporaryDirectory() as tmp:
            config = Config(train_cutoff=self.data.cutoff.date(), responses_dir=Path(tmp) / "responses",
                            model_cache_dir=Path(tmp) / "models",
                            survey={"n_quiz": 5, "n_recs": 3, "quiz_pool_size": 21,
                                    "ask_preference": False},
                            models={"a": {"type": "popularity"}, "b": {"type": "xgboost"}})

            def use_fitted(wrapper, data, cache_dir):
                wrapper.model = copy.deepcopy(self.fitted.model)

            with patch("remy.abtest.app.load_config", return_value=config), \
                 patch("remy.abtest.app.load_training_data", return_value=self.data), \
                 patch.object(XGBoostSurvey, "fit_or_load", use_fitted), TestClient(app) as client:
                xgb = app.state.models["xgboost"].model
                state, seen = copy.deepcopy(xgb.state_), copy.deepcopy(xgb.seen_)
                for rating in [5, 2]:
                    self.assertEqual(client.post("/start", data={}).url.path, "/quiz")
                    session = app.state.log.sessions[client.cookies[COOKIE]]
                    quiz_ids = session["quiz_ids"]
                    self.assertEqual(len(quiz_ids), 5)
                    response = client.post("/quiz", data={f"r_{r}": rating for r in quiz_ids})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.url.path, "/compare")
                    for forbidden in ["xgboost", "lightgbm", "popularity", "profile_cosine"]:
                        self.assertNotIn(forbidden, response.text.lower())
                    form = {}
                    for side, ids in session["lists"].items():
                        self.assertEqual(len(ids), 3)
                        self.assertFalse(set(ids) & set(quiz_ids))
                        form.update({f"item_{side}_{rank}": 4 for rank in range(1, 4)})
                        form[f"list_{side}"] = 4
                    self.assertEqual(client.post("/compare", data=form).url.path, "/done")
                    self.assert_nested_equal(xgb.state_, state)
                    self.assert_nested_equal(xgb.seen_, seen)
                records = [json.loads(line) for line in app.state.log.path.read_text().splitlines()]
                self.assertEqual(len(records), 2)
                self.assertEqual(set(records[0]["models"]), {"popularity", "xgboost"})


if __name__ == "__main__":
    unittest.main()
