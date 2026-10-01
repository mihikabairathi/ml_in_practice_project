"""Small behavior checks for the shared rolling experiment path."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from remy.model.item_knn import ItemKNNRecommender
from remy.splits import RollingSplits
from scripts.run_experiments import evaluate_quarter, parse_args, run


class ItemKNNTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame(
            [
                (1, 10, 5, "2006-01-01"),
                (1, 10, 5, "2006-01-02"),  # duplicate must not double the signal
                (2, 10, 5, "2006-01-03"),
                (2, 20, 5, "2006-01-04"),
                (3, 30, 1, "2006-01-05"),
                (1, 20, 5, "2007-01-01"),  # future must not enter the fit
            ],
            columns=["user_id", "recipe_id", "rating", "date"],
        )
        self.history["date"] = pd.to_datetime(self.history["date"])
        self.recipes = pd.DataFrame({"id": [10, 20, 30, 40]})

    def test_neighbors_cutoff_and_fallback(self):
        model = ItemKNNRecommender(k_neighbors=1).fit(
            self.history, recipes=self.recipes, asof="2007-01-01"
        )
        self.assertEqual(model.R_[model.user_to_idx_[1]].sum(), 1)
        self.assertEqual(model.recommend_candidates(1), [20])
        self.assertEqual(model.recommend(1, 3), [20, 30, 40])
        self.assertEqual(model.recommend(3, 3), [10, 20, 40])
        self.assertEqual(model.recommend(999, 4), [10, 20, 30, 40])
        self.assertEqual(model.S_.diagonal().sum(), 0)

    def test_invalid_neighbor_count(self):
        with self.assertRaises(ValueError):
            ItemKNNRecommender(k_neighbors=0)

    def test_no_positive_interactions_uses_popularity(self):
        history = self.history.assign(rating=1)
        model = ItemKNNRecommender().fit(history, recipes=self.recipes, asof="2007-01-01")
        self.assertEqual(model.S_.nnz, 0)
        self.assertEqual(model.recommend(1, 3), [20, 30, 40])


class RunnerTests(unittest.TestCase):
    def test_rolling_quarter_uses_only_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interactions = pd.DataFrame(
                [(1, 10, 5, "2006-01-01"), (1, 20, 5, "2006-01-02"),
                 (1, 30, 5, "2006-01-03"), (2, 10, 5, "2006-01-03"),
                 (2, 40, 5, "2006-01-04"), (1, 40, 5, "2007-02-01")],
                columns=["user_id", "recipe_id", "rating", "date"],
            )
            interactions["date"] = pd.to_datetime(interactions["date"])
            recipes = pd.DataFrame({"id": [10, 20, 30, 40, 50], "submitted": pd.to_datetime(["2005-01-01"] * 5)})
            windows = pd.DataFrame({"start": pd.to_datetime(["2007-01-01"]), "end": pd.to_datetime(["2007-04-01"]), "split": ["val"]})
            interactions.to_parquet(root / "interactions.parquet")
            recipes.to_parquet(root / "recipes.parquet")
            windows.to_parquet(root / "windows.parquet")
            quarter = RollingSplits(root).evaluation_quarters("val")[0]
            self.assertEqual(len(quarter.history), 5)
            self.assertEqual(len(quarter.outcomes), 1)
            model = ItemKNNRecommender().fit(quarter.history, recipes=quarter.recipes)
            self.assertNotIn(40, model.seen_[1])
            self.assertEqual(evaluate_quarter(model, quarter, "item_knn")["HR@10"], 1)
            args = parse_args(["--models", "popularity", "item_knn", "--split-dir", directory])
            result = run(args)
            self.assertEqual(set(result["model"]), {"popularity", "item_knn"})
            self.assertTrue((root / "experiments_val.csv").exists())


if __name__ == "__main__":
    unittest.main()
