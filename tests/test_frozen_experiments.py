"""Configuration provenance and runner integration checks (no downloaded data)."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from remy.evaluate import METRICS, evaluate_quarter
from remy.experiment_config import load_config, make_configured_model
from remy.model.lightgbm import ALL_ROUND1
from remy.model.popularity import Popularity
from remy.paths import PROJECT_ROOT
from remy.splits import RollingSplits
from scripts.run_experiments import completed_top_k, make_model, parse_args, run, validate_recommendations

CONFIG = PROJECT_ROOT / "configs/frozen_models.toml"


class FrozenExperimentsTest(unittest.TestCase):
    def test_loading_manifest_does_not_load_native_model_runtimes(self):
        code = ("import sys; from remy.experiment_config import load_config; "
                "load_config(sys.argv[1]); "
                "assert not {'torch', 'lightgbm', 'xgboost'} & sys.modules.keys()")
        subprocess.run([sys.executable, "-c", code, str(CONFIG)], check=True)

    def test_report_selections_are_explicit(self):
        models = load_config(CONFIG)["models"]
        self.assertEqual(set(models), {"popularity", "knn", "embedding", "embedding_trained", "lightgbm", "xgboost"})
        self.assertEqual(tuple(models["lightgbm"]["params"]["feature_groups"]), ALL_ROUND1)
        self.assertEqual(models["knn"]["params"]["cf_weight"], 0.1)
        self.assertEqual(models["knn"]["params"]["k_neighbors"], 300)
        self.assertEqual(models["knn"]["params"]["shrinkage"], 20)
        self.assertEqual(models["embedding_trained"]["params"]["variant"], "softmax")

    def test_legacy_cli_and_frozen_selection(self):
        args = parse_args(["--models", "knn"])
        m = make_model("knn", args)
        self.assertEqual((m.k_neighbors, m.shrinkage, m.cf_weight), (50, 0, 1))
        args = parse_args(["--config", str(CONFIG), "--models", "knn", "--split", "test"])
        self.assertEqual(args.models, ["knn"])
        self.assertEqual(args.split, "test")
        self.assertEqual(make_model("knn", args).cf_weight, 0.1)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--config", str(CONFIG), "--boost-rounds=2"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--config", str(CONFIG), "--models", "llm"])

    def test_unknown_parameter_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text('schema_version = 1\n[models.popularity]\ntype = "popularity"\nparams = { typo = 3 }\n')
            with self.assertRaisesRegex(ValueError, "typo"):
                make_configured_model(load_config(path)["models"]["popularity"])

    def test_saved_completion_matches_evaluator_ranks(self):
        seen = {1: {2, 6}}
        ranking = [1, 2, 3, 4, 5, 6, 7]
        recs = {1: [5, 3]}
        self.assertEqual(completed_top_k(1, recs, ranking, seen, 5), [5, 3, 1, 4, 7])
        self.assertEqual(completed_top_k(1, {}, ranking, seen, 3), [1, 3, 4])
        self.assertEqual(completed_top_k(1, recs, ranking, seen, 1), [5])
        with self.assertRaises(ValueError):
            validate_recommendations({1: [2]}, [1], ranking, seen)
        with self.assertRaises(ValueError):
            validate_recommendations({1: [3, 3]}, [1], ranking, seen)

    def test_knn_shrinkage_uses_co_like_support(self):
        from remy.model.knn import KNNRecommender

        h = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "recipe_id": [1, 2, 1, 2, 1], "rating": [5]*5})
        m = KNNRecommender(shrinkage=20, cf_weight=0.1).fit(h)
        a, b = m.item_to_idx_[1], m.item_to_idx_[2]
        self.assertAlmostEqual(float(m.S_[a, b]), (2 / np.sqrt(3*2)) * (2 / 22), places=6)

    def test_all_six_models_run_and_saved_predictions_reproduce_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data = root / "rolling"
            emb = root / "embeddings"
            data.mkdir()
            emb.mkdir()
            ids = np.arange(1, 21)
            recipes = pd.DataFrame({
                "id": ids, "submitted": pd.Timestamp("2000-01-01"),
                "tags": [["dessert" if i % 2 else "soup", "easy"] for i in ids],
                "minutes": ids * 5, "n_steps": ids % 5 + 1, "n_ingredients": ids % 7 + 1,
                "nutrition": [[100.0 + i, 1, 2] for i in ids],
            })
            history = []
            for user in range(1, 5):
                history.extend((user, recipe, "2002-01-01", 5) for recipe in [user, user+1, user+2])
                history.append((user, user+7, "2003-02-01", 5))
                history.append((user, user+12, "2004-02-01", 5))
            interactions = pd.DataFrame(history, columns=["user_id", "recipe_id", "date", "rating"])
            interactions["date"] = pd.to_datetime(interactions.date)
            windows = pd.DataFrame({"start": pd.to_datetime(["2003-01-01", "2004-01-01"]),
                                    "end": pd.to_datetime(["2003-04-01", "2004-04-01"]),
                                    "split": ["train", "val"]})
            recipes.to_parquet(data / "recipes.parquet")
            interactions.to_parquet(data / "interactions.parquet")
            windows.to_parquet(data / "windows.parquet")
            np.save(emb / "id.npy", ids)
            rng = np.random.default_rng(42)
            arrays = {"id": ids}
            for field in ("name", "ingredients", "vectors"):
                vectors = rng.normal(size=(len(ids), 8)).astype(np.float32)
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                arrays[field] = vectors
                np.save(emb / f"{field}.npy", vectors)
            # Check TOML rating keys are converted before the embedding model reads ratings.
            model = make_configured_model(load_config(CONFIG)["models"]["embedding"], arrays)
            self.assertEqual(model.rating_weights, {5: 1.0, 4: 0.5})
            args = parse_args(["--config", str(CONFIG), "--split-dir", str(data),
                               "--embedding-dir", str(emb), "--output", str(root / "run/results.csv"),
                               "--max-users", "2", "--no-plots"])
            results = run(args)
            self.assertEqual(len(results), 6)
            self.assertTrue(np.isfinite(results[METRICS]).all().all())
            q = RollingSplits(data).evaluation_quarters("val")[0]
            popularity = Popularity().fit(q.history, q.recipes)
            cohorts = []
            for name in args.models:
                frame = pd.read_parquet(root / f"run/recommendations/2004-01-01/{name}.parquet")
                cohorts.append(frame.user_id.tolist())
                recs = frame.set_index("user_id").recipe_ids.to_dict()
                validate_recommendations(recs, frame.user_id, recipes.id, popularity.seen_)
                from dataclasses import replace
                scored = replace(q, outcomes=q.outcomes[q.outcomes.user_id.isin(frame.user_id)])
                recovered = evaluate_quarter(scored, popularity.ranking_, popularity.seen_, recs)
                original = results[results.model.eq(name)].iloc[0]
                np.testing.assert_allclose([recovered[m] for m in METRICS], original[METRICS].to_numpy(float))
            self.assertTrue(all(cohort == cohorts[0] for cohort in cohorts))
            metadata = json.loads((root / "run/run_config.json").read_text())
            self.assertEqual(metadata["model_specs"]["knn"]["params"]["cf_weight"], 0.1)
            self.assertIn("src/remy/model/knn.py", metadata["source_sha256"])
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
