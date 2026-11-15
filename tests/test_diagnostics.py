from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from remy.checkpoints import atomic_json, file_sha256
from remy.diagnostics import RecipeSimilarity, evaluate_diagnostics, score_list
from scripts.embed_recipes import generate_minilm
from scripts.run_frozen_test import valid_record


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        vectors = np.array([[1, 0], [1, 0], [0, 1], [.85, np.sqrt(1-.85**2)], [-1, 0]], dtype=np.float32)
        self.similarity = RecipeSimilarity({"id": np.arange(1, 6), "name": vectors, "ingredients": vectors})

    def test_partial_credit_threshold_and_one_to_one_matching(self):
        exact, soft, diversity, pairs = score_list([1, 3], {4}, self.similarity)
        self.assertEqual(exact, 0)
        self.assertAlmostEqual(soft, .85, places=6)
        self.assertAlmostEqual(diversity, 1)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["exact"])
        self.assertEqual(score_list([1, 3], {4}, self.similarity, threshold=.9)[1], 0)
        # Two identical-looking recommendations cannot both claim one held-out recipe.
        self.assertEqual(len(score_list([1, 2], {4}, self.similarity)[3]), 1)

    def test_exact_matches_are_locked_and_similarity_is_half_each_field(self):
        exact, soft, diversity, pairs = score_list([1, 2], {1, 3}, self.similarity)
        self.assertEqual(exact, .5)
        self.assertEqual(soft, .5)
        self.assertEqual(diversity, 0)
        self.assertEqual(pairs, [{"recommended_id": 1, "relevant_id": 1, "similarity": 1.0, "exact": True}])
        name = np.array([[1, 0], [1, 0]], dtype=np.float32)
        ingredients = np.array([[1, 0], [0, 1]], dtype=np.float32)
        similarity = RecipeSimilarity({"id": [1, 2], "name": name, "ingredients": ingredients})
        self.assertEqual(float(similarity.pairs([1], [2])[0, 0]), .5)

    def test_exposure_counts_only_top_ten_on_a_shared_cohort(self):
        vectors = np.ones((20, 2), dtype=np.float32)
        similarity = RecipeSimilarity({"id": np.arange(1, 21), "name": vectors, "ingredients": vectors})
        quarter = SimpleNamespace(recipes=pd.DataFrame({"id": np.arange(1, 21)}),
                                  history=pd.DataFrame(columns=["user_id", "recipe_id"]),
                                  outcomes=pd.DataFrame({"user_id": [1, 2], "recipe_id": [11, 1], "rating": [5, 5]}))
        predictions = pd.DataFrame({"user_id": [1, 2], "recipe_ids": [list(range(1, 21)), list(range(11, 21))+[1]]})
        metrics, users, pairs, top = evaluate_diagnostics(quarter, predictions, similarity)
        self.assertEqual(metrics["Top10 Exposure Share"], .5)
        self.assertEqual(metrics["Recall@10"], 0) # Both exact positives are at rank 11.
        predictions["recipe_ids"] = [list(range(1, 21)), list(range(1, 21))]
        self.assertEqual(evaluate_diagnostics(quarter, predictions, similarity)[0]["Top10 Exposure Share"], 1)
        with self.assertRaises(ValueError):
            evaluate_diagnostics(quarter, predictions.iloc[:1], similarity)

    def test_corrupted_artifact_is_not_treated_as_a_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prediction").write_text("good")
            atomic_json(root / "checkpoint.json", {"files": {"prediction": file_sha256(root / "prediction")}})
            self.assertIsNotNone(valid_record(root / "checkpoint.json", root))
            (root / "prediction").write_text("corrupt")
            self.assertIsNone(valid_record(root / "checkpoint.json", root))


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, values, **kwargs):
        return bytes(values).decode()


class FakeEncoder:
    tokenizer = FakeTokenizer()

    def __init__(self, fail_on=None):
        self.calls = 0
        self.texts = []
        self.fail_on = fail_on

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("simulated interruption")
        self.texts.extend(texts)
        values = np.array([list(hashlib.sha256(text.encode()).digest()[:4]) for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class EmbeddingResumeTest(unittest.TestCase):
    def test_resume_from_last_flushed_chunk_and_reject_changed_inputs(self):
        recipes = pd.DataFrame({"id": np.arange(1, 6), "name": [f"recipe {i}" for i in range(1, 6)],
                                "ingredients": [["flour", "water"]]*5})
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                generate_minilm(recipes, root, chunk_size=2, encoder=FakeEncoder(fail_on=2))
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["fields"]["name"]["rows"], 2)
            self.assertFalse((root / "name.npy").exists())
            encoder = FakeEncoder()
            generate_minilm(recipes, root, chunk_size=2, encoder=encoder)
            self.assertEqual(encoder.texts[0], "recipe 3")
            self.assertEqual(len(encoder.texts), 13)
            self.assertTrue(json.loads((root / "progress.json").read_text())["complete"])
            before = file_sha256(root / "name.npy")
            generate_minilm(recipes, root, encoder=FakeEncoder(fail_on=1))
            self.assertEqual(file_sha256(root / "name.npy"), before)
            changed = recipes.copy()
            changed.loc[0, "name"] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                generate_minilm(changed, root, encoder=FakeEncoder())


if __name__ == "__main__":
    unittest.main()
