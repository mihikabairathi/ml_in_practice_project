"""Fixed top-ten substitute, diversity and exposure diagnostics."""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

DIAGNOSTIC_METRICS = ["Soft Recall@10", "Diversity@10", "Top10 Exposure Share"]


class RecipeSimilarity:
    """Uncentered cosine, equally weighted over frozen name/ingredient fields."""

    def __init__(self, embeddings):
        ids = np.asarray(embeddings["id"])
        if len(ids) != len(set(ids)):
            raise ValueError("Embedding recipe IDs must be unique")
        self.rows = pd.Series(np.arange(len(ids)), index=ids)
        self.vectors = []
        for field in ("name", "ingredients"):
            values = np.asarray(embeddings[field], dtype=np.float32)
            if len(values) != len(ids) or not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
                raise ValueError(f"Invalid {field} embeddings")
            self.vectors.append(normalize(values))

    def pairs(self, left, right):
        a = self.rows.loc[list(left)].to_numpy()
        b = self.rows.loc[list(right)].to_numpy()
        return np.clip(sum(v[a] @ v[b].T for v in self.vectors) / 2, 0, 1)


def score_list(recommended, relevant, similarity, threshold=0.8):
    """Lock exact IDs first, then maximum-weight one-to-one substitute matches."""
    recommended, relevant = list(recommended), set(relevant)
    exact = set(recommended) & relevant
    matches = [{"recommended_id": int(r), "relevant_id": int(r), "similarity": 1.0, "exact": True}
               for r in sorted(exact)]
    left = [r for r in recommended if r not in exact]
    right = sorted(relevant - exact)
    if left and right:
        values = similarity.pairs(left, right)
        weights = np.where(values >= threshold, values, 0)
        rows, cols = linear_sum_assignment(weights, maximize=True)
        matches.extend({"recommended_id": int(left[i]), "relevant_id": int(right[j]),
                        "similarity": float(values[i, j]), "exact": False}
                       for i, j in zip(rows, cols) if weights[i, j] > 0)
    soft = sum(match["similarity"] for match in matches) / len(relevant)
    if len(recommended) < 2:
        raise ValueError("At least two recommendations are required for diversity")
    pairs = similarity.pairs(recommended, recommended)
    diversity = float((1 - pairs[np.triu_indices(len(recommended), 1)]).mean())
    return len(exact) / len(relevant), float(soft), diversity, matches


def evaluate_diagnostics(quarter, predictions, similarity, threshold=0.8, k=10):
    if k != 10 or not 0 <= threshold <= 1:
        raise ValueError("The approved diagnostics use k=10 and a similarity threshold in [0,1]")
    truth = quarter.outcomes[quarter.outcomes.rating.eq(5)].groupby("user_id").recipe_id.agg(set)
    if predictions.user_id.duplicated().any() or set(predictions.user_id) != set(truth.index):
        raise ValueError("Predictions must contain exactly one list for every scored user")
    catalog = set(quarter.recipes.id)
    seen = quarter.history.groupby("user_id").recipe_id.agg(set).to_dict()
    users, matches, exposure = [], [], Counter()
    for user, recipes in zip(predictions.user_id, predictions.recipe_ids):
        recipes = list(recipes[:k])
        if len(recipes) != k or len(set(recipes)) != k or not set(recipes) <= catalog or set(recipes) & seen.get(user, set()):
            raise ValueError(f"User {user} does not have ten distinct, available, unseen recommendations")
        exact, soft, diversity, paired = score_list(recipes, truth[user], similarity, threshold)
        users.append({"user_id": int(user), "Recall@10": exact, "Soft Recall@10": soft, "Diversity@10": diversity})
        matches.extend({"user_id": int(user), **pair} for pair in paired)
        exposure.update(recipes)
    per_user = pd.DataFrame(users)
    metrics = per_user.drop(columns="user_id").mean().to_dict()
    metrics["Top10 Exposure Share"] = sum(count for _, count in exposure.most_common(10)) / (k * len(users))
    pair_frame = pd.DataFrame(matches, columns=["user_id", "recommended_id", "relevant_id", "similarity", "exact"])
    return metrics, per_user, pair_frame, exposure.most_common(10)
