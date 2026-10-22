"""Ranking metrics for one quarterly task of the shared rolling splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

METRICS = ["NDCG", "NDCG@Candidates", "HR@10", "HR@100", "Recall@10", "Recall@100"]
DEFAULT_KS = (10, 100)


def evaluate_quarter(
    quarter,
    global_ranking: Sequence,
    seen: Mapping,
    recs: Mapping | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    candidate_k: int = 1000,
) -> dict:
    """Full-ranking and fixed-cutoff NDCG, HR@k and Recall@k.

    quarter: a remy.splits.Quarter; only its outcomes and users are read
    global_ranking: ranking provided by a model, as a list of recipe_ids
    seen: {user_id: set(recipe_id)} previously reviewed recipes by each user
    recs: optional {user_id: [recipe_id, ...]} ranked before the global fallback;
        candidate lists must exclude seen recipes. An empty mapping means the
        model uses the global ranking directly.
    ks: which top-k metrics to compute
    candidate_k: shared NDCG cutoff for all models, normally the shortlist size

    Positives are 5-star reviews in the following quarter, from users with at
    least three prior reviews of **any** rating, on available, previously
    unseen recipes.
    Users with no positive that quarter receive no ranking score.
    """

    positives = quarter.outcomes[quarter.outcomes["rating"] == 5]
    if positives.empty:
        raise ValueError(f"No scorable users at {quarter.start.date()}")
    global_rank = {recipe: rank for rank, recipe in enumerate(global_ranking, start=1)}
    recs = {} if recs is None else recs

    scores = []
    for user_id, rows in positives.groupby("user_id"):
        relevant = set(rows["recipe_id"])

        # Compute ranks from model candidates, then complete with global ranking.
        cands = recs.get(user_id, [])
        cand_rank = {recipe: rank for rank, recipe in enumerate(cands, start=1)}
        # Exclude seen and candidate recipes from the global fallback.
        blocked = set(seen.get(user_id, ())) | set(cands)
        blocked_ranks = np.sort([global_rank[r] for r in blocked if r in global_rank])
        ranks = np.array([
            cand_rank[r] if r in cand_rank
            else len(cands) + global_rank[r] - np.searchsorted(blocked_ranks, global_rank[r])
            for r in relevant
        ])

        # compute metrics
        dcg = (1 / np.log2(ranks + 1)).sum()
        ideal = (1 / np.log2(np.arange(2, len(relevant) + 2))).sum()
        metrics = {"NDCG": dcg / ideal}

        # Apply the same shortlist-length cutoff to every model. The ranks
        # beyond a model's returned candidates still follow the global fallback.
        candidate_ranks = ranks[ranks <= candidate_k]
        candidate_ideal_count = min(len(relevant), candidate_k)
        candidate_ideal = (
            1 / np.log2(np.arange(2, candidate_ideal_count + 2))
        ).sum()
        candidate_dcg = (1 / np.log2(candidate_ranks + 1)).sum()
        metrics["NDCG@Candidates"] = (
            candidate_dcg / candidate_ideal if candidate_ideal_count else 0.0
        )

        for k in ks:
            hits = (ranks <= k).sum()
            metrics[f"HR@{k}"] = float(hits > 0)
            metrics[f"Recall@{k}"] = hits / len(relevant)
        scores.append(metrics)

    if not scores:
        raise ValueError(f"No scorable users at {quarter.start.date()}")
    result = {
        "start": quarter.start,
        "eligible_users": len(quarter.users),
        "scored_users": len(scores),
        "positives": len(positives),
    }
    result.update(pd.DataFrame(scores).mean().to_dict())
    return result
