"""
Ranking metrics for one quarterly task of the shared rolling splits.
"""

import numpy as np
import pandas as pd


def evaluate_quarter(quarter, global_ranking, seen, recs={}, ks=(10, 100)) -> dict:
    """Full-ranking NDCG, HR@k and Recall@k, averaged over users with a 5-star outcome.

    quarter: a remy.splits.Quarter; only its outcomes and users are read
    global_ranking: ranking provided by a model, as a list of recipe_ids
    seen: {user_id: set(recipe_id)} previously reviewed recipes by each user
    recs: optional {user_id: [recipe_id, ...]} ranked ahead of the fallback global order, must exclude seen recipes
    ks: which top-k metrics to compute

    Positives are 5-star reviews in the following quarter, from users with at least three prior reviews of **any** rating, on available, previously unseen recipes. 
    Users with no positive that quarter receive no ranking score.
    """

    positives = quarter.outcomes[quarter.outcomes["rating"] == 5]
    global_rank = {recipe: rank for rank, recipe in enumerate(global_ranking, start=1)}

    scores = []
    for user_id, rows in positives.groupby("user_id"):
        relevant = set(rows["recipe_id"])

        # Compute ranks of relevant recipes, using candidates first, then fallback global ranking.
        # For the popularity model, this will be empty
        cands = recs.get(user_id, [])
        cand_rank = {recipe: rank for rank, recipe in enumerate(cands, start=1)}
        blocked = set(seen.get(user_id, ())) | set(cands) # recipes that should not be included in the fallback
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

    if recs:
        # Share of positives inside the candidates: the reranker's ceiling.
        in_cands = [r in set(recs.get(u, ())) for u, r in zip(positives["user_id"], positives["recipe_id"])]
        result["candidate_recall"] = float(np.mean(in_cands))
    return result
