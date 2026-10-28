"""Zero-shot LLM baseline: an instruct model picks k recipes from a popularity shortlist."""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cache

import pandas as pd

from remy.model.llm_prompts import build_messages, candidate_line, history_line
from remy.model.popularity import Popularity


@cache
def load_llm(model_name: str, max_model_len: int = 16384):
    """Load a vLLM engine once per process; every quarter reuses it.

    vLLM is imported here so the rest of the project does not need it
    (install with: pip install -e ".[llm]"). Needs a GPU, e.g. one L40S.
    """
    from vllm import LLM

    return LLM(model=model_name, max_model_len=max_model_len,
               enable_prefix_caching=True, seed=0)


class LLMRecommender:
    """Ask an LLM to choose k recipes for each user, with no training.

    For each user, the prompt shows the ``n_candidates`` most-reviewed recipes
    the user has not reviewed yet (name + a few tags), followed by the user's
    ``n_history`` most recent 4-5 star recipes. The model answers in JSON, and
    guided decoding forces exactly k ids from that user's candidate list.
    Duplicate ids are dropped and any gaps are filled from the same list in
    popularity order, so every user gets k recommendations.

    Only k recipes are returned, so metrics beyond @k (HR@100, full NDCG)
    mostly measure the evaluator's popularity fallback, not the LLM.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        n_candidates: int = 200,
        n_history: int = 30,
        k: int = 10,
    ):
        if n_candidates < k or n_history < 1 or k < 1:
            raise ValueError("Need n_candidates >= k >= 1 and n_history >= 1")
        self.model_name = model_name
        self.n_candidates = n_candidates
        self.n_history = n_history
        self.k = k

    def fit(self, interactions: pd.DataFrame, recipes=None, training_quarters=()) -> "LLMRecommender":
        """Store popularity order and the history to show in prompts; nothing is learned."""
        popularity = Popularity().fit(interactions, recipes)
        self.ranking_, self.seen_ = popularity.ranking_, popularity.seen_
        self.history_ = interactions
        self.recipes_ = recipes.set_index("id")[["name", "tags"]]
        return self

    def candidates(self, user_id: int) -> list[int]:
        """The user's shortlist: most-reviewed recipes they have not reviewed."""
        seen = self.seen_.get(user_id, set())
        result = []
        for recipe_id in self.ranking_:
            if len(result) == self.n_candidates:
                break
            if recipe_id not in seen:
                result.append(recipe_id)
        return result

    def prompt(self, user_id: int, candidates: list[int], user_history: pd.DataFrame) -> list[dict]:
        """Chat messages for one user (see remy.model.llm_prompts)."""
        # A few reviews predate their recipe's listed submission date, so the
        # recipe is missing from this quarter's catalog; leave those out.
        user_history = user_history[user_history["recipe_id"].isin(self.recipes_.index)]
        liked = user_history[user_history["rating"] >= 4]
        if liked.empty:  # e.g. a user who only left low or unrated reviews
            liked = user_history
        liked = liked.tail(self.n_history)
        candidate_lines = [
            candidate_line(r, self.recipes_.at[r, "name"], self.recipes_.at[r, "tags"])
            for r in candidates
        ]
        history_lines = [
            history_line(self.recipes_.at[r, "name"], rating)
            for r, rating in zip(liked["recipe_id"], liked["rating"])
        ]
        return build_messages(candidate_lines, history_lines, self.k)

    def recommend_batch(self, user_ids: Iterable[int], k: int | None = None) -> dict[int, list[int]]:
        """Run one batched LLM call for all users and return k recipe ids each.

        Raw model responses are kept in ``self.responses_`` for inspection.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if not hasattr(self, "ranking_"):
            raise RuntimeError("Call fit before recommend_batch")
        k = self.k if k is None else min(k, self.k)
        user_ids = list(user_ids)
        history = self.history_[self.history_["user_id"].isin(user_ids)]
        history_by_user = dict(list(history.sort_values("date").groupby("user_id")))

        candidates, messages, params = {}, [], []
        for user_id in user_ids:
            candidates[user_id] = self.candidates(user_id)
            user_history = history_by_user.get(user_id, history.iloc[:0])
            messages.append(self.prompt(user_id, candidates[user_id], user_history))
            params.append(SamplingParams(
                temperature=0.0, max_tokens=256,
                structured_outputs=StructuredOutputsParams(
                    json=answer_schema(candidates[user_id], self.k)),
            ))

        # Qwen3 models "think" before answering unless told not to; thinking
        # was ~140x slower with no quality gain in our tests.
        outputs = load_llm(self.model_name).chat(
            messages, params, use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )

        recommendations, self.responses_ = {}, []
        for user_id, output in zip(user_ids, outputs):
            text = output.outputs[0].text
            picked = parse_recipe_ids(text, candidates[user_id])
            # Fill any gaps (duplicates, unparseable output) in popularity order.
            filled = picked + [r for r in candidates[user_id] if r not in picked]
            recommendations[user_id] = filled[:k]
            self.responses_.append(dict(user_id=int(user_id), response=text,
                                        n_valid=len(picked), llm_ids=picked))
        return recommendations

    def recommend(self, user_id: int, k: int | None = 10) -> list[int]:
        return self.recommend_batch([user_id], k=k)[user_id]


def answer_schema(candidates: list[int], k: int) -> dict:
    """JSON schema for {"recipe_ids": [...]}: exactly k ids from the candidates."""
    return {
        "type": "object",
        "properties": {"recipe_ids": {
            "type": "array",
            "items": {"type": "integer", "enum": [int(r) for r in candidates]},
            "minItems": k,
            "maxItems": k,
        }},
        "required": ["recipe_ids"],
        "additionalProperties": False,
    }


def parse_recipe_ids(text: str, candidates: list[int]) -> list[int]:
    """Valid, distinct candidate ids from a JSON response, in the model's order."""
    try:
        ids = json.loads(text)["recipe_ids"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    allowed = set(candidates)
    result = []
    for recipe_id in ids:
        if isinstance(recipe_id, int) and recipe_id in allowed and recipe_id not in result:
            result.append(recipe_id)
    return result
