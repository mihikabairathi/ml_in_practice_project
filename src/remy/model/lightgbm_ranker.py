"""LightGBM LambdaRank reranker over popularity + trending candidates.

Same interface as Popularity (fit / recommend), plus recommend_many for fast
batch scoring during evaluation.

Pipeline
--------
1. Candidate generation (per user): the top ``n_popular`` all-time most-reviewed
   unseen recipes plus the top ``n_trending`` unseen recipes by review count in
   the last ``trending_days`` days.
2. Features for each (user, recipe) pair, computed ONLY from reviews before the
   task's start date (no leakage).
3. A LightGBM ``lambdarank`` model, trained on earlier completed quarterly
   tasks (history -> next-quarter outcomes), reranks the candidates.

Recipes outside the candidate set are not scored by the model; recommend()
appends them in popularity order, and the evaluation in the notebook ranks them
the same way, so full-ranking NDCG stays exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # imported lazily-friendly so the module can be read without lightgbm installed
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


ITEM_FEATURES = [
    "item_log_cnt",
    "item_cnt_trend",
    "item_cnt_365",
    "item_mean_rating",
    "item_pos_rate",
    "item_age_days",
    "item_log_pop_rank",
]
USER_FEATURES = [
    "user_n_reviews",
    "user_n_90d",
    "user_mean_rating",
    "user_pos_rate",
    "user_days_since_last",
    "user_days_since_first",
    "user_mean_item_log_cnt",
]
PAIR_FEATURES = ["source", "cand_pos", "pop_vs_user_taste", "trend_ratio"]
FEATURES = ITEM_FEATURES + USER_FEATURES + PAIR_FEATURES

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 100],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 100,
    "verbosity": -1,
}


@dataclass
class _State:
    """Everything known as of one task's start date."""

    asof: pd.Timestamp
    item: pd.DataFrame  # indexed by recipe_id, ITEM_FEATURES columns
    user: pd.DataFrame  # indexed by user_id, USER_FEATURES columns
    seen: dict  # user_id -> set(recipe_id)
    pop_list: list  # all available recipes, most reviewed first
    trend_list: list  # recipes with recent reviews, most recent reviews first
    pop_rank: dict  # recipe_id -> 1-based position in pop_list
    scores: pd.Series  # all-time review counts, most reviewed first


class LightGBMRanker:
    """Rerank popular/trending candidates with a LightGBM LambdaRank model.

    Parameters
    ----------
    n_popular, n_trending: candidates per user from each source (unseen only).
    trending_days: lookback window for the trending source and trend features.
    n_train_quarters: use at most this many of the MOST RECENT completed
        training tasks (None = all). This is the "how many windows to train
        over" knob.
    max_train_users: sample at most this many users per training quarter.
    num_boost_round: boosting rounds.
    params: overrides merged into DEFAULT_PARAMS.
    seed: controls user sampling and LightGBM randomness.
    """

    def __init__(
        self,
        n_popular: int = 200,
        n_trending: int = 100,
        trending_days: int = 90,
        n_train_quarters: int | None = 8,
        max_train_users: int | None = 3000,
        num_boost_round: int = 200,
        params: dict | None = None,
        seed: int = 0,
    ):
        self.n_popular = n_popular
        self.n_trending = n_trending
        self.trending_days = trending_days
        self.n_train_quarters = n_train_quarters
        self.max_train_users = max_train_users
        self.num_boost_round = num_boost_round
        self.params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
        self.seed = seed

    # ------------------------------------------------------------------ state
    def _build_state(self, history: pd.DataFrame, recipes, asof) -> _State:
        asof = pd.Timestamp(asof)
        h = history[history["date"] < asof]
        rated = h[h["rating"] > 0]  # rating 0 = unrated

        counts = h["recipe_id"].value_counts()
        if recipes is not None:
            counts = counts.reindex(recipes["id"], fill_value=0)
        counts = counts.sort_index().sort_values(ascending=False, kind="stable")
        pop_list = counts.index.tolist()
        pop_rank = {r: i for i, r in enumerate(pop_list, start=1)}

        trend_cut = asof - pd.Timedelta(days=self.trending_days)
        year_cut = asof - pd.Timedelta(days=365)
        trend_cnt = h.loc[h["date"] >= trend_cut, "recipe_id"].value_counts()
        year_cnt = h.loc[h["date"] >= year_cut, "recipe_id"].value_counts()
        trend_cnt = trend_cnt[trend_cnt.index.isin(counts.index)]
        trend_list = trend_cnt.sort_values(ascending=False, kind="stable").index.tolist()

        item = pd.DataFrame(index=counts.index)
        item["item_log_cnt"] = np.log1p(counts)
        item["item_cnt_trend"] = trend_cnt.reindex(item.index, fill_value=0)
        item["item_cnt_365"] = year_cnt.reindex(item.index, fill_value=0)
        g = rated.groupby("recipe_id")["rating"]
        item["item_mean_rating"] = g.mean().reindex(item.index)
        item["item_pos_rate"] = (rated["rating"] == 5).groupby(rated["recipe_id"]).mean().reindex(item.index)
        if recipes is not None:
            sub = recipes.set_index("id")["submitted"].reindex(item.index)
            item["item_age_days"] = (asof - pd.to_datetime(sub)).dt.days
        else:
            first = h.groupby("recipe_id")["date"].min().reindex(item.index)
            item["item_age_days"] = (asof - first).dt.days
        item["item_log_pop_rank"] = np.log1p(pd.Series(pop_rank).reindex(item.index))

        user = pd.DataFrame(index=h["user_id"].unique())
        ug = h.groupby("user_id")
        user["user_n_reviews"] = ug.size()
        user["user_n_90d"] = h.loc[h["date"] >= trend_cut].groupby("user_id").size().reindex(user.index, fill_value=0)
        user["user_mean_rating"] = rated.groupby("user_id")["rating"].mean().reindex(user.index)
        user["user_pos_rate"] = (rated["rating"] == 5).groupby(rated["user_id"]).mean().reindex(user.index)
        user["user_days_since_last"] = (asof - ug["date"].max()).dt.days
        user["user_days_since_first"] = (asof - ug["date"].min()).dt.days
        log_cnt = h["recipe_id"].map(item["item_log_cnt"])
        user["user_mean_item_log_cnt"] = log_cnt.groupby(h["user_id"]).mean().reindex(user.index)

        seen = h.groupby("user_id")["recipe_id"].agg(set).to_dict()
        return _State(asof, item[ITEM_FEATURES], user[USER_FEATURES], seen, pop_list, trend_list, pop_rank, counts)

    # ------------------------------------------------------- candidates + X
    def _candidate_frame(self, state: _State, user_ids) -> pd.DataFrame:
        """One row per (user, candidate) with all FEATURES filled in."""
        us, rs, src, pos = [], [], [], []
        for u in user_ids:
            seen = state.seen.get(u, ())
            picked = []
            for r in state.pop_list:
                if len(picked) == self.n_popular:
                    break
                if r not in seen:
                    picked.append(r)
            chosen = set(picked)
            n = len(picked)
            us.extend([u] * n)
            rs.extend(picked)
            src.extend([0] * n)
            pos.extend(range(n))
            trend = []
            for r in state.trend_list:
                if len(trend) == self.n_trending:
                    break
                if r not in seen and r not in chosen:
                    trend.append(r)
            n = len(trend)
            us.extend([u] * n)
            rs.extend(trend)
            src.extend([1] * n)
            pos.extend(range(n))
        df = pd.DataFrame({"user_id": us, "recipe_id": rs, "source": src, "cand_pos": pos})
        df = df.join(state.item, on="recipe_id").join(state.user, on="user_id")
        df["pop_vs_user_taste"] = df["item_log_cnt"] - df["user_mean_item_log_cnt"]
        df["trend_ratio"] = df["item_cnt_trend"] / (df["item_cnt_365"] + 1.0)
        return df

    # -------------------------------------------------------------------- fit
    def fit(self, interactions: pd.DataFrame, *, recipes=None, training_quarters=(), asof=None) -> "LightGBMRanker":
        """Fit on earlier quarterly tasks; keep state from `interactions`.

        interactions / recipes: the CURRENT quarter's history and recipe metadata.
        training_quarters: completed Quarter tasks (each has history, outcomes,
            recipes, users, start). Only the last n_train_quarters are used.
        asof: the current task's start date; defaults to the day after the
            latest interaction.
        """
        if asof is None:
            asof = interactions["date"].max().normalize() + pd.Timedelta(days=1)
        self.state_ = self._build_state(interactions, recipes, asof)
        self.pop_rank_ = self.state_.pop_rank
        self.scores_ = self.state_.scores  # review counts, like Popularity.scores_
        self.ranking_ = self.state_.pop_list
        self.seen_ = self.state_.seen

        quarters = list(training_quarters)
        if self.n_train_quarters is not None:
            quarters = quarters[-self.n_train_quarters :]
        self.n_train_quarters_used_ = len(quarters)
        self.model_ = None
        if not quarters:  # nothing to learn from: behaves like Popularity
            return self
        if lgb is None:
            raise ImportError("lightgbm is required to fit the ranker")

        rng = np.random.default_rng(self.seed)
        frames, group_sizes = [], []
        for q in quarters:
            pos = q.outcomes[q.outcomes["rating"] == 5][["user_id", "recipe_id"]].drop_duplicates()
            users = pos["user_id"].unique()
            if self.max_train_users is not None and len(users) > self.max_train_users:
                users = rng.choice(users, self.max_train_users, replace=False)
            qstate = self._build_state(q.history, q.recipes, q.start)
            df = self._candidate_frame(qstate, np.sort(users))
            pos = pos.assign(label=1)
            df = df.merge(pos, on=["user_id", "recipe_id"], how="left")
            df["label"] = df["label"].fillna(0).astype(int)
            # Rows for a user are contiguous (see _candidate_frame); drop users
            # whose positives are all outside the candidate set (no signal).
            has_pos = df.groupby("user_id", sort=False)["label"].transform("max") > 0
            df = df[has_pos]
            group_sizes.extend(df.groupby("user_id", sort=False).size().tolist())
            frames.append(df)
        train = pd.concat(frames, ignore_index=True)
        self.train_rows_ = len(train)
        self.train_groups_ = len(group_sizes)
        dtrain = lgb.Dataset(train[FEATURES], label=train["label"], group=group_sizes)
        self.model_ = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        return self

    # --------------------------------------------------------------- inference
    def recommend_many(self, user_ids, k: int | None = None) -> dict:
        """Ranked candidate lists per user (model-scored candidates only).

        Anything not listed is implicitly ranked after, in popularity order.
        """
        self._check_fit()
        user_ids = list(user_ids)
        df = self._candidate_frame(self.state_, user_ids)
        if self.model_ is None:
            df["score"] = df["item_log_cnt"] - 1e-6 * df["source"]
        else:
            df["score"] = self.model_.predict(df[FEATURES])
        df = df.sort_values(["user_id", "score"], ascending=[True, False], kind="stable")
        if k is not None:
            df = df.groupby("user_id", sort=False).head(k)
        out = df.groupby("user_id", sort=False)["recipe_id"].agg(list).to_dict()
        return {u: out.get(u, []) for u in user_ids}

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Up to k available, unseen recipe IDs: reranked candidates first,
        then popularity order if k exceeds the candidate count."""
        if k < 0:
            raise ValueError("k must be nonnegative")
        self._check_fit()
        result = self.recommend_many([user_id])[user_id][:k]
        if len(result) < k:
            have, seen = set(result), self.seen_.get(user_id, set())
            for r in self.ranking_:
                if len(result) == k:
                    break
                if r not in seen and r not in have:
                    result.append(r)
        return result

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        self._check_fit()
        if self.model_ is None:
            raise RuntimeError("No trained model (no training quarters were provided)")
        return pd.Series(
            self.model_.feature_importance(importance_type), index=FEATURES
        ).sort_values(ascending=False)

    def _check_fit(self):
        if not hasattr(self, "state_"):
            raise RuntimeError("Call fit before recommend")
