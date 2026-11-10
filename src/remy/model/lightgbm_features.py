"""LightGBM reranker with user, recipe and user x recipe interaction features.

Feature-toggleable version of the baseline remy/model/lightgbm.py (same API).
Defaults reproduce the original model; only `feature_groups` changes the inputs.
Save as remy/model/lightgbm_features.py and keep the original file untouched.

Everything fed to the model at time T is computed only from interactions
strictly before T:
  * training quarter q -> features from quarter.history (before q.start),
    labels from quarter.outcomes (during q)
  * inference          -> features from `interactions` (before the eval quarter)
"""
from __future__ import annotations

import ast
from collections.abc import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from remy.splits import MIN_REVIEWS

DATE_COLS = ("date", "review_date", "timestamp", "created_at")

# ---- feature groups (toggle with feature_groups=...) ----
GROUP_COLS = {
    # original baseline features
    "user_base": ["user_n_reviews", "user_mean_rating", "user_pos_rate"],
    "tags": [],  # one-hot recipe tags, appended separately
    # new user features
    "user_activity": ["user_std_rating", "user_days_since_last", "user_tenure_days",
                      "user_n_reviews_90d", "user_n_reviews_365d"],
    "user_taste": ["user_n_liked", "user_pref_log_minutes", "user_pref_n_ingredients",
                   "user_pref_log_calories", "user_pref_log_pop"],
    # new recipe features (recomputed from history at every cutoff)
    "recipe_pop": ["recipe_log_pop", "recipe_pop_90d", "recipe_pop_365d", "recipe_log_pop_rank",
                   "recipe_recent_share", "recipe_age_days", "recipe_days_since_last"],
    "recipe_rating": ["recipe_pos_rate", "recipe_mean_rating"],
    # extra popularity signals (round 2)
    "pop_season": ["recipe_season_share", "recipe_season_log_count", "cutoff_quarter"],
    "pop_trend": ["recipe_trend_ratio", "recipe_decay_pop"],
    "pop_5star": ["recipe_log_pos_count", "recipe_log_pos_count_90d"],
    # static recipe metadata
    "recipe_static": ["recipe_log_minutes", "recipe_n_steps", "recipe_n_ingredients",
                      "recipe_log_calories", "recipe_n_tags"],
    # user x recipe interactions
    "inter_tag": ["tag_cosine", "tag_affinity"],
    "inter_diff": ["diff_minutes", "diff_ingredients", "diff_calories", "diff_pop"],
}
ALL_GROUPS = tuple(GROUP_COLS)
BASELINE_GROUPS = ("user_base", "tags")  # reproduces the original model's features
USER_FEATURES = (GROUP_COLS["user_base"] + GROUP_COLS["user_activity"] + GROUP_COLS["user_taste"])
RECIPE_STATE_FEATURES = (GROUP_COLS["recipe_pop"] + GROUP_COLS["recipe_rating"] + GROUP_COLS["pop_season"]
                          + GROUP_COLS["pop_trend"] + GROUP_COLS["pop_5star"])
# round-1 'all' (before the extra popularity groups), for comparisons with earlier results
ALL_ROUND1 = tuple(g for g in GROUP_COLS if not g.startswith("pop_"))
RECIPE_STATIC_FEATURES = GROUP_COLS["recipe_static"]
INTERACTION_FEATURES = GROUP_COLS["inter_tag"] + GROUP_COLS["inter_diff"]
PREF_COLS = ["user_pref_log_minutes", "user_pref_n_ingredients",
             "user_pref_log_calories", "user_pref_log_pop"]


def _calories(value):
    """First entry of the Food.com `nutrition` field (calories); NaN if unparseable."""
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return float(value[0])
    except Exception:
        return np.nan


class LightGBM:
    """
    LightGBM reranker over a popularity shortlist.

    Training examples: for every completed quarter, each user's new 5-star recipes
    (that they had not reviewed before) are positives. Negatives are a mix of
    random available recipes and "hard" negatives drawn from that quarter's
    popularity shortlist, so training matches the shortlist reranking task at
    inference. Users with < MIN_REVIEWS history get the popularity fallback.
    """

    def __init__(
        self,
        negatives_per_positive=10,
        hard_negative_frac=0.0,          # 0 = uniform random negatives (original)
        num_boost_round=200,
        n_candidates=1000,
        seed=10718,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        max_depth=-1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        lambda_l1=0.0,
        lambda_l2=0.0,
        objective="binary",          # "binary" or "lambdarank"
        min_tag_count=1,             # 1 = keep every tag (original)
        liked_min_rating=4,          # rating that counts as "liked" for taste profiles
        prior_strength=10,           # pseudo-count for smoothed recipe rating stats
        feature_groups=ALL_GROUPS,   # which feature groups the model sees
        filter_seen_positives=False, # original kept positives the user had reviewed before
        num_threads=0,
    ):
        self.negatives_per_positive = negatives_per_positive
        self.hard_negative_frac = hard_negative_frac
        self.num_boost_round = num_boost_round
        self.n_candidates = n_candidates
        self.seed = seed
        self.min_tag_count = min_tag_count
        self.liked_min_rating = liked_min_rating
        self.prior_strength = prior_strength
        self.feature_groups = tuple(feature_groups)
        self.filter_seen_positives = filter_seen_positives
        self.objective = objective
        if objective not in ("binary", "lambdarank"):
            raise ValueError("objective must be 'binary' or 'lambdarank'")
        self.params = {
            "objective": objective,
            "metric": "binary_logloss" if objective == "binary" else "ndcg",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "max_depth": max_depth,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "num_threads": num_threads,
            "verbosity": -1,
            "seed": seed,
        }

    # ------------------------------------------------------------------ fit
    def fit(self, interactions: pd.DataFrame, recipes=None, training_quarters=(), cutoff=None) -> "LightGBM":
        """
        interactions / recipes: what is known at the prediction cutoff.
        training_quarters: completed quarters (history -> features, outcomes -> labels).
        cutoff: start of the quarter being predicted (used for "days since" features).
                Defaults to one day after the latest interaction.
        """
        if recipes is None:
            raise ValueError("recipes with a 'tags' column are required")
        if not training_quarters:
            raise ValueError("At least one completed training quarter is required")

        self.date_col_ = self._find_date_col(interactions)
        dc = self.date_col_
        self.cutoff_ = (pd.Timestamp(cutoff) if cutoff is not None
                        else pd.to_datetime(interactions[dc]).max() + pd.Timedelta(days=1))

        # ---- static catalog structures (tags, metadata) ----
        recipes = recipes.drop_duplicates("id").reset_index(drop=True)
        self.recipe_ids_ = recipes["id"].to_numpy()
        self.recipe_row_ = pd.Series(np.arange(len(recipes)), index=recipes["id"])
        self._catalog_set = set(self.recipe_ids_.tolist())

        tags = recipes["tags"].map(self.as_tag_list)
        enc = MultiLabelBinarizer(sparse_output=True)
        mat = enc.fit_transform(tags).astype(np.float32).tocsr()
        keep = np.flatnonzero(np.asarray(mat.sum(axis=0)).ravel() >= self.min_tag_count)
        self.tag_names_ = list(enc.classes_[keep])
        self.recipe_tags_ = mat[:, keep].tocsr()
        self.recipe_tags_norm_ = normalize(self.recipe_tags_)
        self.recipe_n_tags_ = np.maximum(np.asarray(self.recipe_tags_.sum(axis=1)).ravel(), 1.0)
        self.recipe_static_ = self._static(recipes, tags)

        # ---- which columns the model sees ----
        bad = set(self.feature_groups) - set(ALL_GROUPS)
        if bad:
            raise ValueError(f"unknown feature groups: {bad}")
        self.dense_cols_ = [c for g in ALL_GROUPS if g in self.feature_groups for c in GROUP_COLS[g]]
        self.use_tags_ = "tags" in self.feature_groups
        self.feature_names_ = self.dense_cols_ + ([f"tag_{t}" for t in self.tag_names_] if self.use_tags_ else [])

        # ---- current state used at recommendation time ----
        self.state_ = self._state(interactions, self.cutoff_)
        self.seen_ = interactions.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self.user_features_ = self.state_["user"]
        self.available_recipes_ = recipes["id"].tolist()

        counts = interactions["recipe_id"].value_counts().reindex(recipes["id"], fill_value=0)
        counts = counts.sort_index().sort_values(ascending=False, kind="stable")
        self.ranking_ = counts.index.tolist()
        self.candidates_ = self.ranking_[: self.n_candidates]

        # ---- training set ----
        rng = np.random.default_rng(self.seed)
        X_parts, y_parts, group_sizes = [], [], []
        for quarter in list(training_quarters):
            out = self._quarter_examples(quarter, rng)
            if out is None:
                continue
            X_q, y_q, g_q = out
            X_parts.append(X_q)
            y_parts.append(y_q)
            group_sizes.extend(g_q)

        if not X_parts:
            raise ValueError("Training quarters contain no interactions")
        X_train = sparse.vstack(X_parts, format="csr")
        y_train = np.concatenate(y_parts)
        if np.unique(y_train).size < 2:
            raise ValueError("Training data must contain both positive and negative examples")

        train = lgb.Dataset(
            X_train, label=y_train, feature_name=self.feature_names_,
            group=group_sizes if self.objective == "lambdarank" else None,
        )
        self.model_ = lgb.train(self.params, train, num_boost_round=self.num_boost_round)
        return self

    def _quarter_examples(self, quarter, rng):
        """Positive/negative (user, recipe) pairs + features for one completed quarter."""
        positives = quarter.outcomes.loc[
            quarter.outcomes["rating"] == 5, ["user_id", "recipe_id"]
        ].drop_duplicates()
        if positives.empty:
            return None

        st = self._state(quarter.history, quarter.start)
        seen = quarter.history.groupby("user_id")["recipe_id"].agg(set).to_dict()
        available = np.asarray([r for r in quarter.recipes["id"].tolist() if r in self._catalog_set])
        if len(available) == 0:
            return None
        # quarter's popularity shortlist = hard-negative pool (mirrors inference candidates)
        q_counts = quarter.history["recipe_id"].value_counts()
        hard_pool = np.asarray([r for r in q_counts.index[: self.n_candidates] if r in self._catalog_set])

        users, recs, labels, groups = [], [], [], []
        for user_id, user_pos in positives.groupby("user_id"):
            blocked = seen.get(user_id, set())
            pos_ids = [r for r in user_pos["recipe_id"].tolist()
                       if r in self._catalog_set and not (self.filter_seen_positives and r in blocked)]
            if not pos_ids:
                continue
            n_neg = self.negatives_per_positive * len(pos_ids)
            n_hard = int(round(n_neg * self.hard_negative_frac)) if len(hard_pool) else 0
            taken = set(blocked) | set(pos_ids)
            neg = self._sample(rng, hard_pool, taken, n_hard)
            taken |= set(neg)
            neg += self._sample(rng, available, taken, n_neg - len(neg))

            ids = pos_ids + neg
            users.extend([user_id] * len(ids))
            recs.extend(ids)
            labels.extend([1] * len(pos_ids) + [0] * len(neg))
            groups.append(len(ids))

        if not recs:
            return None
        X = self._features(users, recs, st)
        return X, np.asarray(labels, dtype=np.int8), groups

    @staticmethod
    def _sample(rng, pool, taken, n):
        out, taken = [], set(taken)
        if n <= 0 or len(pool) == 0:
            return out
        for _ in range(5):
            for r in rng.choice(pool, size=max(2 * (n - len(out)), 8)):
                if r not in taken:
                    taken.add(r)
                    out.append(r)
                    if len(out) == n:
                        return out
        return out

    # ------------------------------------------------------------ recommend
    def recommend(self, user_id: int, k: int | None = 10) -> list[int]:
        """Rank shortlisted unseen recipes for one user (popularity fallback if < MIN_REVIEWS)."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before recommend")
        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        seen = self.seen_.get(user_id, set())
        unseen = [r for r in self.candidates_ if r not in seen]

        if user_id not in self.user_features_.index or len(seen) < MIN_REVIEWS:
            ranked = [r for r in self.ranking_ if r not in seen]
        else:
            X = self._features([user_id] * len(unseen), unseen, self.state_)
            scores = self.model_.predict(X)
            ranked = [unseen[i] for i in np.lexsort((np.asarray(unseen), -scores))]
        return ranked if k is None else ranked[:k]

    def recommend_batch(self, user_ids: Iterable[int], k: int | None = None) -> dict[int, list[int]]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before recommend_batch")
        if k is not None and k < 0:
            raise ValueError("k must be nonnegative")
        return {u: self.recommend(u, k=k) for u in user_ids}

    # ------------------------------------------------------------- features
    def _find_date_col(self, df):
        for c in DATE_COLS:
            if c in df.columns:
                return c
        raise ValueError(f"No date column found in interactions (looked for {DATE_COLS}); "
                         "set DATE_COLS in lightgbm.py")

    def _static(self, recipes, tags):
        """Time-invariant recipe metadata, indexed like the catalog. Missing columns -> NaN."""
        n = len(recipes)
        nan = np.full(n, np.nan)
        minutes = recipes["minutes"].to_numpy(float) if "minutes" in recipes else nan
        steps = recipes["n_steps"].to_numpy(float) if "n_steps" in recipes else nan
        ingr = recipes["n_ingredients"].to_numpy(float) if "n_ingredients" in recipes else nan
        cal = recipes["nutrition"].map(_calories).to_numpy(float) if "nutrition" in recipes else nan
        return pd.DataFrame({
            "recipe_log_minutes": np.log1p(np.clip(minutes, 0, None)),
            "recipe_n_steps": steps,
            "recipe_n_ingredients": ingr,
            "recipe_log_calories": np.log1p(np.clip(cal, 0, None)),
            "recipe_n_tags": tags.map(len).to_numpy(float),
        }, index=recipes["id"])

    def _state(self, history: pd.DataFrame, cutoff) -> dict:
        """Everything derived from `history` (all rows strictly before `cutoff`)."""
        dc = self.date_col_
        ids = self.recipe_ids_
        cutoff = pd.Timestamp(cutoff)
        h = history[["user_id", "recipe_id", "rating", dc]].copy()
        h[dc] = pd.to_datetime(h[dc])
        rated = h[h["rating"] > 0]
        d90, d365 = cutoff - pd.Timedelta(days=90), cutoff - pd.Timedelta(days=365)

        # ---- users ----
        g = h.groupby("user_id")
        user = pd.DataFrame({
            "user_n_reviews": g.size(),
            "user_days_since_last": (cutoff - g[dc].max()).dt.days,
            "user_tenure_days": (cutoff - g[dc].min()).dt.days,
        })
        gr = rated.groupby("user_id")["rating"]
        user["user_mean_rating"] = gr.mean()
        user["user_std_rating"] = gr.std()
        user["user_pos_rate"] = (rated["rating"] == 5).groupby(rated["user_id"]).mean()
        user["user_n_reviews_90d"] = h[h[dc] >= d90].groupby("user_id").size()
        user["user_n_reviews_365d"] = h[h[dc] >= d365].groupby("user_id").size()
        user[["user_n_reviews_90d", "user_n_reviews_365d"]] = (
            user[["user_n_reviews_90d", "user_n_reviews_365d"]].fillna(0))

        # ---- recipes (time-varying) ----
        rg = h.groupby("recipe_id")
        cnt = rg.size().reindex(ids, fill_value=0)
        c90 = h[h[dc] >= d90].groupby("recipe_id").size().reindex(ids, fill_value=0)
        c365 = h[h[dc] >= d365].groupby("recipe_id").size().reindex(ids, fill_value=0)
        n_rated = rated.groupby("recipe_id").size().reindex(ids, fill_value=0)
        n_pos = (rated["rating"] == 5).groupby(rated["recipe_id"]).sum().reindex(ids, fill_value=0)
        s_rat = rated.groupby("recipe_id")["rating"].sum().reindex(ids, fill_value=0)
        gp = float((rated["rating"] == 5).mean()) if len(rated) else 0.5
        gm = float(rated["rating"].mean()) if len(rated) else 4.0
        m = self.prior_strength
        first = rg[dc].min().reindex(ids)
        last = rg[dc].max().reindex(ids)
        cnt_v = cnt.to_numpy(float)

        rec = pd.DataFrame(index=ids)
        rec["recipe_log_pop"] = np.log1p(cnt_v)
        rec["recipe_pop_90d"] = np.log1p(c90.to_numpy(float))
        rec["recipe_pop_365d"] = np.log1p(c365.to_numpy(float))
        rec["recipe_log_pop_rank"] = np.log1p(cnt.rank(ascending=False, method="min").to_numpy())
        rec["recipe_pos_rate"] = (n_pos.to_numpy(float) + m * gp) / (n_rated.to_numpy(float) + m)
        rec["recipe_mean_rating"] = (s_rat.to_numpy(float) + m * gm) / (n_rated.to_numpy(float) + m)
        rec["recipe_age_days"] = (cutoff - first).dt.days.to_numpy(float)
        rec["recipe_days_since_last"] = (cutoff - last).dt.days.to_numpy(float)
        rec["recipe_recent_share"] = np.where(cnt_v > 0, c90.to_numpy(float) / np.maximum(cnt_v, 1), np.nan)

        # ---- extra popularity signals (all from rows before cutoff) ----
        # trend: last 90 days vs the 90 days before that; decay: exponentially weighted count (half-life 90d)
        prev90 = (h[(h[dc] >= cutoff - pd.Timedelta(days=180)) & (h[dc] < d90)]
                  .groupby("recipe_id").size().reindex(ids, fill_value=0))
        rec["recipe_trend_ratio"] = np.log((c90.to_numpy(float) + 1) / (prev90.to_numpy(float) + 1))
        w = pd.Series(0.5 ** ((cutoff - h[dc]).dt.days.to_numpy(float) / 90.0), index=h.index)
        rec["recipe_decay_pop"] = np.log1p(w.groupby(h["recipe_id"]).sum().reindex(ids, fill_value=0).to_numpy(float))
        # 5-star popularity (matches the 5-star target)
        five = h[h["rating"] == 5]
        rec["recipe_log_pos_count"] = np.log1p(
            five.groupby("recipe_id").size().reindex(ids, fill_value=0).to_numpy(float))
        rec["recipe_log_pos_count_90d"] = np.log1p(
            five[five[dc] >= d90].groupby("recipe_id").size().reindex(ids, fill_value=0).to_numpy(float))
        # seasonality: share of the recipe's past reviews in the same calendar quarter as the cutoff
        # (the quarter being predicted is known in advance; counts use earlier years only)
        same_q = (h[h[dc].dt.quarter == cutoff.quarter]
                  .groupby("recipe_id").size().reindex(ids, fill_value=0).to_numpy(float))
        rec["recipe_season_share"] = (same_q + m * 0.25) / (cnt_v + m)  # smoothed toward uniform 0.25
        rec["recipe_season_log_count"] = np.log1p(same_q)
        rec["cutoff_quarter"] = float(cutoff.quarter)

        # ---- user taste profiles from liked recipes ----
        liked = h.loc[h["rating"] >= self.liked_min_rating, ["user_id", "recipe_id"]].drop_duplicates()
        liked = liked[liked["recipe_id"].isin(self._catalog_set)]
        codes, uniq = pd.factorize(liked["user_id"])
        rows = self.recipe_row_.reindex(liked["recipe_id"]).to_numpy(dtype=int)
        P = sparse.csr_matrix((np.ones(len(codes), np.float32), (codes, rows)),
                              shape=(len(uniq), len(ids)))
        n_liked = np.asarray(P.sum(axis=1)).ravel()
        tag_counts = (P @ self.recipe_tags_).tocsr()
        freq = sparse.diags(1.0 / np.maximum(n_liked, 1)) @ tag_counts
        zero = sparse.csr_matrix((1, tag_counts.shape[1]))  # row for users with no liked recipes
        freq = sparse.vstack([freq, zero]).tocsr()
        norm = sparse.vstack([normalize(tag_counts), zero]).tocsr()

        S = self.recipe_static_
        M = np.column_stack([S["recipe_log_minutes"].to_numpy(), S["recipe_n_ingredients"].to_numpy(),
                             S["recipe_log_calories"].to_numpy(), rec["recipe_log_pop"].to_numpy()])
        num = P @ np.nan_to_num(M)
        den = P @ (~np.isnan(M)).astype(np.float32)
        pref = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        pref_df = pd.DataFrame(pref, index=uniq, columns=PREF_COLS)
        pref_df["user_n_liked"] = n_liked
        user = user.join(pref_df)
        user["user_n_liked"] = user["user_n_liked"].fillna(0)

        return {
            "user": user, "recipe": rec, "freq": freq, "norm": norm,
            "urow": pd.Series(np.arange(len(uniq)), index=uniq), "zero_row": len(uniq),
        }

    def _features(self, user_ids, recipe_ids, st) -> sparse.csr_matrix:
        user_ids, recipe_ids = np.asarray(user_ids), np.asarray(recipe_ids)
        u = st["user"].reindex(user_ids).reset_index(drop=True)
        r = st["recipe"].reindex(recipe_ids).reset_index(drop=True)
        s = self.recipe_static_.reindex(recipe_ids).reset_index(drop=True)
        rrows = self.recipe_row_.reindex(recipe_ids).to_numpy(dtype=int)
        urows = st["urow"].reindex(user_ids).fillna(st["zero_row"]).to_numpy(dtype=int)

        df = pd.concat([u[USER_FEATURES], r[RECIPE_STATE_FEATURES], s[RECIPE_STATIC_FEATURES]], axis=1)
        df["tag_cosine"] = np.asarray(
            st["norm"][urows].multiply(self.recipe_tags_norm_[rrows]).sum(axis=1)).ravel()
        df["tag_affinity"] = np.asarray(
            st["freq"][urows].multiply(self.recipe_tags_[rrows]).sum(axis=1)).ravel() / self.recipe_n_tags_[rrows]
        df["diff_minutes"] = s["recipe_log_minutes"] - u["user_pref_log_minutes"]
        df["diff_ingredients"] = s["recipe_n_ingredients"] - u["user_pref_n_ingredients"]
        df["diff_calories"] = s["recipe_log_calories"] - u["user_pref_log_calories"]
        df["diff_pop"] = r["recipe_log_pop"] - u["user_pref_log_pop"]

        blocks = [sparse.csr_matrix(df[self.dense_cols_].to_numpy(np.float32))]
        if self.use_tags_:
            blocks.append(self.recipe_tags_[rrows])
        return sparse.hstack(blocks, format="csr")

    @staticmethod
    def as_tag_list(value):
        """Normalize the parquet tag value to a list without changing tag content."""
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        return list(value) if not isinstance(value, str) else [value]
