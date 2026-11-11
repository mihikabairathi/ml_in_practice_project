"""Check that A/B survey wrappers serve new users exactly like the fitted model.

For each model, real training users are copied into a fresh "new" user through
SurveyModel.recommend (add_user -> model.recommend -> remove_user). The
recommendations must match the model's own recommend() for the real user, and
the model's state must be unchanged afterwards. Run this after adding a model to
remy.abtest.models.MODELS or changing a model's per-user state.

    python scripts/check_survey_models.py                      # every registered model
    python scripts/check_survey_models.py --models lightgbm --cutoff 2008-01-01
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd

from remy.abtest.data import load_training_data
from remy.abtest.models import MODELS

NEW_USER_ID = -1


def state_snapshot(model) -> dict:
    """Sizes of everything a wrapper might touch, to confirm remove_user undid add_user."""
    snap = {}
    for name, value in vars(model).items():
        if isinstance(value, dict) and name == "state_":
            snap.update({f"state_.{k}": getattr(v, "shape", None) or len(v)
                         for k, v in value.items() if hasattr(v, "__len__") or hasattr(v, "shape")})
        elif hasattr(value, "shape"):
            snap[name] = value.shape
        elif isinstance(value, (dict, list, set, pd.Series, pd.DataFrame)):
            snap[name] = len(value)
    return snap


def check(name: str, data, n_users: int, k: int, seed: int) -> None:
    wrapper = MODELS[name]()
    start = time.time()
    wrapper.fit_once(data)
    print(f"{name}: fit in {time.time() - start:.1f}s")

    history = data.history
    counts = history.groupby("user_id").size()
    rng = np.random.default_rng(seed)
    # Mix of light users (like survey participants) and heavier ones
    light = counts[counts.between(3, 10)].index.to_numpy()
    heavy = counts[counts.between(11, 200)].index.to_numpy()
    users = np.concatenate([rng.choice(light, n_users // 2, replace=False),
                            rng.choice(heavy, n_users - n_users // 2, replace=False)])

    before = state_snapshot(wrapper.model)
    seen_before = copy.copy(wrapper.model.seen_)
    times = []
    for user_id in users:
        rows = history[history["user_id"] == user_id].assign(user_id=NEW_USER_ID)
        expected = [int(r) for r in wrapper.model.recommend(user_id, k=k)]
        start = time.time()
        got = wrapper.recommend(rows, k=k)
        times.append(time.time() - start)
        if got != expected:
            raise AssertionError(f"{name}: user {user_id} differs\n  model:   {expected}\n  wrapper: {got}")
    after = state_snapshot(wrapper.model)
    if before != after or wrapper.model.seen_.keys() != seen_before.keys():
        changed = {key: (before.get(key), after.get(key)) for key in before.keys() | after.keys()
                   if before.get(key) != after.get(key)}
        raise AssertionError(f"{name}: model state not restored: {changed}")
    print(f"{name}: OK - {len(users)} copied users match; state restored; "
          f"median recommend {1000 * np.median(times):.1f} ms")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--cutoff", default="2016-01-01", help="Training cutoff (default: 2016-01-01)")
    parser.add_argument("--n-users", type=int, default=40, help="Real users to copy per model")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    supervised = any(MODELS[name].supervised for name in args.models)
    data = load_training_data(pd.Timestamp(args.cutoff).date(), supervised)
    for name in args.models:
        check(name, data, args.n_users, args.k, args.seed)


if __name__ == "__main__":
    main()
