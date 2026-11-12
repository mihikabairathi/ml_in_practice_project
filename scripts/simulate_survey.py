"""Fill the A/B survey with simulated participants, to try the app and analysis.

Each simulated participant has a hidden taste (a few liked recipe tags) and a
personal rating bias. They answer the quiz and rate both lists according to
that taste, so a model that personalises from the quiz can genuinely do better.
Participants go through the real app (start -> quiz -> compare), using the
models and settings in config.yaml, but responses go to a separate file.

    python scripts/simulate_survey.py                       # 40 participants
    python scripts/simulate_survey.py -n 100 --output-dir data/processed/abtest/my_samples

Each run writes a new responses_<timestamp>.jsonl in the output folder, the same
way each start of the real server does.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import yaml

from remy.paths import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "abtest" / "sample_responses"
TASTE_TAGS = [
    "chicken", "beef", "pork", "seafood", "vegetarian", "pasta", "desserts", "breakfast",
    "mexican", "italian", "asian", "soups-stews", "salads", "chocolate", "breads",
    "cheese", "potatoes", "crock-pot-slow-cooker", "healthy", "comfort-food",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-n", "--participants", type=int, default=40)
    parser.add_argument("--retake-share", type=float, default=0.2,
                        help="Share of participants who take the survey twice (same ?pid=)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Folder for the responses file (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--config", type=Path, help="Survey config (default: src/remy/abtest/config.yaml)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Same config as the real survey, but write responses to --output-dir.
    from remy.abtest.config import DEFAULT_CONFIG
    config = yaml.safe_load((args.config or DEFAULT_CONFIG).read_text())
    config["responses_dir"] = str(args.output_dir.resolve())
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f)
    os.environ["REMY_ABTEST_CONFIG"] = f.name

    from fastapi.testclient import TestClient
    from remy.abtest.app import COOKIE, app

    rng = np.random.default_rng(args.seed)
    n_retakes = int(round(args.participants * args.retake_share))
    runs = list(range(args.participants)) + list(rng.choice(args.participants, n_retakes, replace=False))
    personas = {
        i: {"tags": set(rng.choice(TASTE_TAGS, 3, replace=False)), "bias": rng.normal(0, 0.5)}
        for i in range(args.participants)
    }

    with TestClient(app) as client:
        recipes = app.state.book.recipes

        def rate(person, recipe_id):
            match = len(person["tags"] & set(recipes.at[recipe_id, "tags"]))
            score = 2.3 + 1.1 * min(match, 2) + person["bias"] + rng.normal(0, 0.8)
            return int(np.clip(round(score), 1, 5))

        for n, i in enumerate(runs, start=1):
            person = personas[i]
            page = client.post("/start", data={"external_id": f"sim-{i:03d}"}).text
            quiz = sorted(set(map(int, re.findall(r'name="r_(\d+)"', page))))
            page = client.post("/quiz", data={f"r_{r}": rate(person, r) for r in quiz}).text

            lists = app.state.log.sessions[client.cookies[COOKIE]]["lists"]
            form, means = {}, {}
            for side, recipe_ids in lists.items():
                ratings = [rate(person, r) for r in recipe_ids]
                form |= {f"item_{side}_{rank}": v for rank, v in enumerate(ratings, start=1)}
                means[side] = np.mean(ratings)
                form[f"list_{side}"] = int(np.clip(round(means[side] + rng.normal(0, 0.5)), 1, 5))
            gap = means["left"] - means["right"]
            form["preference"] = "none" if abs(gap) < 0.2 else ("left" if gap > 0 else "right")
            assert client.post("/compare", data=form).url.path == "/done"
            print(f"\r{n}/{len(runs)} surveys", end="", flush=True)

    os.unlink(f.name)
    print(f"\nWrote {len(runs)} surveys ({args.participants} participants) to {app.state.log.path}")
    print(f"Analyse every run in that folder:\n  REMY_ABTEST_RESPONSES={args.output_dir.resolve()} "
          "uv run jupyter lab notebooks/abtest_analysis.ipynb")


if __name__ == "__main__":
    main()
