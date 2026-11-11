"""Training data at the configured cutoff, plus recipe cards for display."""

from __future__ import annotations

import datetime as dt
import random
import re
import string
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from remy.splits import Quarter, RollingSplits

# Course buckets, in priority order: the first tag a recipe has decides its bucket.
COURSES = {
    "beverages": "Drinks",
    "desserts": "Dessert",
    "breakfast": "Breakfast",
    "soups-stews": "Soup & stew",
    "salads": "Salad",
    "appetizers": "Appetizer",
    "breads": "Bread",
    "side-dishes": "Side dish",
    "main-dish": "Main dish",
}
# Structural or vague Food.com tags that say little about the dish.
GENERIC_TAGS = {
    "preparation", "time-to-make", "course", "main-ingredient", "dietary", "occasion",
    "cuisine", "low-in-something", "high-in-something", "free-of-something", "equipment",
    "number-of-servings", "taste-mood", "healthy-2", "presentation", "technique",
    "seasonal", "holiday-event", "served-hot", "served-cold", "to-go", "north-american",
    "american", "oven", "stove-top", "for-1-or-2", "for-large-groups", "beginner-cook",
    "low-protein", "low-sodium", "low-cholesterol", "low-saturated-fat", "low-calorie",
    "low-fat", "healthy", "easy", "inexpensive", "dinner-party", "kid-friendly",
    *COURSES,
}
MIN_TAG_COUNT = 200  # ignore very rare tags when picking chips
NUTRIENTS = {  # index in Food.com's nutrition list -> label (all but calories are % daily value)
    1: "Fat", 2: "Sugar", 3: "Sodium", 4: "Protein", 6: "Carbs",
}


@dataclass
class TrainingData:
    """What every model's fit() sees: the same rules as remy.splits.Quarter."""

    cutoff: pd.Timestamp
    history: pd.DataFrame          # reviews strictly before cutoff
    catalog: pd.DataFrame          # recipes submitted strictly before cutoff
    training_quarters: list[Quarter] = field(default_factory=list)  # supervised models only


def load_training_data(cutoff: dt.date, supervised: bool) -> TrainingData:
    splits = RollingSplits()
    cutoff = pd.Timestamp(cutoff)
    last = splits.windows["end"].max()
    if cutoff > last:
        raise ValueError(f"train_cutoff {cutoff.date()} is after the split data ends ({last.date()})")
    history = splits.interactions[splits.interactions["date"] < cutoff]
    catalog = splits.recipes[splits.recipes["submitted"] < cutoff]
    quarters = splits.training_quarters(cutoff) if supervised else []
    if supervised and not quarters:
        raise ValueError(f"No completed training quarters before {cutoff.date()}")
    return TrainingData(cutoff, history, catalog, quarters)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return re.sub(r" ([,.;:!?)])", r"\1", text)


def _sentence(text: str) -> str:
    """Food.com text is lower-cased; restore sentence capitals and a capital I."""
    text = re.sub(r"\bi\b", "I", _clean(text))
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text[:1].upper() + text[1:]


def _title(name: str) -> str:
    """Names lost their apostrophes ("kittencal s meatballs"); restore them and title-case."""
    return string.capwords(re.sub(r"(\w) s\b", r"\1's", _clean(name)))


def _truncate(text: str, limit: int = 220) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def format_minutes(minutes: int) -> str | None:
    if minutes <= 0 or minutes > 3 * 24 * 60:  # the data has some absurd outliers
        return None
    hours, mins = divmod(int(minutes), 60)
    if not hours:
        return f"{mins} min"
    return f"{hours} h" + (f" {mins} min" if mins else "")


class RecipeBook:
    """Turns recipe rows into display-ready card dicts and samples quiz recipes."""

    def __init__(self, data: TrainingData, quiz_pool_size: int):
        history = data.history
        rated = history[history["rating"] > 0]
        self.n_reviews = history["recipe_id"].value_counts()
        self.avg_rating = rated.groupby("recipe_id")["rating"].mean()
        self.recipes = data.catalog.set_index("id")
        tag_counts = self.recipes["tags"].explode().value_counts()
        self.tag_counts = tag_counts[tag_counts >= MIN_TAG_COUNT]

        top = self.n_reviews.reindex(self.recipes.index, fill_value=0).nlargest(quiz_pool_size)
        self.quiz_pool: dict[str, list[int]] = {}
        for recipe_id in top.index:
            self.quiz_pool.setdefault(self.course(recipe_id), []).append(int(recipe_id))

    def course(self, recipe_id: int) -> str:
        tags = set(self.recipes.at[recipe_id, "tags"])
        return next((label for tag, label in COURSES.items() if tag in tags), "Other")

    def sample_quiz(self, n: int, seed: int) -> list[int]:
        """Round-robin over course buckets so the quiz covers different kinds of dishes."""
        rng = random.Random(seed)
        buckets = [rng.sample(ids, len(ids)) for ids in self.quiz_pool.values()]
        rng.shuffle(buckets)
        chosen = []
        while len(chosen) < n and any(buckets):
            for bucket in buckets:
                if bucket and len(chosen) < n:
                    chosen.append(bucket.pop())
        return chosen

    def card(self, recipe_id: int) -> dict:
        row = self.recipes.loc[recipe_id]
        tags = [t for t in row["tags"] if t not in GENERIC_TAGS and t in self.tag_counts.index
                and not t.endswith("-or-less")]
        tags = sorted(tags, key=lambda t: self.tag_counts[t])[:4]
        nutrition = np.asarray(row["nutrition"], dtype=float)
        avg = self.avg_rating.get(recipe_id)
        return {
            "id": int(recipe_id),
            "name": _title(row["name"]),
            "course": self.course(recipe_id),
            "description": _truncate(_sentence(row["description"])),
            "time": format_minutes(row["minutes"]),
            "n_ingredients": int(row["n_ingredients"]),
            "n_steps": int(row["n_steps"]),
            "calories": round(nutrition[0]) if len(nutrition) else None,
            "nutrients": [
                {"label": label, "pdv": round(nutrition[i])}
                for i, label in NUTRIENTS.items() if i < len(nutrition)
            ],
            "tags": [t.replace("-", " ") for t in tags],
            "ingredients": [_clean(i) for i in row["ingredients"]],
            "steps": [_sentence(s) for s in row["steps"]],
            "n_reviews": int(self.n_reviews.get(recipe_id, 0)),
            "avg_rating": None if avg is None or np.isnan(avg) else round(float(avg), 1),
        }
