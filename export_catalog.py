import json
from pathlib import Path
import numpy as np
import pandas as pd

from remy.splits import RollingSplits, MIN_REVIEWS

OUTPUT_DIR = Path("docs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_JSON_PATH = OUTPUT_DIR / "catalog.json"


def clean_val(val):
    """Utility to make numpy/pandas values JSON serializable."""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return round(float(val), 4)
    if isinstance(val, (np.ndarray, list)):
        return [clean_val(v) for v in val]
    return val


def main():
    print("Initializing RollingSplits to access recipe dataset...")
    rs = RollingSplits()
    
    df_interactions = rs.interactions
    df_recipes = rs.recipes.copy()

    # Standardize primary key column name
    if "id" in df_recipes.columns and "recipe_id" not in df_recipes.columns:
        df_recipes = df_recipes.rename(columns={"id": "recipe_id"})
    elif "recipe_id" not in df_recipes.columns:
        df_recipes = df_recipes.reset_index()
        if "id" in df_recipes.columns:
            df_recipes = df_recipes.rename(columns={"id": "recipe_id"})

    print("Computing item summary statistics...")
    item_stats = (
        df_interactions.groupby("recipe_id")
        .agg(
            rating_mean=("rating", "mean"),
            rating_count=("rating", "count"),
        )
        .reset_index()
    )

    # Merge statistics back into recipes dataframe
    recipes_merged = df_recipes.merge(item_stats, on="recipe_id", how="left")
    recipes_merged["rating_count"] = recipes_merged["rating_count"].fillna(0).astype(int)
    recipes_merged["rating_mean"] = recipes_merged["rating_mean"].fillna(0.0)

    # Filter items with minimal review threshold
    recipes_merged = recipes_merged[recipes_merged["rating_count"] >= MIN_REVIEWS]

    catalog = []
    print(f"Exporting {len(recipes_merged)} catalog records...")
    
    for _, row in recipes_merged.iterrows():
        tags = row.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = eval(tags)
            except Exception:
                tags = []

        item = {
            "recipe_id": clean_val(row["recipe_id"]),
            "title": str(row.get("name", f"Recipe {row['recipe_id']}")),
            "minutes": clean_val(row.get("minutes", 0)),
            "n_ingredients": clean_val(row.get("n_ingredients", 0)),
            "n_steps": clean_val(row.get("n_steps", 0)),
            "rating_mean": clean_val(row["rating_mean"]),
            "rating_count": clean_val(row["rating_count"]),
            "tags": tags[:8] if isinstance(tags, list) else [],
        }
        catalog.append(item)

    with open(CATALOG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Successfully exported catalog to {CATALOG_JSON_PATH}")


if __name__ == "__main__":
    main()