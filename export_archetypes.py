import json
from pathlib import Path
import pandas as pd
import numpy as np

from remy.splits import RollingSplits
from remy.model.lightgbm_features import LightGBM

OUTPUT_DIR = Path("docs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARCHETYPES_JSON_PATH = OUTPUT_DIR / "archetypes.json"

# Defined persona profiles for pre-computation
ARCHETYPES = {
    "quick_easy": {
        "label": "Quick & Easy Meals",
        "description": "Short prep time and minimal ingredients.",
        "pref_minutes": 15,
        "pref_ingredients": 5,
    },
    "gourmet": {
        "label": "Gourmet Experience",
        "description": "Rich, multi-step detailed culinary recipes.",
        "pref_minutes": 60,
        "pref_ingredients": 12,
    },
    "healthy": {
        "label": "Light & Healthy",
        "description": "Low calories and fresh ingredients.",
        "pref_minutes": 30,
        "pref_ingredients": 7,
    },
}


def main():
    print("Loading data split for evaluation...")
    rs = RollingSplits()
    
    # Train/Eval baseline using lightgbm feature engineering
    lgbm = LightGBM()
    
    # Load catalog data
    catalog_path = OUTPUT_DIR / "catalog.json"
    if not catalog_path.exists():
        raise FileNotFoundError("Please run export_catalog.py first!")

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    df_catalog = pd.DataFrame(catalog)

    archetype_recommendations = {}

    for arch_key, arch_meta in ARCHETYPES.items():
        print(f"Generating recommendations for archetype: {arch_meta['label']}...")
        
        # Rule-based scoring fallback mixed with feature matching
        df_scored = df_catalog.copy()
        
        # Calculate feature similarity distance score for front-end fallback ranking
        time_diff = np.abs(df_scored["minutes"] - arch_meta["pref_minutes"])
        ing_diff = np.abs(df_scored["n_ingredients"] - arch_meta["pref_ingredients"])
        
        df_scored["score"] = (
            df_scored["rating_mean"] * 2.0 
            - (time_diff * 0.05) 
            - (ing_diff * 0.1)
        )
        
        top_items = (
            df_scored.sort_values(by="score", ascending=False)
            .head(10)[["recipe_id", "title", "minutes", "rating_mean", "score"]]
            .to_dict(orient="records")
        )

        archetype_recommendations[arch_key] = {
            "meta": arch_meta,
            "items": top_items,
        }

    with open(ARCHETYPES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(archetype_recommendations, f, indent=2)

    print(f"Successfully exported pre-calculated archetypes to {ARCHETYPES_JSON_PATH}")


if __name__ == "__main__":
    main()