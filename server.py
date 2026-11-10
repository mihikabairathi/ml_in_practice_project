from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import random
import json
import time
from pathlib import Path

# Import your repository's existing modules
from remy.splits import RollingSplits
from remy.model.lightgbm_features import LightGBM

app = FastAPI()

# Enable CORS so your GitHub Pages frontend can communicate with this local API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RollingSplits & LightGBM on server startup
print("Loading Remy models into memory...")
rs = RollingSplits()
lgbm_model = LightGBM()

LOG_FILE = Path("ab_test_logs.jsonl")

class ColdStartPayload(BaseModel):
    user_id: str
    seed_ratings: Dict[int, int]  # recipe_id: rating (e.g., {123: 5, 456: 1})
    max_time: int = 60
    max_ingredients: int = 20

@app.post("/api/recommend")
def generate_recommendations(payload: ColdStartPayload):
    # 1. Determine A/B Variant silently on the server
    variant = random.choice(["Variant_A_Baseline", "Variant_B_LightGBM"])
    
    # 2. Get catalog recipes from splits
    df_recipes = rs.recipes.copy()
    
    # Standardize ID column
    if "id" in df_recipes.columns:
        df_recipes = df_recipes.rename(columns={"id": "recipe_id"})
    
    # Apply basic preference constraints
    df_filtered = df_recipes[
        (df_recipes["minutes"] <= payload.max_time) & 
        (df_recipes["n_ingredients"] <= payload.max_ingredients)
    ].copy()

    # 3. Perform model inference based on hidden variant
    if variant == "Variant_B_LightGBM":
        # Dynamic LightGBM scoring logic incorporating cold start ratings
        # (Using user seed ratings to weight feature matching)
        df_filtered["score"] = df_filtered["minutes"] * -0.01  # Placeholder for LGBM predict
    else:
        # Baseline fallback scoring
        df_filtered["score"] = df_filtered.get("rating_mean", 4.0)

    # 4. Rank and select Top 10 items
    top_items = df_filtered.sort_values(by="score", ascending=False).head(10)
    
    # Prepare clean payload (Opaque: NO variant info sent to frontend)
    recommendations = []
    for _, row in top_items.iterrows():
        recommendations.append({
            "recipe_id": int(row["recipe_id"]),
            "title": str(row.get("name", f"Recipe {row['recipe_id']}")),
            "minutes": int(row.get("minutes", 0)),
            "n_ingredients": int(row.get("n_ingredients", 0)),
        })

    # 5. Log telemetry internally to backend disk
    log_entry = {
        "timestamp": time.time(),
        "user_id": payload.user_id,
        "variant": variant,
        "seed_ratings": payload.seed_ratings,
        "recommended_ids": [r["recipe_id"] for r in recommendations],
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Return clean recommendation list to UI
    return {"status": "success", "recommendations": recommendations}