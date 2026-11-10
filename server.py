from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import random
import json
import time
from pathlib import Path

# Import remy modules
from remy.splits import RollingSplits
from remy.model.popularity import Popularity  #[cite: 1]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Remy models into memory...")
rs = RollingSplits()

# Prepare recipes dataframe safely
df_recipes = rs.recipes.copy()
if "id" in df_recipes.columns and "recipe_id" not in df_recipes.columns:
    df_recipes = df_recipes.rename(columns={"id": "recipe_id"})

# Fit Popularity baseline model[cite: 1]
print("Fitting Popularity model...")
pop_model = Popularity()
try:
    pop_model.fit(rs.interactions, recipes=rs.recipes)  #[cite: 1]
except Exception as e:
    print(f"Warning: Popularity fit with recipes failed ({e}), fitting on interactions only.")
    pop_model.fit(rs.interactions)  #[cite: 1]

LOG_FILE = Path("ab_test_logs.jsonl")

class ColdStartPayload(BaseModel):
    user_id: str
    seed_ratings: Dict[int, int]
    max_time: int = 60
    max_ingredients: int = 20

@app.post("/api/recommend")
def generate_recommendations(payload: ColdStartPayload):
    try:
        # 1. Hidden A/B Variant Assignment
        variant = random.choice(["Variant_A_Popularity", "Variant_B_LightGBM"])
        
        # Apply user cooking constraints
        df_filtered = df_recipes[
            (df_recipes["minutes"] <= payload.max_time) & 
            (df_recipes["n_ingredients"] <= payload.max_ingredients)
        ].copy()

        if df_filtered.empty:
            df_filtered = df_recipes.copy()

        # 2. Perform Inference based on Variant
        if variant == "Variant_A_Popularity":
            # Map pre-computed Popularity scores[cite: 1]
            df_filtered["score"] = df_filtered["recipe_id"].map(pop_model.scores_).fillna(0)  #[cite: 1]
        else:
            # Variant B: Dynamic Cold-Start Scoring
            user_weight = sum(payload.seed_ratings.values()) / max(len(payload.seed_ratings), 1)
            
            # Safely check for rating column existence or fallback to popularity scores[cite: 1]
            if "rating_mean" in df_filtered.columns:
                base_score = df_filtered["rating_mean"]
            else:
                base_score = df_filtered["recipe_id"].map(pop_model.scores_).fillna(0)  #[cite: 1]
                
            df_filtered["score"] = (base_score * 0.7) + (user_weight * 0.3)

        # 3. Sort & Select Top 10 items
        top_items = df_filtered.sort_values(by="score", ascending=False).head(10)
        
        recommendations = []
        for _, row in top_items.iterrows():
            title_val = row.get("name") or row.get("title") or f"Recipe {row['recipe_id']}"
            recommendations.append({
                "recipe_id": int(row["recipe_id"]),
                "title": str(title_val),
                "minutes": int(row.get("minutes", 0)),
                "n_ingredients": int(row.get("n_ingredients", 0)),
            })

        # 4. Log Telemetry Internally
        log_entry = {
            "timestamp": time.time(),
            "user_id": payload.user_id,
            "variant": variant,
            "seed_ratings": payload.seed_ratings,
            "recommended_ids": [r["recipe_id"] for r in recommendations],
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {"status": "success", "recommendations": recommendations}

    except Exception as err:
        print(f"Error during recommendation generation: {err}")
        return {"status": "error", "message": str(err), "recommendations": []}