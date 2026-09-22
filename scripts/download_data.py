"""Download Food.com CSVs, convert data types, and store compressed Parquet."""

import ast
from pathlib import Path

import kagglehub
import pandas as pd

from remy.paths import PROJECT_ROOT


def safe_eval_list(val):
    """Safely parse stringified Python lists of strings."""
    if pd.isna(val) or not isinstance(val, str):
        return []
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


def parse_numeric_list(val):
    """Safely parse stringified Python list of numbers into a list of floats."""
    parsed = safe_eval_list(val)
    if isinstance(parsed, list):
        try:
            return [float(x) for x in parsed]
        except (ValueError, TypeError):
            return []
    return []


def process_recipes(csv_path: Path, output_path: Path):
    print(f"⌛️ Processing {csv_path.name}...")
    df = pd.read_csv(csv_path)

    # Identifiers & numerical metadata
    df["id"] = df["id"].astype("int64")
    df["contributor_id"] = df["contributor_id"].astype("int64")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(-1).astype("int32")
    df["n_steps"] = pd.to_numeric(df["n_steps"], errors="coerce").fillna(-1).astype("int16")
    df["n_ingredients"] = pd.to_numeric(df["n_ingredients"], errors="coerce").fillna(-1).astype("int16")

    # Dates
    df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")

    # String arrays (tags, steps, ingredients)
    for col in ["tags", "steps", "ingredients"]:
        if col in df.columns:
            print(f"Parsing string array column: {col}")
            df[col] = df[col].apply(safe_eval_list)

    # Keep nutrition as a single column of float array: list
    # Format: [calories, fat_pdv, sugar_pdv, sodium_pdv, protein_pdv, sat_fat_pdv, carbs_pdv]
    if "nutrition" in df.columns:
        print("Parsing numeric array column: nutrition")
        df["nutrition"] = df["nutrition"].apply(parse_numeric_list)

    # Text fields
    df["name"] = df["name"].fillna("").astype("string")
    df["description"] = df["description"].fillna("").astype("string")

    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    print(f"🍔 Saved: {output_path} ({len(df):,} rows)")


def process_interactions(csv_path: Path, output_path: Path):
    print(f"⌛️ Processing {csv_path.name}...")
    df = pd.read_csv(csv_path)

    df["user_id"] = df["user_id"].astype("int64")
    df["recipe_id"] = df["recipe_id"].astype("int64")
    df["rating"] = df["rating"].astype("int8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["review"] = df["review"].fillna("").astype("string")

    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    print(f"🐀 Saved: {output_path} ({len(df):,} rows)")


def main():
    # Download dataset via kagglehub
    print("⌛️ Downloading dataset from Kaggle...")
    raw_dir = Path(kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions"))
    print(f"💾 Files downloaded to cache: {raw_dir}")

    # Set output destination
    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process primary files
    recipes_csv = raw_dir / "RAW_recipes.csv"
    interactions_csv = raw_dir / "RAW_interactions.csv"

    if recipes_csv.exists():
        process_recipes(recipes_csv, output_dir / "recipes.parquet")

    if interactions_csv.exists():
        process_interactions(interactions_csv, output_dir / "interactions.parquet")

    print(f"✅ Parquet datasets ready at: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
