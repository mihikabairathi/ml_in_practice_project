# Remy

CMU Machine Learning in Practice project: Food.com EDA and recipe recommendation models.

## Install dependencies

Run commands from the repository root. Use **Python 3.12** and choose either UV
or conda. Both install the project and all dependencies, including PyTorch and
Jupyter, from `pyproject.toml`.

### UV

```bash
uv sync --python 3.12
```

### Conda

```bash
conda create -n remy python=3.12 pip
conda activate remy
python -m pip install -e .
```

Both setups install the project in editable mode, so source edits are available
without reinstalling. After changing dependencies in `pyproject.toml`, rerun
`uv sync` or `python -m pip install -e .`. Environments and UV lockfiles stay local.

## Download the data

The download script fetches the
[Food.com Recipes and Interactions dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
from Kaggle and converts its two main CSV files to compressed Parquet files.
If Kaggle asks you to authenticate, sign in to Kaggle and follow the KaggleHub
authentication prompt.

Run the script from the repository root after installing the dependencies:

```bash
# UV
uv run python scripts/download_data.py

# Conda or another active environment
python scripts/download_data.py
```

The original download remains in KaggleHub's local cache. The converted files
are written to:

```text
data/raw/recipes.parquet
data/raw/interactions.parquet
```

The dataset files are ignored by Git, while placeholder files preserve the
`data/raw/` and `data/processed/` directory structure. Each contributor should
run the script locally.

Note: if you are having trouble activating environments, you can run this locally instead: `PYTHONPATH=src python scripts/download_data.py`

## Project structure

```text
data/raw/                     # Generated Food.com Parquet files (ignored)
data/processed/               # Generated Parquet files (ignored)
notebooks/eda.ipynb         # Exploration and plots
src/remy/
  __init__.py
  paths.py                    # Shared PROJECT_ROOT path
  model/
    __init__.py
    popularity.py             # Past-review-count baseline
  splits.py                   # Shared quarterly training/evaluation data
scripts/
  download_data.py            # CSV → Parquet
  make_splits.py              # Save the common quarterly split data
pyproject.toml                # Shared dependencies and package configuration
```

Keep exploration in notebooks, reusable code in `src/remy/`, and runnable
workflows in `scripts/`.

## Shared quarterly splits

After downloading the data, run `python scripts/make_splits.py`. It writes
interactions, recipe metadata, and 52 quarterly boundaries/counts to
`data/processed/rolling/` (ignored by Git).

Training tasks cover 2003–2006, validation 2007–2010, and test 2011–2015.
Earlier reviews remain history. Users need three prior reviews of any rating;
positives are 5-star reviews in the next quarter on available, unseen recipes.
The script's docstring explains the dates and rules.

See [the rolling popularity notebook](notebooks/rolling_popularity.ipynb)
for a short worked example, quarterly metrics, and charts. Existing exploratory
notebooks retain their old splits; use the shared API for comparable experiments.

## Add a model

Keep a plain class with `fit` and `recommend`, following `Popularity`:

- `fit(interactions, *, recipes=None, training_quarters=())` returns `self`.
  Supervised models build features from each training quarter's own history
  and recipes, and labels from its outcomes. Popularity counts past reviews.
- `recommend(user_id, k=10)` returns ranked, distinct recipe IDs from the
  available catalog, excluding that user's previously reviewed recipes.
  Include recipes with no past reviews using a fallback score if necessary.

`RollingSplits()` loads the generated files. Its two public methods are
`training_quarters(cutoff)` and `evaluation_quarters("val" or "test")`.
Both return lists of `Quarter` objects, with `start`, `end`, `history`, `recipes`,
`users` (including prior review counts), and `outcomes`.

Create a fresh model at every evaluation quarter. Fit using its history and
completed earlier training quarters; use its outcomes only for scoring.
Completed validation/test quarters become training history at later cutoffs.
Choose settings on validation and keep them fixed throughout test. Keep
`__init__.py` files empty; no base class or registry is needed.

## File paths

Use the shared root to build paths from any script or notebook:

```python
from remy.paths import PROJECT_ROOT

recipes_path = PROJECT_ROOT / "data" / "processed" / "recipes.parquet"
```

This resolves from the source file, so it does not depend on the working
directory. Use the editable installation described above.
