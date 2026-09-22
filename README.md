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

The `data/` directory is ignored by Git, so each contributor should run the
script locally.

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
    popularity.py             # Example class skeleton; not implemented
scripts/
  download_data.py            # CSV → Parquet
pyproject.toml                # Shared dependencies and package configuration
```

Keep exploration in notebooks, reusable code in `src/remy/`, and runnable
workflows in `scripts/`.

## Add a model

Add a plain class in its own module under `src/remy/model/`. The `Popularity`
skeleton illustrates possible inputs and outputs:

- `fit(interactions)` accepts a training DataFrame with `user_id`, `recipe_id`,
  and `rating` columns and returns `self`.
- `recommend(user_id, k=10)` returns a ranked list of original recipe IDs.
  Popularity will return the same ranking for everyone.

The methods currently raise `NotImplementedError`; fill them in when ready.
There is no base class, training framework, or evaluation code. Keep
`__init__.py` files empty and use absolute imports:

```python
from remy.model.popularity import Popularity
```

## File paths

Use the shared root to build paths from any script or notebook:

```python
from remy.paths import PROJECT_ROOT

recipes_path = PROJECT_ROOT / "data" / "processed" / "recipes.parquet"
```

This resolves from the source file, so it does not depend on the working
directory. Use the editable installation described above.
