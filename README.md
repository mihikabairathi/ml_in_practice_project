# Remy

CMU Machine Learning in Practice project: Food.com EDA and recipe recommendation models.

## Install dependencies

Run commands from the repository root. Use **Python 3.12** and choose either UV or conda. Both will install the project and all dependencies, including PyTorch and Jupyter, from `pyproject.toml`.

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

Both setups install the project in editable mode, so source edits are available without reinstalling. After changing dependencies in `pyproject.toml`, rerun `uv sync` or `python -m pip install -e .`. Environments and UV lockfiles stay local.

## Download the data

The download script fetches the
[Food.com Recipes and Interactions dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) from Kaggle and converts its two main CSV files to compressed Parquet files. If Kaggle asks you to authenticate, sign in to Kaggle and follow the KaggleHub authentication prompt.

Run the script from the repository root after installing the dependencies:

```bash
# UV
uv run python scripts/download_data.py

# Conda or another active environment
python scripts/download_data.py
```

The original download remains in KaggleHub's local cache. The converted files are written to:

```text
data/raw/recipes.parquet
data/raw/interactions.parquet
```

The dataset files are ignored by Git, while placeholder files preserve the
`data/raw/` and `data/processed/` directory structure. Each contributor should run the script locally.

## Project structure

```text
data/raw/                     # Generated Food.com Parquet files (ignored)
data/processed/               # Generated Parquet files (ignored)
notebooks/
  eda.ipynb                   # Exploration and plots
  rolling_popularity.ipynb    # Training and evaluation for the popularity model
  rolling_lightgbm.ipynb      # Training and evaluation for the LightGBM model
src/remy/
  __init__.py
  paths.py                    # Shared PROJECT_ROOT path
  model/
    __init__.py
    popularity.py             # Past-review-count baseline
    item_knn.py               # Item-to-item cosine collaborative filtering
    lightgbm_new.py           # LightGBM reranker on user stats + recipe tags
  evaluate.py                 # Common metric-computing function
  splits.py                   # Shared quarterly training/evaluation data
scripts/
  download_data.py            # CSV to Parquet
  make_splits.py              # Save the common quarterly split data
  run_experiments.py          # Compare models on rolling validation/test tasks
pyproject.toml                # Shared dependencies and package configuration
```

Keep exploration in notebooks, reusable code in `src/remy/`, and runnable
workflows in `scripts/`.

## Shared quarterly splits

After downloading the data, run `python scripts/make_splits.py`. It writes
interactions, recipe metadata, and 52 quarterly boundaries/counts to
`data/processed/rolling/` (ignored by Git).

Training tasks cover 2003–2006, validation 2007–2010, and test 2011–2015.
Earlier reviews remain history. The script's docstring explains the dates, rules, filtering, and labels.

## Run experiments

After downloading the data, prepare the shared rolling-quarter inputs once:

```bash
python scripts/make_splits.py
```

Run supported models together on the same validation quarters:

```bash
python scripts/run_experiments.py --models popularity item_knn lightgbm --split val
```

Other useful commands:

```bash
# A short LightGBM run to inspect the first two validation quarters
python scripts/run_experiments.py --models lightgbm --split val --max-quarters 2

# Override the LightGBM shortlist and tree settings
python scripts/run_experiments.py --models lightgbm --split val \
  --lgbm-candidates 2000 --num-leaves 63 --min-data-in-leaf 50

# After selecting settings on validation, use the same settings for test
python scripts/run_experiments.py --models popularity item_knn lightgbm --split test
```

Model settings exposed by the runner are:

| Model | Argument | Default | Meaning |
| --- | --- | ---: | --- |
| Item KNN | `--k-neighbors` | 50 | Neighbors retained per recipe |
| Item KNN | `--min-rating` | 5 | Minimum rating treated as positive |
| LightGBM | `--negative-samples-per-positive` | 10 | Sampled negatives per historical positive |
| LightGBM | `--boost-rounds` | 200 | Number of boosting rounds |
| LightGBM | `--lgbm-candidates` | 1000 | Popular recipes shortlisted for LightGBM to rank for each user |
| LightGBM | `--learning-rate` | 0.05 | Boosting learning rate |
| LightGBM | `--num-leaves` | 31 | Maximum leaves per tree |
| LightGBM | `--min-data-in-leaf` | 100 | Minimum training rows per leaf |
| LightGBM | `--seed` | 10718 | Random seed for negative sampling and LightGBM |

LightGBM's negative-sampling, rounds, and shortlist defaults match
`notebooks/rolling_lightgbm.ipynb`. It trains on all completed earlier training
quarters available at each checkpoint. Run
`python scripts/run_experiments.py --help` for the full argument list.

Each run creates a unique folder under `artifacts/` with `results.csv`,
`run_config.json`, a quarterly trend chart, a clustered hit-rate/recall chart,
and a separate NDCG chart. The charts use the EDA notebook's background, mint,
peach, and sand palette. Use `--run-name` to name the run folder,
`--artifacts-dir` to change its parent, or `--output` to choose a CSV path and
save charts beside it. `--no-plots` skips chart creation. Results report
full-ranking NDCG, hit rate, and recall at 10 and 100, averaged across eligible
users within each quarter; candidate recall is also reported for models that
shortlist recipes before the popularity fallback. On macOS, LightGBM may need
the OpenMP runtime (`libomp`) installed.

## Add a model

Implement the same small interface used by the experiment runner:

```python
fit(interactions, recipes=None, training_quarters=()) -> self
recommend_many(user_ids, k=None) -> {user_id: [recipe_id, ...]}
```

- `interactions` is the current quarter's history, which includes all ratings
  strictly before its start. `recipes` is the catalog available at that date.
- `training_quarters` contains completed history/label tasks for supervised
  models. Build each training example from that task's own history and labels;
  do not use the evaluation quarter's outcomes during `fit`.
- Store `ranking_`, a popularity-ordered list of all available recipe IDs, and
  `seen_`, a mapping from users to previously reviewed recipe IDs. These let the
  evaluator complete each model's short ranked candidate list with the same
  unseen-popularity fallback.
- `recommend_many` returns each requested user's distinct, unseen candidates in
  rank order. Return `{}` when the model uses the shared global ranking directly
  and has no separate candidate list, as the popularity baseline does.
- Use `remy.evaluate.evaluate_quarter` for metrics. The runner fits a fresh
  instance at every checkpoint and reports the mean of the per-user metrics for
  that quarter.

Register the class in `scripts/run_experiments.py`'s model choices and
`make_model`, then add its options to the argument parser and document the
defaults here. Keep model code in `src/remy/model/`, shared data preparation in
`src/remy/splits.py`, and evaluation logic in `src/remy/evaluate.py`. Keep
`__init__.py` files empty; no base class or model registry is needed. Choose
settings on validation and keep them fixed through the test run.

## File paths

Use the shared root to build paths from any script or notebook:

```python
from remy.paths import PROJECT_ROOT

recipes_path = PROJECT_ROOT / "data" / "processed" / "recipes.parquet"
```

This resolves from the source file, so it does not depend on the working
directory. Use the editable installation described above.
