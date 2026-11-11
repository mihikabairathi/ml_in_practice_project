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
  lightgbm_feature_experiments_part{1,2}.ipynb # LightGBM feature-group ablations
  abtest_analysis.ipynb       # Analysis of A/B survey responses
  rolling_embedding.ipynb     # Frozen embedding evaluation
  rolling_embedding_training.ipynb # Trained scoring over frozen inputs
src/remy/
  __init__.py
  paths.py                    # Shared PROJECT_ROOT path
  model/
    __init__.py
    popularity.py             # Past-review-count baseline
    embedding.py              # Frozen embedding recommendation
    embedding_trained.py      # Learned linear/nonlinear scoring
    knn.py                    # Item-to-item cosine collaborative filtering
    lightgbm.py               # LightGBM reranker with user, recipe and user x recipe features
    llm.py                    # Zero-shot LLM picks 10 recipes from a popularity shortlist
    llm_prompts.py            # Prompt text for llm.py
  abtest/                     # Blind A/B survey web app (see "Blind A/B survey")
  evaluate.py                 # Common metric-computing function
  plots.py                    # Saved experiment charts
  splits.py                   # Shared quarterly training/evaluation data
scripts/
  download_data.py            # CSV to Parquet
  make_splits.py              # Save the common quarterly split data
  embed_recipes.py            # Save MiniLM or TF-IDF recipe inputs
  run_experiments.py          # Compare models on rolling validation/test tasks
  run_llm_baseline.py         # Zero-shot LLM baseline on val + test (GPU)
  check_survey_models.py      # Verify A/B survey wrappers match each model's own recommend()
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

## Embedding recommender

Generate inputs once with `python scripts/embed_recipes.py --device mps` (or `--kind tfidf` for TF-IDF). Run `notebooks/rolling_embedding.ipynb` for frozen similarity or `notebooks/rolling_embedding_training.ipynb` for trained scoring. Edit `model_kwargs` in the notebook, then run its cells for all 16 validation quarters.

## Run experiments

After downloading the data, prepare the shared rolling-quarter inputs once:

```bash
python scripts/make_splits.py
```

Run supported models together on the same validation quarters:

```bash
python scripts/run_experiments.py --models popularity knn lightgbm --split val
```

Other useful commands:

```bash
# A short LightGBM run to inspect the first two validation quarters
python scripts/run_experiments.py --models lightgbm --split val --max-quarters 2

# Override the shared candidate shortlist and LightGBM tree settings
python scripts/run_experiments.py --models lightgbm --split val \
  --n-candidates 2000 --num-leaves 63 --min-data-in-leaf 50

# After selecting settings on validation, use the same settings for test
python scripts/run_experiments.py --models popularity knn lightgbm --split test
```

Model settings exposed by the runner are:

| Model | Argument | Default | Meaning |
| --- | --- | ---: | --- |
| KNN and LightGBM | `--n-candidates` | 1000 | Popular recipes shared as the candidate shortlist |
| KNN | `--k-neighbors` | 50 | Similar recipes retained per recipe in the item graph |
| KNN | `--min-rating` | 5 | Minimum rating treated as positive |
| LightGBM | `--negative-samples-per-positive` | 10 | Sampled negatives per historical positive |
| LightGBM | `--boost-rounds` | 200 | Number of boosting rounds |
| LightGBM | `--learning-rate` | 0.05 | Boosting learning rate |
| LightGBM | `--num-leaves` | 31 | Maximum leaves per tree |
| LightGBM | `--min-data-in-leaf` | 100 | Minimum training rows per leaf |
| LightGBM | `--seed` | 10718 | Random seed for negative sampling and LightGBM |

LightGBM's negative-sampling, rounds, and shortlist defaults match
`notebooks/rolling_lightgbm.ipynb`. It trains on all completed earlier training
quarters available at each checkpoint. Run
`python scripts/run_experiments.py --help` for the full argument list.
KNN and LightGBM use the same popularity-ordered candidate shortlist. KNN ranks
those candidates by collaborative-filter score; `--k-neighbors` controls how
many neighbors each recipe keeps in the similarity graph.

Each run creates a unique folder under `artifacts/` with `results.csv`,
`run_config.json`, a quarterly trend chart, a clustered hit-rate/recall chart,
and a two-panel clustered metric chart with hit rate and recall beside NDCG.
The charts use the EDA notebook's background, mint, peach, and sand palette,
with no gridlines. Use `--run-name` to name the run folder,
`--artifacts-dir` to change its parent, or `--output` to choose a CSV path and
save charts beside it. `--no-plots` skips chart creation. Results report
full-ranking NDCG, NDCG at the shared candidate cutoff, hit rate, and recall at
10 and 100. The shared cutoff is `--n-candidates` (1,000 by default) for every
model, including Popularity. Scores are averaged across users with a 5-star
outcome within each quarter. On macOS, LightGBM may need
the OpenMP runtime (`libomp`) installed.

## Add a model

Implement the same small interface used by the experiment runner:

```python
fit(interactions, recipes=None, training_quarters=()) -> self
recommend(user_id, k=10) -> [recipe_id, ...]
recommend_batch(user_ids, k=None) -> {user_id: [recipe_id, ...]}
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
- `recommend` is the simple single-user API and returns up to `k` distinct,
  unseen recipe IDs in rank order. `recommend_batch` accepts multiple users and
  returns their recommendations in the same mapping format; the experiment
  runner uses this method consistently for every model. Pass `k=None` to return
  each user's full ranked list before the evaluator applies any global fallback.
- Use `remy.evaluate.evaluate_quarter` for metrics. The runner fits a fresh
  instance at every checkpoint and reports the mean of the per-user metrics for
  that quarter.

Register the class in `scripts/run_experiments.py`'s model choices and
`make_model`, then add its options to the argument parser and document the
defaults here. Put model code in `src/remy/model/`, chart code in
`src/remy/plots.py`, shared data preparation in `src/remy/splits.py`, and
evaluation logic in `src/remy/evaluate.py`. Keep `__init__.py` files empty; no
base class or model registry is needed. Choose settings on validation and keep
them fixed through the test run.

### Make a model available in the A/B survey

The survey (`remy.abtest`) uses models unchanged. It fits each model once with
`fit(history, recipes=catalog, training_quarters=...)`, then serves a brand-new
participant by temporarily adding them to the fitted model's **per-user state**,
calling the model's own `recommend(user_id, k)`, and removing them again. All of
this lives in a small wrapper class in `src/remy/abtest/models.py`:

```python
class MyModelSurvey(SurveyModel):
    model_class = MyModel        # the class from src/remy/model/
    supervised = False           # True if fit() needs training_quarters

    def add_user(self, ratings):          # ratings: user_id, recipe_id, date, rating (one user)
        super().add_user(ratings)         # sets model.seen_[user_id]
        ...                               # add anything else recommend() looks up per user

    def remove_user(self, user_id):
        super().remove_user(user_id)      # undo add_user exactly
        ...

MODELS["mymodel"] = MyModelSurvey        # then use `type: mymodel` in config.yaml
```

To write `add_user`, look at what your `recommend(user_id)` reads that `fit`
built from `interactions`, and build the same thing for the new user's rows.
Reuse the model's own helper for this when it has one (see `LightGBMSurvey`).
Learned item-side state (similarities, trees, rankings) is never touched.

- If `recommend` only uses `seen_` (like Popularity), set `model_class` and stop
  there.
- KNN adds a row to its user-item matrix.
- LightGBM adds rows to its user feature table and tag-taste matrices.

Then check that the wrapper serves new users exactly like the model does:

```bash
uv run python scripts/check_survey_models.py --models mymodel
```

The check copies real training users into a new user and requires identical
recommendations and fully restored model state. Fitted models are cached by
cutoff, params and a hash of the model's source file, so editing a model makes
the survey refit it automatically.

## File paths

Use the shared root to build paths from any script or notebook:

```python
from remy.paths import PROJECT_ROOT

recipes_path = PROJECT_ROOT / "data" / "processed" / "recipes.parquet"
```

This resolves from the source file, so it does not depend on the working
directory. Use the editable installation described above.

## Zero-shot LLM baseline

`remy.model.llm.LLMRecommender` shows an instruct LLM (default `Qwen/Qwen3-8B`) the
200 most-reviewed recipes a user has not reviewed yet, plus their 30 most recent
4-5 star recipes, and asks for 10 recipe IDs. Guided decoding forces valid JSON
with IDs from that list. Nothing is trained. The prompts live in
`src/remy/model/llm_prompts.py`.

It needs a CUDA GPU (one 48GB L40S is plenty) and vLLM:

```bash
python -m pip install -e ".[llm]"
python scripts/run_llm_baseline.py --run-dir artifacts/llm_baseline
```

A full val + test run takes roughly 8-10 GPU hours with Qwen3-8B on one L40S. Rerunning with the
same `--run-dir` resumes after the last finished quarter. Only 10 recipes are
returned per user, so compare models on HR@10, Recall@10 and NDCG@10; the @100
and full-NDCG numbers mostly reflect the popularity fallback after those 10.

## Blind A/B survey

`remy.abtest` is a small FastAPI web app for human evaluation of two recommenders.
Each participant:

1. Rates a short cold-start quiz of popular recipes, spread across courses, from 1 to 5 stars.
   The `n_recs` most-reviewed recipes are skipped, so Popularity can always show its true top picks.
2. Sees two lists side by side, "List 1" and "List 2", one from each model.
   Which model goes on the left is random, and model names never reach the browser.
3. Rates every recipe, rates each list overall, and says which list they'd rather cook from.

### How it works

- **Training.** At startup, each model is fit once on `RollingSplits` data before `train_cutoff`,
  using its normal `fit`. Fitted models are cached in `data/processed/abtest/models/`,
  so later restarts take seconds.
- **Serving a participant.** Their quiz answers become rows in the usual interactions schema.
  The model's wrapper (`remy/abtest/models.py`) adds them to the fitted model's per-user state,
  calls the model's own `recommend`, and removes them. Each request takes a few to about 100 ms.
  See "Make a model available in the A/B survey" above.
- **Sessions.** In-progress surveys live in memory and are lost if the server restarts.
  Every run of the survey is a new user. The thank-you page offers "Take it again".
- **Output.** Each completed survey is appended as one JSON line to
  `data/processed/abtest/responses.jsonl`. The line records:
  - user ID and optional `external_id`
  - timestamps
  - cutoff, both models and their params
  - quiz ratings
  - both lists, with each list's model, side and overall rating, and every item's rank and rating
  - the preferred model

### Configure

Edit `src/remy/abtest/config.yaml`:

- `train_cutoff`
- the two models (`models.a` / `models.b`: `type` from `MODELS` plus constructor `params`)
- the quiz and list sizes
- whether to show community stats
- the output paths

To use another file, set `REMY_ABTEST_CONFIG=/path/to/config.yaml`. To record an external participant ID
(e.g. a class roster ID), add `?pid=<id>` to the invite link.

### Run locally

The app needs `data/processed/rolling/`, made by `download_data.py` then `make_splits.py`.

```bash
uv sync --extra abtest
uv run uvicorn remy.abtest.app:app --port 8000      # then open http://localhost:8000
```

The first start fits the models: about 2 minutes for Popularity plus LightGBM at the 2016 cutoff.
Wait for `Survey ready` in the log. `GET /healthz` returns `{"ok": true}` once the app is serving.

### Deploy

Requirements, whatever the host:

- **One process.** Surveys in progress are held in memory. Run a single uvicorn worker
  (no `--workers N`, no multiple replicas). One process handles many concurrent participants.
- **About 4 GB of RAM** for the training data and fitted models (about 3 GB measured when loading cached models; the first LightGBM fit needs more headroom). Also about 3 GB of disk
  for the Python environment, the rolling data and the model cache.
- **A persistent disk** for `responses.jsonl` and the model cache. Hosts with ephemeral disks
  lose responses on every restart.
- **The data.** Either run `download_data.py` and `make_splits.py` on the host
  (needs Kaggle credentials), or copy `data/processed/rolling/` there.
  Also copy `data/processed/abtest/models/` to skip the first fit; this needs the same package versions.
- **HTTPS** if participants join over the internet.

The options, from least to most setup:

| Option | Good for | Notes |
|---|---|---|
| **Laptop + tunnel** | A pilot or a short, supervised study | Run the app locally, then expose it with `cloudflared tunnel --url http://localhost:8000` (Cloudflare Tunnel, free) or `ngrok http 8000`. You get an HTTPS URL to share. The laptop must stay awake and online. |
| **A small VM** (lab or university server, AWS Lightsail/EC2, GCP, DigitalOcean; 4 GB RAM) | A study that runs for days or weeks | The most robust option; setup is below. |
| **PaaS with a persistent disk** (Render web service + disk, Fly.io + volume) | If you'd rather not manage a server | Choose an instance with at least 4 GB of RAM. Put `data/processed/` on the mounted disk: set `responses_path` and `model_cache_dir` there, and upload the rolling data to it or symlink. Start command: `uvicorn remy.abtest.app:app --host 0.0.0.0 --port $PORT`. |

Avoid free or serverless hosts (Hugging Face Spaces free tier, Streamlit Cloud, Vercel, Lambda).
Their disks are wiped, memory is too small, and in-memory sessions don't survive between requests.

VM setup (Ubuntu):

```bash
# 1. Code, environment and data
git clone <repo-url> remy && cd remy
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra abtest
rsync -a <laptop>:remy/data/processed/rolling/ data/processed/rolling/   # or download_data.py + make_splits.py

# 2. Keep it running across logouts and reboots (or use tmux for a quick test)
sudo tee /etc/systemd/system/remy-survey.service >/dev/null <<UNIT
[Unit]
Description=Remy A/B survey
After=network.target
[Service]
WorkingDirectory=$PWD
ExecStart=$PWD/.venv/bin/uvicorn remy.abtest.app:app --host 127.0.0.1 --port 8000 --proxy-headers
Restart=on-failure
User=$USER
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl enable --now remy-survey
journalctl -u remy-survey -f        # wait for "Survey ready"

# 3. HTTPS: point a DNS name at the VM, then let Caddy fetch a certificate
sudo apt install -y caddy
echo 'survey.example.org {
  reverse_proxy localhost:8000
}' | sudo tee /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

While the study runs:

- **Back up the results.** Copy `data/processed/abtest/responses.jsonl` off the machine regularly,
  for example with a daily `rsync` or `cron` job.
- **Changing the config.** Changing `config.yaml` mid-study needs a restart.
  The analysis notebook keeps only the most recent config's responses.

### Analyse

Open `notebooks/abtest_analysis.ipynb`. It runs paired, per-participant comparisons with CIs,
paired t and Wilcoxon tests, preference sign tests, rating-by-rank plots and sanity checks.
Set `REMY_ABTEST_RESPONSES` to point it at another file. To load the data directly:

```python
import pandas as pd
from remy.abtest.responses import load_responses

raw = pd.read_json("data/processed/abtest/responses.jsonl", lines=True)  # one row per survey
items = load_responses("data/processed/abtest/responses.jsonl")           # one row per rated recommendation
```
