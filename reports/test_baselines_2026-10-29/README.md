# Rolling test evaluation: Popularity, original LightGBM, selected LightGBM

Evaluated all 20 test quarters, January 2011 through December 2015. Settings were fixed before test evaluation; no test-set tuning was performed. Results below are the unweighted mean of the per-quarter means across scorable users, matching the existing runner and notebooks. Each model scored 29,079 user-quarter pairs (not distinct users).

| Model | Recall@10 | HR@10 | Recall@100 | HR@100 | NDCG | NDCG@Candidates |
|---|---|---|---|---|---|---|
| Popularity | 0.017190 | 0.026649 | 0.077067 | 0.119976 | 0.111315 | 0.042029 |
| Original LightGBM | 0.012635 | 0.019648 | 0.057271 | 0.092803 | 0.107822 | 0.038536 |
| LightGBM ALL_ROUND1 | 0.018727 | 0.029459 | 0.086321 | 0.134743 | 0.113430 | 0.044143 |

ALL_ROUND1 improves Recall@10 by 8.94% relative to Popularity (+0.001537 absolute), winning in 13/20 quarters. This does not clear the document's 10% relative Recall@10 deployment threshold. These offline results do not establish online user satisfaction.

## Selected model identity

Used Emma's `src/remy/model/lightgbm_features.py` from commit `df1a029`, with `feature_groups=ALL_ROUND1`. This reproduces the document's round-one **all** feature combination (validation NDCG 0.1267, Recall@10 0.0167), rather than the subsequently added round-two popularity combinations. The saved part-one notebook output lists the original nine feature groups, before the round-two additions.

Feature groups: `user_base`, `tags`, `user_activity`, `user_taste`, `recipe_pop`, `recipe_rating`, `recipe_static`, `inter_tag`, `inter_diff`. Excluded `pop_season`, `pop_trend`, and `pop_5star` to match the selected experiment in the supplied document.

Used the model defaults: binary objective, 10 random negatives per positive, no hard negatives, 200 boosting rounds, learning rate 0.05, 31 leaves, minimum 100 rows per leaf, 1,000 candidates, seed 10718. The other constructor settings are recorded in `selected_config.json`. Four threads were used for each fit. The original baseline uses `src/remy/model/lightgbm.py` from commit `17f32d0` (unchanged by Emma's update).

## Protocol and execution

Used the existing `RollingSplits`, model `fit`/`recommend_batch`, and `evaluate_quarter` implementations. Each fresh model trains on all completed earlier quarterly tasks; each training task builds features from its own preceding history. Earlier validation/test outcomes enter training only after their outcome window finishes. Evaluation uses eligible users with at least three prior reviews and a new 5-star outcome on an available, unseen recipe. Users with no positive outcome are not scored. Current-quarter outcomes are used only for scoring. Rankings beyond the 1,000-candidate shortlist use the common unseen-popularity fallback. NDCG is full-ranking; NDCG@Candidates uses a cutoff of 1,000 for every model.

The existing baseline runner used these arguments, executed through the feature-name wrapper described below:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/run_experiments.py \
  --models popularity lightgbm --split test \
  --run-name test_popularity_lightgbm_20261029
```

At 2011 Q4, LightGBM rejected a colon in the tag `less_thansql:name_topics_of_recipegreater_than`. The runs therefore use a local wrapper around `lgb.Dataset` that replaces display names with `feature_0`, `feature_1`, etc. Feature values, column order, sampling, training, and predictions are unchanged. A small deterministic check confirmed identical predictions before/after renaming, and the restarted baseline reproduced the completed earlier quarter scores. No collaborator model source was edited.

The selected run follows the part-one notebook's `run` loop, with `SPLIT="test"`, all 20 quarters, `feature_groups=ALL_ROUND1`, `num_threads=4`, and `cutoff=q.start`, using the same feature-name workaround. Independent test quarters were distributed between local workers, with the same settings and seed per quarter. Their saved outputs were combined after verifying exactly one result for each quarter. Local one-off execution wrappers remain in ignored `scratchpad/`, following AGENTS.md.

## Where the selected model differs from Popularity

Mean absolute differences by year (selected minus Popularity):

| Year | Recall@10 difference | NDCG difference |
|---|---:|---:|
| 2011 | +0.001333 | +0.001349 |
| 2012 | -0.002062 | +0.000623 |
| 2013 | +0.001064 | +0.002192 |
| 2014 | -0.000763 | +0.001877 |
| 2015 | +0.008114 | +0.004531 |

See `selected_minus_popularity.csv` for all metrics in each quarter. Differences describe this test set; no settings were changed based on them.

## Verification and artifacts

Verified exactly 20 rows per model, identical quarter dates, eligible/scored user counts and positive counts across models, no duplicate model-quarter rows, and finite metrics between zero and one. All 60 model-quarter evaluations completed.

- `quarterly_results.csv`: full precision per-quarter metrics for all three models.
- `summary.csv`: full precision macro averages.
- `baseline_config.json`, `selected_config.json`: run settings.
- `baseline_provenance.json`: package versions and code/data hashes; the same environment and split files were used for the selected run.

The PR adds evaluation artifacts only; no existing source or notebook files were changed.
