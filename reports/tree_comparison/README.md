# Recipe reranker comparison

## Emma’s LightGBM experiments

Emma kept the binary reranking task and varied the features in two validation
notebooks ([round one](../../notebooks/lightgbm_feature_experiments_part1.ipynb),
[round two](../../notebooks/lightgbm_feature_experiments_part2.ipynb)). The
original model used each user’s review count, mean rating, and five-star rate,
along with binary recipe tags. Round one added these feature groups.

| Group | Signals |
| --- | --- |
| User activity | Rating variability, days since the last review, tenure, and 90- and 365-day review counts |
| User taste | Count of distinct recipes rated at least four stars and their average cooking time, ingredient count, calories, and popularity |
| Recipe popularity | Lifetime, 90-day, and 365-day counts, popularity rank, recent share, and time since first and latest review |
| Recipe ratings | Mean rating and five-star rate smoothed toward the global averages |
| Recipe metadata | Cooking time, steps, ingredient count, calories, and number of tags |
| User–recipe match | Tag cosine similarity, tag affinity, and differences between the recipe and the user’s taste profile |

On Emma’s 16 validation quarters, baseline NDCG was 0.1214. Adding recipe
popularity alone reached 0.1259. All round-one groups reached 0.1267 and beat
the baseline in all 16 quarters. The taste-relative differences also performed
well alone at 0.1261, in part because one of the differences uses popularity.
Recipe rating averages alone scored 0.1188.

Round two tried same-calendar-quarter seasonality, recent versus preceding
90-day trend, review counts with a 90-day half-life, and lifetime and recent
five-star counts. Adding all of these to the popularity baseline reached
0.1267. Adding taste and tag match to that setup reached 0.1274 in the saved
validation results. This comparison uses the round-one feature set from the
team’s prior held-out LightGBM run, so the family comparison has a fixed
reference. Round-two features are outside this comparison.

## Protocol

Every family sees the same training pairs, feature columns, and top-1,000
popularity shortlist. Features are computed at the historical cutoff for each
training quarter. Five-star outcomes are positives, with up to ten sampled
available, unseen negatives per positive. The evaluator excludes previously
seen recipes and completes each ranking with the popularity fallback.

The comparison evaluates all 16 rolling validation quarters from 2007 through
2010. It uses every completed training quarter at each cutoff. Mean
validation NDCG selects one setting per challenger, with equal weight per
quarter. The reported comparison uses validation scores.

- LightGBM uses 200 rounds, learning rate 0.05, and 31 leaves.
- XGBoost compares depths 3 and 6, with 120 rounds and learning rate 0.08.
- Random Forest compares depth 8 with minimum leaf size 40 against depth 14
  with minimum leaf size 20. Both use 32 trees, the square root of the feature
  count per split, and a 20% bootstrap sample per tree to keep training time
  practical. Each tree draws its sample from the shared training matrix.
- XGBoost stores numeric zeros explicitly in the sparse input. Absent tag
  entries represent tag absence. Random Forest encodes missing numeric values
  with a sentinel below the observed feature range.

Run with Python 3.12 after preparing the rolling data with the repository’s
split script. The run used LightGBM 4.7.0, scikit-learn 1.9.1, and XGBoost 3.4.1.

```sh
uv sync --extra trees
OMP_NUM_THREADS=4 uv run --extra trees python scripts/run_tree_comparison.py \
  --validation-only --out artifacts/tree_comparison
```

## Validation results

| Model | Mean NDCG | HR@100 | Recall@100 |
| --- | ---: | ---: | ---: |
| LightGBM | 0.12674 | 0.17979 | 0.07908 |
| XGBoost, depth 3 | 0.12557 | 0.17376 | 0.07623 |
| XGBoost, depth 6 | 0.12664 | 0.18048 | 0.07935 |
| Random Forest, depth 8 | 0.12258 | 0.14357 | 0.06169 |
| Random Forest, depth 14 | 0.12498 | 0.16166 | 0.06990 |

Depth-6 XGBoost finished 0.00010 behind LightGBM in mean NDCG, with
slightly higher HR@100 and Recall@100. Depth-14 Random Forest finished
0.00176 behind LightGBM in mean NDCG. See the
[quarterly results](quarterly.csv) and [metric averages](summary.csv).
