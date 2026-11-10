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

For this pilot, the runner evaluates the first quarter of each validation year
(2007–2010) and test year (2011–2015). It uses every completed training quarter
at each cutoff, including earlier test quarters once their labels are historical.
Mean validation NDCG picks one setting per challenger. The test comparison
uses equal weight per quarter and the fixed LightGBM reference.

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
split script. The pilot used LightGBM 4.7.0, scikit-learn 1.9.1, and XGBoost 3.4.1.

```sh
uv sync --extra trees
OMP_NUM_THREADS=4 uv run --extra trees python scripts/run_tree_comparison.py \
  --stride 4 --out artifacts/tree_comparison
```

Use `--stride 1` to evaluate every validation and test quarter.

## Validation results

| Model | Mean NDCG | HR@100 | Recall@100 |
| --- | ---: | ---: | ---: |
| LightGBM | 0.13102 | 0.20233 | 0.08613 |
| XGBoost, depth 3 | 0.12948 | 0.19466 | 0.08168 |
| XGBoost, depth 6 | 0.13097 | 0.20236 | 0.08621 |
| Random Forest, depth 8 | 0.12603 | 0.15451 | 0.06124 |
| Random Forest, depth 14 | 0.12865 | 0.17603 | 0.07177 |

The depth-6 XGBoost and depth-14 Random Forest advance to test.

## Test results

| Model | Mean NDCG | NDCG@Candidates | HR@10 | HR@100 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.11551 | 0.04689 | 0.03304 | 0.14954 | 0.01978 | 0.09050 |
| XGBoost, depth 6 | 0.11568 | 0.04707 | 0.03342 | 0.15217 | 0.01941 | 0.09259 |
| Random Forest, depth 14 | 0.11390 | 0.04528 | 0.02675 | 0.14216 | 0.01636 | 0.08824 |

XGBoost gained 0.00018 mean NDCG over LightGBM and won three of five
quarters. It gained 0.26 percentage points in HR@100, while LightGBM
retained a slight edge in Recall@10. The five Q1 cutoffs give only a narrow
view of performance across seasons. Random Forest lost 0.00160 mean NDCG
and beat LightGBM on one quarter.

The LightGBM test metrics match the prior run's corresponding five quarters
to floating-point precision. See [quarterly results](quarterly.csv) and
[split averages](summary.csv).
