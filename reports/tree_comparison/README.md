# Rolling recipe recommendation results

All scores are means across the same 16 rolling validation quarters, with
equal weight per quarter. Each fitted model used only completed quarters at
its prediction cutoff. Recall@10 was the primary metric.

## LightGBM comparison

| | LightGBM chosen in the report | New LightGBM winner |
|---|---|---|
| Basic features | Recipe tags, user review count, mean rating, five-star rate | Same |
| Popularity | Lifetime and recent review activity, rank, and recency | Same, plus seasonality, trend and decay, and lifetime and recent five-star counts |
| Other features | User activity and taste, recipe ratings and metadata, tag match, and differences from the user’s taste | Drops those groups and adds three MiniLM similarity scores from recipe names and ingredients |
| Tree settings | 200 rounds, 31 leaves, learning rate 0.05 | 260 rounds, 15 leaves, learning rate 0.05, larger minimum leaves, 80% feature sampling, and L2 regularization |
| **Recall@10, same 16 validation quarters** | **1.668%** | **1.837%** |

## Six-model comparison

| Model | R@10 | R@100 | HR@10 | HR@100 | NDCG |
|---|---:|---:|---:|---:|---:|
| Non-ML popularity | 1.567% | 7.03% | 3.98% | 16.04% | 0.1251 |
| Trained MiniLM embedding model | 1.452% | 7.02% | 3.87% | 16.51% | 0.1249 |
| Random Forest + MiniLM | 1.648% | 8.14% | 4.33% | 18.30% | 0.1267 |
| LightGBM chosen in the report | 1.668% | 7.91% | 4.31% | 17.98% | 0.1267 |
| New LightGBM + MiniLM | 1.837% | 8.47% | 4.75% | 18.94% | 0.1280 |
| XGBoost + MiniLM | 1.861% | 8.50% | 4.78% | 19.13% | 0.1283 |

The new LightGBM search screened 15 feature and tree settings. The XGBoost
search screened 20 settings with Emma’s round-one features, then evaluated
the four strongest across all 16 quarters. After testing additional
features, a four-setting follow-up found the depth-five winner. The Random
Forest comparison included two initial settings and the final dense-feature
configuration. These are validation results used for model selection.

## Final model inputs

**All three models** use the following numeric features.

- User review count, mean rated score, and share of rated reviews that were
  five stars.
- Log lifetime review count, log 90-day and 365-day review counts, log
  popularity rank, share of lifetime reviews in the past 90 days, days
  since first review, and days since most recent review.
- Smoothed share of reviews in the same calendar quarter, log count of
  reviews in that calendar quarter, and the predicted calendar quarter
  number.
- Log ratio of reviews in the latest 90 days to the preceding 90 days,
  with one added to each count, and log exponentially decayed review
  count with a 90-day half-life.
- Log lifetime five-star count and log five-star count over the past 90 days.
- MiniLM name cosine to the user’s profile, ingredient cosine to the
  profile, and the best mean name/ingredient cosine across the three most
  recent liked recipes. A like is a rating of four or five, weighted 0.5
  and 1.0 respectively in the profile. The embedding vectors are centered
  using the available catalog at each historical cutoff.

**LightGBM and XGBoost** also use one binary flag for each recipe tag.

**XGBoost and Random Forest** also use the count of distinct liked recipes
and their mean log cooking time, ingredient count, log calories, and log
popularity. They use tag cosine similarity to the user’s liked-recipe tag
profile and tag affinity, the mean historical like frequency of the
candidate’s tags. Random Forest uses these two aggregate tag scores with
no individual tag flags.

## Final settings

| Model | Estimator settings |
|---|---|
| LightGBM | Binary objective, 260 boosting rounds, learning rate 0.05, 15 leaves, minimum 180 rows per leaf, feature fraction 0.80, L2 3.0, no row bagging or L1 penalty |
| XGBoost | Binary logistic objective, histogram trees, 120 rounds, depth 5, learning rate 0.08229484, minimum child weight 5.78537736, row subsampling 0.88533705, tree-level column subsampling 0.65261736, L2 1.09711599 |
| Random Forest | 32 trees, depth 12, minimum 40 rows per leaf, Gini splits, 50% of features considered per split, 12% bootstrap sample per tree |

All three use seed 10718, up to ten uniformly sampled unseen negatives per
five-star positive, and the original 1,000 lifetime-popularity candidates.

## Candidate coverage

The original shortlist’s oracle Recall@10 is 22.78%. It is 16.55% at 500
popularity candidates, 29.81% at 2,000, and 40.55% at 5,000. Temporal
popularity at 1,000 reached 23.65%. A small MiniLM tail reached 23.25%,
while a 50/50 popularity and MiniLM split dropped to 21.00%. KNN tail
replacement dropped it to 21.87% for 100 slots and 20.21% for 250 slots.

With the final XGBoost held fixed on four screening quarters, the original
pool scored 1.7673% actual Recall@10. The 500, temporal 1,000, and 5,000
alternatives scored 1.7643%, 1.7692%, and 1.7651%, respectively. None
advanced. The earlier round-one XGBoost + MiniLM run also found a near
tie across all 16 quarters, at 1.7418% for 1,000 and 1.7431% for 5,000.
The LightGBM pool screen likewise found essentially unchanged Recall@10
on the matched cutoffs it completed. The final LightGBM pool checks
were canceled.

On the earlier round-one XGBoost + MiniLM rank diagnostic, 236,338 of
279,537 five-star outcomes were outside the original shortlist. Of the
43,199 inside, 2,992 were ranked in the top ten and 28,402 were ranked
below 100. These are event counts. The oracle score first averages over
users, then over quarters.

## Using the two new model files

Install XGBoost 3.x in the project environment for the XGBoost model.
Both models use the project’s saved MiniLM recipe-name and ingredient
arrays in `data/processed/embeddings/minilm_joint`. The model defaults are
the selected settings.

```python
from remy.model.random_forest import RandomForest
from remy.model.xgboost import XGBoost
from remy.splits import RollingSplits

splits = RollingSplits()
quarter = splits.evaluation_quarters("val")[0]
model = XGBoost()  # Or RandomForest()
model.fit(quarter.history, quarter.recipes,
          splits.training_quarters(quarter.start), cutoff=quarter.start)
recipes = model.recommend(int(quarter.users.user_id.iloc[0]), k=10)
```

[Final averages](final_summary.csv) · [Final per-quarter metrics](final_quarterly.csv).
The [earlier comparison](summary.csv) is retained for reference.
