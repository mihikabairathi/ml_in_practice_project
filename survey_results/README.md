# Survey results

`responses_20261118-034121.jsonl` contains one completed blind A/B survey comparing XGBoost
and popularity, completed on November 17, 2026 at 10:44 p.m. Eastern
(November 18 at 03:44 UTC). It is an unchanged copy of the local response log
from `data/processed/abtest/responses/`.

The JSONL record includes the five quiz ratings, both recommendation lists and
all recipe ratings, overall list ratings, model configuration, and timestamps.

| Model | Overall list rating (1–5) | Mean recipe rating (1–5) | Recipes rated |
| --- | ---: | ---: | ---: |
| popularity | 3 | 3.9 | 10 |
| xgboost | 4 | 4.1 | 10 |

No explicit preference was collected (`ask_preference: false`). These are
results from one participant and do not establish a population-level difference.

To load the saved results using the existing helpers:

```python
from remy.abtest.responses import load_responses, read_surveys

surveys = read_surveys("survey_results")
items = load_responses("survey_results")
```
