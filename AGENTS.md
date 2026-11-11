# Instructions for Agents

## Working Nicely on a Group Project

This is a group project and, while this code is research and not a production product, we want to maintain standards and not trample on others' work. Please adhere to the following principles:
1. Always work on a branch and submit your work via PRs with clear descriptions of what was changed and why.
2. Before implementing anything new, make sure that it is not already implemented in some form. Whenever possible, it is better to use or adapt existing code rather than coming up with bespoke solutions.
3. When modifying existing code, check whether the code was written by this user or by another collaborator. If it was written by someone else, only make necessary targeted changes needed for your use-case and mention it clearly in your PR.

## Folder Rules

- `scratchpad` (untracked): any one-offs, random tests, local code; use liberally
- `notebooks`: tracked notebooks can be exploratory but should have a clear narrative flow and demonstrate some process/result
- `scripts`: only store scripts here that will be used for important, repeatable tasks
- `src`: official product code; add here only when requested by the user
## Models and the A/B Survey

Models in `src/remy/model/` share one interface:
- `fit(interactions, recipes=None, training_quarters=())`
- `recommend(user_id, k)`
- `recommend_batch(user_ids, k)`
- the `ranking_` and `seen_` attributes

See "Add a model" in the README. The blind A/B survey (`src/remy/abtest/`) uses these models **unchanged**:
- It fits each model once.
- For each participant, a `SurveyModel` wrapper in `src/remy/abtest/models.py` adds their quiz ratings to the fitted model's per-user state, calls the model's own `recommend`, then removes them.

Rules:
1. **Adding a model.** If it should be usable in the survey, add a `SurveyModel` subclass in `src/remy/abtest/models.py` and register it in `MODELS`. Set:
   - `model_class`
   - `supervised` (True if `fit` needs `training_quarters`)
   - `add_user`/`remove_user` for every per-user structure that `recommend` reads beyond `seen_`

   Build the new user's state with the model's own helpers where possible. Never modify learned item-side state.
2. **Changing a model's per-user state.** If you change what `fit` builds per user or what `recommend` looks up per user, update its survey wrapper in the same PR.
3. **Verify.** After either change, run `python scripts/check_survey_models.py --models <name>`. It must report that copied users match and that the model state was restored. Mention the result in the PR.
4. **Survey settings.** Model choice and params live in `src/remy/abtest/config.yaml`. Don't hard-code model settings in the app.
5. **Keep participants blind.** Never put model names, scores or other model-identifying data in templates or HTML responses.
