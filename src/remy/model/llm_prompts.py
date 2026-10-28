"""Prompt text for the zero-shot LLM recommender (remy.model.llm)."""

SYSTEM_PROMPT = (
    "You are a recipe recommender for the website Food.com. "
    "Given a list of candidate recipes and a user's past liked recipes, "
    "pick the recipes the user is most likely to cook and rate 5 stars next."
)

USER_PROMPT = """\
CANDIDATE RECIPES (format: id: name [tags]):
{candidates}

USER'S MOST RECENT LIKED RECIPES (oldest first):
{history}

Choose the {k} candidate recipes this user is most likely to try and love next, \
best first. Only use ids from the candidate list, with no duplicates. \
Respond with only JSON of the form {{"recipe_ids": [id1, id2, ...]}}."""

# Food.com tags that nearly every recipe has; they waste tokens without
# describing the dish, so they are skipped when showing a recipe's tags.
GENERIC_TAGS = {
    "time-to-make", "course", "main-ingredient", "preparation", "occasion",
    "cuisine", "dietary", "equipment", "number-of-servings", "3-steps-or-less",
    "4-hours-or-less", "60-minutes-or-less", "30-minutes-or-less",
    "15-minutes-or-less", "free-of-something", "low-in-something",
    "easy", "north-american", "american",
}


def candidate_line(recipe_id, name, tags, n_tags=3):
    """One candidate, e.g. '25885: banana banana bread [weeknight, breads, fruit]'."""
    shown = [tag for tag in tags if tag not in GENERIC_TAGS][:n_tags]
    line = f"{recipe_id}: {name}"
    return f"{line} [{', '.join(shown)}]" if shown else line


def history_line(name, rating):
    """One past review, e.g. '- onion bread pudding (5 stars)'."""
    return f"- {name} ({int(rating)} stars)"


def build_messages(candidate_lines, history_lines, k):
    """Chat messages for one user."""
    user = USER_PROMPT.format(
        candidates="\n".join(candidate_lines),
        history="\n".join(history_lines),
        k=k,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
