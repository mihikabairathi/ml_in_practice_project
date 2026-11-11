"""FastAPI app for the blind A/B recipe survey.

Run from the repository root:  uv run uvicorn remy.abtest.app:app --port 8000
"""

from __future__ import annotations

import logging
import random
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from remy.abtest.config import load_config
from remy.abtest.data import RecipeBook, load_training_data
from remy.abtest.db import Database
from remy.abtest.models import MODELS

HERE = Path(__file__).parent
COOKIE = "remy_participant"
SIDES = ("left", "right")
log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    specs = config.models
    wrappers = {key: MODELS[spec.type](**spec.params) for key, spec in specs.items()}
    supervised = any(w.supervised for w in wrappers.values())
    log.info("Loading training data before %s", config.train_cutoff)
    data = load_training_data(config.train_cutoff, supervised)
    for key, wrapper in wrappers.items():
        log.info("Fitting or loading model %s (%s)", key, specs[key].type)
        wrapper.fit_or_load(data, config.model_cache_dir)
    app.state.config = config
    app.state.models = {config.model_label(k): w for k, w in wrappers.items()}
    app.state.book = RecipeBook(
        data, config.survey.quiz_pool_size, skip_top=config.survey.n_recs
    )
    app.state.db = Database(config.db_path)
    log.info("Survey ready: %s vs %s", *app.state.models)
    yield


app = FastAPI(title="Remy recipe survey", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

STAGE_PAGES = {"quiz": "/quiz", "compare": "/compare", "done": "/done"}


def _participant(request: Request, stage: str):
    """Return the participant if they are at `stage`, else a redirect to where they belong."""
    participant_id = request.cookies.get(COOKIE)
    participant = participant_id and request.app.state.db.participant(participant_id)
    if not participant:
        return None, RedirectResponse("/", status_code=303)
    if participant["stage"] != stage:
        return None, RedirectResponse(STAGE_PAGES[participant["stage"]], status_code=303)
    return participant, None


def _rating(value) -> int:
    try:
        rating = int(value)
    except (TypeError, ValueError):
        rating = 0
    if not 1 <= rating <= 5:
        raise HTTPException(400, "Please rate every recipe from 1 to 5 stars.")
    return rating


@app.get("/")
def index(request: Request, pid: str | None = None):
    participant_id = request.cookies.get(COOKIE)
    participant = participant_id and request.app.state.db.participant(participant_id)
    return templates.TemplateResponse(request, "index.html", {
        "external_id": pid,
        "resume": STAGE_PAGES[participant["stage"]] if participant else None,
        "survey": request.app.state.config.survey,
    })


@app.post("/start")
async def start(request: Request):
    form = await request.form()
    state = request.app.state
    labels = list(state.models)
    random.SystemRandom().shuffle(labels)  # blind, random left/right placement
    participant_id = str(uuid.uuid4())
    quiz_ids = state.book.sample_quiz(state.config.survey.n_quiz, seed=secrets.randbits(64))
    state.db.create_participant(
        participant_id, (form.get("external_id") or None), labels[0], labels[1],
        quiz_ids, state.config.model_dump_json(),
    )
    response = RedirectResponse("/quiz", status_code=303)
    response.set_cookie(COOKIE, participant_id, max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax")
    return response


@app.get("/quiz")
def quiz(request: Request):
    participant, redirect = _participant(request, "quiz")
    if redirect:
        return redirect
    book = request.app.state.book
    cards = [book.card(r) for r in request.app.state.db.quiz_ids(participant["participant_id"])]
    return templates.TemplateResponse(request, "quiz.html", {
        "cards": cards, "survey": request.app.state.config.survey,
    })


@app.post("/quiz")
async def submit_quiz(request: Request):
    participant, redirect = _participant(request, "quiz")
    if redirect:
        return redirect
    state = request.app.state
    form = await request.form()
    pid = participant["participant_id"]
    ratings = {r: _rating(form.get(f"r_{r}")) for r in state.db.quiz_ids(pid)}

    # The participant becomes a new user in each model's own interactions schema.
    cutoff = pd.Timestamp(state.config.train_cutoff)
    user_ratings = pd.DataFrame({
        "user_id": -participant["id"],  # negative: never collides with Food.com user ids
        "recipe_id": list(ratings),
        "date": cutoff - pd.Timedelta(days=1),
        "rating": list(ratings.values()),
    })
    shown = []
    for side, label in zip(SIDES, (participant["model_left"], participant["model_right"])):
        recs = state.models[label].recommend(user_ratings, k=state.config.survey.n_recs)
        shown += [(label, side, rank, recipe_id) for rank, recipe_id in enumerate(recs, start=1)]
    state.db.save_quiz(pid, ratings, shown)
    return RedirectResponse("/compare", status_code=303)


def _lists(request: Request, participant) -> dict[str, list[dict]]:
    book = request.app.state.book
    lists = {side: [] for side in SIDES}
    for item in request.app.state.db.shown(participant["participant_id"]):
        lists[item["side"]].append({"rank": item["rank"], **book.card(item["recipe_id"])})
    return lists


@app.get("/compare")
def compare(request: Request):
    participant, redirect = _participant(request, "compare")
    if redirect:
        return redirect
    return templates.TemplateResponse(request, "compare.html", {
        "lists": _lists(request, participant), "survey": request.app.state.config.survey,
    })


@app.post("/compare")
async def submit_compare(request: Request):
    participant, redirect = _participant(request, "compare")
    if redirect:
        return redirect
    state = request.app.state
    form = await request.form()
    models = {"left": participant["model_left"], "right": participant["model_right"]}
    item_ratings = {
        (item["side"], item["rank"]): _rating(form.get(f"item_{item['side']}_{item['rank']}"))
        for item in state.db.shown(participant["participant_id"])
    }
    list_ratings = {side: (models[side], _rating(form.get(f"list_{side}"))) for side in SIDES}
    preferred = None
    if state.config.survey.ask_preference:
        choice = form.get("preference")
        if choice not in (*SIDES, "none"):
            raise HTTPException(400, "Please choose which list you prefer.")
        preferred = models.get(choice)
    state.db.save_compare(participant["participant_id"], item_ratings, list_ratings,
                          preferred, state.config.survey.ask_preference)
    return RedirectResponse("/done", status_code=303)


@app.get("/done")
def done(request: Request):
    participant, redirect = _participant(request, "done")
    if redirect:
        return redirect
    return templates.TemplateResponse(request, "done.html", {})


@app.get("/healthz")
def healthz():
    return {"ok": True}
