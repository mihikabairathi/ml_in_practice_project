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
from remy.abtest.responses import ResponseLog, now
from remy.abtest.models import MODELS

HERE = Path(__file__).parent
COOKIE = "remy_participant"
SIDES = ("left", "right")
# Participants are added to a model only inside SurveyModel.recommend, under that
# model's lock, and removed straight after, so one temporary id is enough. It is
# negative so it can never collide with a Food.com user id.
MODEL_USER_ID = -1
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
    app.state.log = ResponseLog(config.responses_path)
    log.info("Survey ready: %s vs %s", *app.state.models)
    yield


app = FastAPI(title="Remy recipe survey", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

STAGE_PAGES = {"quiz": "/quiz", "compare": "/compare", "done": "/done"}


def _session(request: Request) -> dict | None:
    return request.app.state.log.sessions.get(request.cookies.get(COOKIE, ""))


def _at_stage(request: Request, stage: str):
    """Return the session if it is at `stage`, else a redirect to where it belongs."""
    session = _session(request)
    if not session:
        return None, RedirectResponse("/", status_code=303)
    if session["stage"] != stage:
        return None, RedirectResponse(STAGE_PAGES[session["stage"]], status_code=303)
    return session, None


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
    session = _session(request)
    resume = STAGE_PAGES[session["stage"]] if session and session["stage"] != "done" else None
    return templates.TemplateResponse(request, "index.html", {
        "external_id": pid, "resume": resume, "survey": request.app.state.config.survey,
    })


@app.post("/start")
async def start(request: Request):
    """Every run is a new user: fresh id, fresh quiz, fresh random left/right."""
    form = await request.form()
    state = request.app.state
    labels = list(state.models)
    random.SystemRandom().shuffle(labels)  # blind, random left/right placement
    user_id = str(uuid.uuid4())
    state.log.sessions[user_id] = {
        "user_id": user_id,
        "external_id": form.get("external_id") or None,
        "started_at": now(),
        "stage": "quiz",
        "models": {"left": labels[0], "right": labels[1]},
        "quiz_ids": state.book.sample_quiz(state.config.survey.n_quiz, seed=secrets.randbits(64)),
    }
    response = RedirectResponse("/quiz", status_code=303)
    response.set_cookie(COOKIE, user_id, max_age=60 * 60 * 24, httponly=True, samesite="lax")
    return response


@app.get("/quiz")
def quiz(request: Request):
    session, redirect = _at_stage(request, "quiz")
    if redirect:
        return redirect
    cards = [request.app.state.book.card(r) for r in session["quiz_ids"]]
    return templates.TemplateResponse(request, "quiz.html", {
        "cards": cards, "survey": request.app.state.config.survey,
    })


@app.post("/quiz")
async def submit_quiz(request: Request):
    session, redirect = _at_stage(request, "quiz")
    if redirect:
        return redirect
    state = request.app.state
    form = await request.form()
    ratings = {r: _rating(form.get(f"r_{r}")) for r in session["quiz_ids"]}

    # The participant becomes a temporary new user in each model's interactions schema.
    cutoff = pd.Timestamp(state.config.train_cutoff)
    user_ratings = pd.DataFrame({
        "user_id": MODEL_USER_ID,
        "recipe_id": list(ratings),
        "date": cutoff - pd.Timedelta(days=1),
        "rating": list(ratings.values()),
    })
    session["quiz_ratings"] = ratings
    session["lists"] = {
        side: state.models[label].recommend(user_ratings, k=state.config.survey.n_recs)
        for side, label in session["models"].items()
    }
    session["stage"] = "compare"
    return RedirectResponse("/compare", status_code=303)


@app.get("/compare")
def compare(request: Request):
    session, redirect = _at_stage(request, "compare")
    if redirect:
        return redirect
    book = request.app.state.book
    lists = {
        side: [{"rank": rank, **book.card(r)} for rank, r in enumerate(ids, start=1)]
        for side, ids in session["lists"].items()
    }
    return templates.TemplateResponse(request, "compare.html", {
        "lists": lists, "survey": request.app.state.config.survey,
    })


@app.post("/compare")
async def submit_compare(request: Request):
    session, redirect = _at_stage(request, "compare")
    if redirect:
        return redirect
    state = request.app.state
    config = state.config
    form = await request.form()
    book = state.book
    lists = []
    for n, side in enumerate(SIDES, start=1):
        items = [
            {"rank": rank, "recipe_id": r, "name": book.card(r)["name"],
             "rating": _rating(form.get(f"item_{side}_{rank}"))}
            for rank, r in enumerate(session["lists"][side], start=1)
        ]
        lists.append({
            "label": f"List {n}", "side": side, "model": session["models"][side],
            "overall_rating": _rating(form.get(f"list_{side}")), "items": items,
        })
    preferred = None
    if config.survey.ask_preference:
        choice = form.get("preference")
        if choice not in (*SIDES, "none"):
            raise HTTPException(400, "Please choose which list you prefer.")
        preferred = session["models"].get(choice)

    model_keys = {config.model_label(k): k for k in config.models}
    state.log.append({
        "user_id": session["user_id"],
        "external_id": session["external_id"],
        "started_at": session["started_at"],
        "completed_at": now(),
        "train_cutoff": str(config.train_cutoff),
        "models": {
            label: {"type": config.models[key].type, "params": config.models[key].params}
            for label, key in model_keys.items()
        },
        "survey": config.survey.model_dump(),
        "quiz": [
            {"position": i, "recipe_id": r, "name": book.card(r)["name"], "rating": rating}
            for i, (r, rating) in enumerate(session["quiz_ratings"].items(), start=1)
        ],
        "lists": lists,
        "preferred_model": preferred,  # null = no preference (or not asked)
    })
    state.log.sessions[session["user_id"]] = {"stage": "done"}  # drop the bulky state
    return RedirectResponse("/done", status_code=303)


@app.get("/done")
def done(request: Request):
    session, redirect = _at_stage(request, "done")
    if redirect:
        return redirect
    return templates.TemplateResponse(request, "done.html", {})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse("/static/favicon.svg", status_code=301)


@app.get("/healthz")
def healthz():
    return {"ok": True}
