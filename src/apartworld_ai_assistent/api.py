"""HTTP API for the booking assistant.

Run (Swagger UI at http://127.0.0.1:8000/docs):
    python -m uv run uvicorn apartworld_ai_assistent.api:create_app_from_env --factory --reload
"""

import os
from collections.abc import Callable
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from apartworld_ai_assistent.calendar_sync import CalendarEvent, CalendarSync, EventResult
from apartworld_ai_assistent.inventory import Inventory, load_inventory
from apartworld_ai_assistent.llm import LlmClient, OpenAiCompatibleClient
from apartworld_ai_assistent.pipeline import (
    ProcessedRequest,
    RequestStatus,
    Resolution,
    ResolutionAction,
    handle_email,
)
from apartworld_ai_assistent.stores import (
    EventStore,
    InMemoryEventStore,
    InMemoryRequestStore,
    NotInQueueError,
    RequestStore,
)


class EmailIn(BaseModel):
    email: str = Field(min_length=1, max_length=10_000, examples=[
        "Hi, we are 4 colleagues, 10-13 March 2027, near Smíchov. Thanks, Anna",
    ])


class ResolveIn(BaseModel):
    action: ResolutionAction
    resolved_by: str = Field(min_length=1, examples=["staff-name"])
    reply_text: str | None = Field(default=None, max_length=10_000, examples=[
        "Text you sent to the client (optional if the assistant prepared a draft)",
    ])
    note: str | None = Field(default=None, max_length=2_000, examples=["internal note, optional"])


class ErrorOut(BaseModel):
    detail: str


NOT_FOUND = {404: {"model": ErrorOut, "description": "No request with this id"}}
CONFLICT = {409: {
    "model": ErrorOut,
    "description": "Already resolved, or the request is not in the review queue",
}}


def create_app(
    llm: LlmClient,
    inventory: Inventory,
    now: Callable[[], datetime] = datetime.now,
    requests: RequestStore | None = None,
    events: EventStore | None = None,
    storage: str = "memory",
) -> FastAPI:
    app = FastAPI(
        title="Apart World booking assistant",
        description="Scenario 2: incoming email -> verified offer or a task for a person.",
        version="0.1.0",
    )
    store = requests or InMemoryRequestStore()
    calendar = CalendarSync(inventory, now, events or InMemoryEventStore())
    app.state.store = store

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Opening the bare address shows the API documentation."""
        return RedirectResponse("/docs")

    @app.get("/health", tags=["service"])
    def health() -> dict:
        return {"status": "ok", "units": len(inventory.units), "storage": storage}

    @app.post("/requests", status_code=201, tags=["requests"])
    def create_request(body: EmailIn) -> ProcessedRequest:
        """Process an incoming email: parse it, check units, draft the reply."""
        moment = now()
        item = handle_email(body.email, llm, inventory, today=moment.date(), now=moment)
        store.add(item)
        return item

    @app.get("/requests", tags=["requests"])
    def list_requests(status: RequestStatus | None = None) -> list[ProcessedRequest]:
        return store.list(status)

    @app.get("/requests/{request_id}", tags=["requests"], responses=NOT_FOUND)
    def get_request(request_id: str) -> ProcessedRequest:
        item = store.get(request_id)
        if item is None:
            raise HTTPException(status_code=404, detail="request not found")
        return item

    @app.get("/review-queue", tags=["staff"])
    def review_queue() -> list[ProcessedRequest]:
        """Emails a person still has to handle, oldest first. Resolved ones drop out."""
        return [i for i in store.list(RequestStatus.MANUAL_REVIEW) if i.resolution is None]

    @app.post("/review-queue/{request_id}/resolve", tags=["staff"], responses={**NOT_FOUND, **CONFLICT})
    def resolve_review_item(request_id: str, body: ResolveIn) -> ProcessedRequest:
        """Staff closes a queue item: answered the client, or dismissed it."""
        item = store.get(request_id)
        if item is None:
            raise HTTPException(status_code=404, detail="request not found")
        reply_text = body.reply_text or item.reply_text
        if body.action == ResolutionAction.REPLY_SENT and not reply_text:
            raise HTTPException(status_code=422, detail="reply_text is required: there is no draft to send")
        resolution = Resolution(
            action=body.action, resolved_by=body.resolved_by, resolved_at=now(),
            reply_text=reply_text if body.action == ResolutionAction.REPLY_SENT else None,
            note=body.note,
        )
        try:
            return store.resolve(request_id, resolution)
        except KeyError as e:
            raise HTTPException(status_code=404, detail="request not found") from e
        except NotInQueueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.post("/webhooks/calendar", tags=["webhooks"])
    def calendar_webhook(event: CalendarEvent) -> EventResult:
        """Booking created / cancelled on a channel. Safe to deliver twice."""
        return calendar.apply(event)

    @app.get("/calendar-events", tags=["staff"])
    def calendar_events(alerts_only: bool = False) -> list[EventResult]:
        """Webhook log; alerts_only=true shows conflicts and unknown listings."""
        return calendar.log(only_alerts=alerts_only)

    return app


def create_app_from_env() -> FastAPI:
    """Real LLM and PostgreSQL from .env.

    DATABASE_URL set -> PostgreSQL (run `alembic upgrade head` first);
    not set -> in-memory storage, data is lost on restart.
    APP_FIXED_NOW pins the clock for the test data.
    """
    from dotenv import load_dotenv

    load_dotenv()
    fixed = os.environ.get("APP_FIXED_NOW")
    now = (lambda: datetime.fromisoformat(fixed)) if fixed else datetime.now
    url = os.environ.get("DATABASE_URL")
    if url:
        from apartworld_ai_assistent.db import SqlEventStore, SqlRequestStore, make_engine

        engine = make_engine(url)
        return create_app(
            OpenAiCompatibleClient.from_env(), load_inventory(), now,
            requests=SqlRequestStore(engine), events=SqlEventStore(engine), storage="postgresql",
        )
    return create_app(OpenAiCompatibleClient.from_env(), load_inventory(), now)
