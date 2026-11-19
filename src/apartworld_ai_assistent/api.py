"""HTTP API for the booking assistant.

Run (Swagger UI at http://127.0.0.1:8000/docs):
    python -m uv run uvicorn apartworld_ai_assistent.api:create_app_from_env --factory --reload
"""

import os
import threading
from collections.abc import Callable
from datetime import datetime

from fastapi import FastAPI, HTTPException
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


class RequestStore:
    """In-memory store. Enough for the demo; a real system uses the database."""

    def __init__(self):
        self._items: dict[str, ProcessedRequest] = {}
        self._lock = threading.Lock()

    def add(self, item: ProcessedRequest) -> None:
        with self._lock:
            self._items[item.id] = item

    def get(self, item_id: str) -> ProcessedRequest | None:
        return self._items.get(item_id)

    def resolve(self, item_id: str, resolution: Resolution) -> ProcessedRequest:
        """Close a review item once. Raises KeyError / ValueError for the API to map."""
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                raise KeyError(item_id)
            if item.status != RequestStatus.MANUAL_REVIEW:
                raise ValueError("request is not in the review queue")
            if item.resolution is not None:
                raise ValueError("already resolved")
            updated = item.model_copy(update={"resolution": resolution})
            self._items[item_id] = updated
            return updated

    def list(self, status: RequestStatus | None = None) -> list[ProcessedRequest]:
        with self._lock:
            items = list(self._items.values())
        if status:
            items = [i for i in items if i.status == status]
        return sorted(items, key=lambda i: i.received_at)


def create_app(
    llm: LlmClient,
    inventory: Inventory,
    now: Callable[[], datetime] = datetime.now,
) -> FastAPI:
    app = FastAPI(
        title="Apart World booking assistant",
        description="Scenario 2: incoming email -> verified offer or a task for a person.",
        version="0.1.0",
    )
    store = RequestStore()
    calendar = CalendarSync(inventory, now)
    app.state.store = store

    @app.get("/health", tags=["service"])
    def health() -> dict:
        return {"status": "ok", "units": len(inventory.units)}

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
        except ValueError as e:
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
    """Real LLM from .env. APP_FIXED_NOW pins the clock for the test data."""
    from dotenv import load_dotenv

    load_dotenv()
    fixed = os.environ.get("APP_FIXED_NOW")
    now = (lambda: datetime.fromisoformat(fixed)) if fixed else datetime.now
    return create_app(OpenAiCompatibleClient.from_env(), load_inventory(), now)
