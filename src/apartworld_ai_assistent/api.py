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

from apartworld_ai_assistent.inventory import Inventory, load_inventory
from apartworld_ai_assistent.llm import LlmClient, OpenAiCompatibleClient
from apartworld_ai_assistent.pipeline import ProcessedRequest, RequestStatus, handle_email


class EmailIn(BaseModel):
    email: str = Field(min_length=1, max_length=10_000, examples=[
        "Hi, we are 4 colleagues, 10-13 March 2027, near Smíchov. Thanks, Anna",
    ])


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

    @app.get("/requests/{request_id}", tags=["requests"])
    def get_request(request_id: str) -> ProcessedRequest:
        item = store.get(request_id)
        if item is None:
            raise HTTPException(status_code=404, detail="request not found")
        return item

    @app.get("/review-queue", tags=["staff"])
    def review_queue() -> list[ProcessedRequest]:
        """Emails a person has to handle, oldest first."""
        return store.list(RequestStatus.MANUAL_REVIEW)

    return app


def create_app_from_env() -> FastAPI:
    """Real LLM from .env. APP_FIXED_NOW pins the clock for the test data."""
    from dotenv import load_dotenv

    load_dotenv()
    fixed = os.environ.get("APP_FIXED_NOW")
    now = (lambda: datetime.fromisoformat(fixed)) if fixed else datetime.now
    return create_app(OpenAiCompatibleClient.from_env(), load_inventory(), now)
