"""Restart the service on the same PostgreSQL: nothing is lost, nothing is applied twice."""

from datetime import datetime

from fastapi.testclient import TestClient

from apartworld_ai_assistent.api import create_app
from apartworld_ai_assistent.db import SqlEventStore, SqlRequestStore
from apartworld_ai_assistent.intake import ExtractedRequest
from apartworld_ai_assistent.inventory import load_inventory
from apartworld_ai_assistent.llm import LlmError
from apartworld_ai_assistent.reply import ReplyText

NOW = datetime(2026, 11, 19, 12, 0)
SMICHOV = {"check_in": "2027-03-10", "check_out": "2027-03-13", "guests": 3,
           "language": "en", "preferences": ["Smíchov"]}
REPLY = {"greeting": "Hello,", "intro": "Here are our options.", "closing": "See you soon."}
WEBHOOK = {"event_id": "evt-ota-1", "type": "booking_created", "listing_id": "booking-river-deluxe-2",
           "check_in": "2027-03-10", "check_out": "2027-03-13"}


class SchemaFakeLlm:
    def __init__(self, answers):
        self.answers = answers

    def extract(self, system_prompt, user_text, schema):
        answer = self.answers[schema]
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate(answer)


def start(engine, answers=None) -> TestClient:
    """A fresh service process: new app, units.json reloaded, same database."""
    llm = SchemaFakeLlm(answers or {ExtractedRequest: SMICHOV, ReplyText: REPLY})
    app = create_app(llm, load_inventory(), now=lambda: NOW,
                     requests=SqlRequestStore(engine), events=SqlEventStore(engine), storage="postgresql")
    return TestClient(app)


def offered(client) -> list[str]:
    return [o["unit_id"] for o in client.post("/requests", json={"email": "x"}).json()["offers"]]


def test_requests_and_staff_decisions_survive_restart(pg_engine):
    first = start(pg_engine, {ExtractedRequest: LlmError("503"), ReplyText: REPLY})
    queued = first.post("/requests", json={"email": "anything"}).json()
    first.post(f"/review-queue/{queued['id']}/resolve",
               json={"action": "dismissed", "resolved_by": "staff", "note": "spam"})

    second = start(pg_engine)
    assert second.get("/health").json()["storage"] == "postgresql"
    assert second.get(f"/requests/{queued['id']}").json()["resolution"]["note"] == "spam"
    assert second.get("/review-queue").json() == []


def test_webhook_bookings_survive_restart(pg_engine):
    first = start(pg_engine)
    assert "river-deluxe-suite" in offered(first)
    assert first.post("/webhooks/calendar", json=WEBHOOK).json()["status"] == "applied"

    second = start(pg_engine)  # units.json reloaded, bookings replayed from the journal
    assert "river-deluxe-suite" not in offered(second)


def test_repeated_webhook_after_restart_is_still_a_duplicate(pg_engine):
    start(pg_engine).post("/webhooks/calendar", json=WEBHOOK)
    second = start(pg_engine)
    assert second.post("/webhooks/calendar", json=WEBHOOK).json()["status"] == "duplicate"
    log = second.get("/calendar-events").json()
    assert len(log) == 1 and log[0]["duplicates"] == 1


def test_conflict_alert_survives_restart(pg_engine):
    first = start(pg_engine)
    first.post("/webhooks/calendar", json=WEBHOOK)
    first.post("/webhooks/calendar",
               json={**WEBHOOK, "event_id": "evt-direct-1", "listing_id": "direct-river-deluxe-suite"})
    alerts = start(pg_engine).get("/calendar-events", params={"alerts_only": True}).json()
    assert [(a["event_id"], a["status"]) for a in alerts] == [("evt-direct-1", "conflict")]


def test_cancellation_after_restart_frees_the_unit(pg_engine):
    start(pg_engine).post("/webhooks/calendar", json=WEBHOOK)
    second = start(pg_engine)
    cancel = {**WEBHOOK, "event_id": "evt-ota-1-cancel", "type": "booking_cancelled"}
    assert second.post("/webhooks/calendar", json=cancel).json()["status"] == "applied"
    assert "river-deluxe-suite" in offered(start(pg_engine))
