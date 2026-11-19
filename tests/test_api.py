from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from apartworld_ai_assistent.api import create_app
from apartworld_ai_assistent.intake import ExtractedRequest
from apartworld_ai_assistent.inventory import load_inventory
from apartworld_ai_assistent.llm import LlmError
from apartworld_ai_assistent.reply import ReplyText

NOW = datetime(2026, 11, 19, 12, 0)

EXTRACTED = {"check_in": "2027-03-10", "check_out": "2027-03-13", "guests": 4,
             "language": "en", "preferences": ["Smíchov"], "client_name": "Anna"}
REPLY = {"greeting": "Dear Anna,", "intro": "Here are our options for your stay.",
         "closing": "We look forward to hosting you."}


class SchemaFakeLlm:
    """Answers by the schema asked for: intake and reply are two different calls."""

    def __init__(self, answers: dict[type[BaseModel], dict | Exception]):
        self.answers = answers

    def extract(self, system_prompt, user_text, schema):
        answer = self.answers[schema]
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate(answer)


def client_with(answers) -> TestClient:
    app = create_app(SchemaFakeLlm(answers), load_inventory(), now=lambda: NOW)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return client_with({ExtractedRequest: EXTRACTED, ReplyText: REPLY})


def post(client, email="We are 4, 10-13 March 2027, Smíchov. Anna"):
    response = client.post("/requests", json={"email": email})
    assert response.status_code == 201, response.text
    return response.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "units": 12}


def test_email_gets_reply_with_verified_offers(client):
    body = post(client)
    assert body["status"] == "ready"
    assert body["reply_text"].startswith("Dear Anna,")
    assert [o["unit_id"] for o in body["offers"]] == ["river-deluxe-suite", "victory-castle-suite"]
    assert body["offers"][0]["url"].startswith("https://apartworldgroup.com/")
    assert body["request"]["guests"] == 4


def test_request_can_be_read_back(client):
    created = post(client)
    fetched = client.get(f"/requests/{created['id']}").json()
    assert fetched == created


def test_unknown_request_is_404(client):
    assert client.get("/requests/nope").status_code == 404


def test_missing_details_get_a_question_without_llm_reply():
    client = client_with({ExtractedRequest: {**EXTRACTED, "guests": None, "language": "ru"},
                          ReplyText: LlmError("must not be called")})
    body = post(client)
    assert body["status"] == "needs_info"
    assert body["missing"] == ["guests"]
    assert "количество гостей" in body["reply_text"]
    assert "даты" not in body["reply_text"]


def test_llm_outage_lands_in_review_queue():
    client = client_with({ExtractedRequest: LlmError("503"), ReplyText: REPLY})
    body = post(client)
    assert body["status"] == "manual_review"
    queue = client.get("/review-queue").json()
    assert [item["id"] for item in queue] == [body["id"]]


def test_review_queue_holds_only_manual_review(client):
    post(client)  # ready
    assert client.get("/review-queue").json() == []


def test_filter_requests_by_status(client):
    post(client)
    assert len(client.get("/requests", params={"status": "ready"}).json()) == 1
    assert client.get("/requests", params={"status": "no_options"}).json() == []


def test_empty_email_is_rejected(client):
    assert client.post("/requests", json={"email": ""}).status_code == 422


def test_swagger_is_available(client):
    assert client.get("/docs").status_code == 200
    assert "/review-queue" in client.get("/openapi.json").json()["paths"]


# --- staff closes review items ---

@pytest.fixture
def queued():
    """A client whose LLM is down, so every email lands in the review queue."""
    client = client_with({ExtractedRequest: LlmError("503"), ReplyText: REPLY})
    item = post(client)
    return client, item["id"]


def resolve(client, request_id, **body):
    return client.post(f"/review-queue/{request_id}/resolve", json={"resolved_by": "dmitrij", **body})


def test_staff_reply_closes_the_item(queued):
    client, rid = queued
    response = resolve(client, rid, action="reply_sent", reply_text="Dear Anna, we checked by hand...")
    assert response.status_code == 200
    resolution = response.json()["resolution"]
    assert resolution["action"] == "reply_sent"
    assert resolution["resolved_by"] == "dmitrij"
    assert resolution["reply_text"].startswith("Dear Anna")
    assert resolution["resolved_at"] == NOW.isoformat()
    assert client.get("/review-queue").json() == []
    assert client.get(f"/requests/{rid}").json()["resolution"]["action"] == "reply_sent"


def test_dismiss_needs_no_text(queued):
    client, rid = queued
    response = resolve(client, rid, action="dismissed", note="spam")
    assert response.status_code == 200
    assert response.json()["resolution"]["note"] == "spam"
    assert response.json()["resolution"]["reply_text"] is None


def test_reply_without_any_text_is_rejected(queued):
    client, rid = queued
    assert resolve(client, rid, action="reply_sent").status_code == 422
    assert len(client.get("/review-queue").json()) == 1  # still waiting


def test_item_cannot_be_resolved_twice(queued):
    client, rid = queued
    resolve(client, rid, action="dismissed")
    second = resolve(client, rid, action="reply_sent", reply_text="late answer")
    assert second.status_code == 409
    assert client.get(f"/requests/{rid}").json()["resolution"]["action"] == "dismissed"


def test_only_review_items_can_be_resolved(client):
    rid = post(client)["id"]  # status ready
    assert resolve(client, rid, action="dismissed").status_code == 409


def test_resolve_unknown_request_is_404(client):
    assert resolve(client, "nope", action="dismissed").status_code == 404


def test_resolved_by_is_required(queued):
    client, rid = queued
    response = client.post(f"/review-queue/{rid}/resolve", json={"action": "dismissed"})
    assert response.status_code == 422
