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
    assert client.get("/health").json() == {"status": "ok", "units": 12, "storage": "memory"}


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


def test_error_responses_are_documented_in_swagger(client):
    paths = client.get("/openapi.json").json()["paths"]
    resolve_codes = paths["/review-queue/{request_id}/resolve"]["post"]["responses"]
    assert {"200", "404", "409", "422"} <= set(resolve_codes)
    assert "404" in paths["/requests/{request_id}"]["get"]["responses"]


def test_swagger_example_does_not_impersonate_a_real_person(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ResolveIn"]
    assert schema["properties"]["resolved_by"]["examples"] == ["staff-name"]


def test_root_redirects_to_swagger(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert "/" not in client.get("/openapi.json").json()["paths"]


# --- large groups ---

SCHOOL = {"check_in": "2027-03-10", "check_out": "2027-03-13", "guests": 40,
          "language": "cs", "preferences": ["Smíchov"], "client_name": "Jana"}


def group_client(**overrides):
    return client_with({ExtractedRequest: {**SCHOOL, **overrides},
                        ReplyText: LlmError("group reply is not written by the LLM")})


def test_group_keeps_the_parsed_request_for_staff():
    body = post(group_client())
    assert body["status"] == "manual_review" and body["reason"] == "large_group"
    assert body["request"]["guests"] == 40
    assert body["request"]["client_name"] == "Jana"


def test_group_gets_free_units_of_any_size_requested_district_first():
    offers = post(group_client())["offers"]
    ids = [o["unit_id"] for o in offers]
    # Smíchov units first; far districts kept: a group may need them
    assert ids[:2] == ["river-deluxe-suite", "victory-castle-suite"]
    assert "old-town-residence" in ids
    assert all(o["max_guests"] for o in offers)
    assert "dicaprio-penthouse" not in ids  # booked on these dates


def test_group_note_says_whether_the_group_fits():
    body = post(group_client())
    places = sum(o["max_guests"] for o in body["offers"])
    assert f"{places} places in total" in body["staff_note"]
    assert "enough for the whole group" in body["staff_note"]
    huge = post(group_client(guests=500))
    assert "NOT enough" in huge["staff_note"]


def test_group_note_says_whether_the_district_alone_is_enough():
    # Smíchov: River Deluxe (8) + Victory Castle (16) = 24 places for 40 guests
    note = post(group_client())["staff_note"]
    assert "In or near the requested district: 2 units, 24 places (other districts are needed)" in note
    small = post(group_client(guests=20))["staff_note"]
    assert "24 places (fits there)" in small


def test_group_gets_a_translated_draft_reply():
    body = post(group_client())
    assert body["reply_text"].startswith(
        "Děkujeme za Vaši poptávku! Pro skupinu 40 osob v termínu 10.03.2027"
    )
    assert body["reply_text"].endswith("S pozdravem,\ntým Apart World")


def test_staff_can_send_the_group_draft_as_is():
    client = group_client()
    body = post(client)
    resolved = client.post(f"/review-queue/{body['id']}/resolve",
                           json={"action": "reply_sent", "resolved_by": "staff"}).json()
    assert resolved["resolution"]["reply_text"] == body["reply_text"]
