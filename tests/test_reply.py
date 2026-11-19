from datetime import date, datetime

import pytest

from apartworld_ai_assistent.availability import CheckResult, Verdict
from apartworld_ai_assistent.intake import BookingRequest
from apartworld_ai_assistent.inventory import load_inventory
from apartworld_ai_assistent.llm import FakeLlmClient, LlmError
from apartworld_ai_assistent.reply import ReplyStatus, compose_reply, find_forbidden
from apartworld_ai_assistent.search import SearchResult, search

TODAY = date(2026, 11, 19)
NOW = datetime(2026, 11, 19, 12, 0)

REQUEST = BookingRequest(
    check_in=date(2027, 3, 10), check_out=date(2027, 3, 13),
    guests=4, language="en", client_name="Anna",
)
GOOD_TEXT = {
    "greeting": "Dear Anna,",
    "intro": "Thank you for your request for 10-13 March 2027 for 4 guests. Here are our options:",
    "closing": "We look forward to hosting you. Apart World team",
}


@pytest.fixture(scope="module")
def found() -> SearchResult:
    return search(load_inventory(), REQUEST.check_in, REQUEST.check_out, REQUEST.guests, NOW)


def compose(found, text=GOOD_TEXT, request=REQUEST, **kw):
    llm = FakeLlmClient(answer=text)
    return compose_reply(request, found, llm, TODAY, **kw), llm


# --- happy path ---

def test_reply_contains_llm_text_and_code_rendered_offers(found):
    result, _ = compose(found)
    assert result.status == ReplyStatus.READY
    assert result.text.startswith("Dear Anna,")
    first = found.available[0]
    assert first.unit.name in result.text
    assert f"{first.total_price_czk:,}".replace(",", " ") + " CZK" in result.text


def test_at_most_three_options(found):
    result, _ = compose(found)
    assert len(result.offers) == 3
    assert "4. " not in result.text


def test_direct_link_is_shown(found):
    river = next(o for o in found.available if o.unit.id == "river-deluxe-suite")
    result, _ = compose(found, max_options=len(found.available))
    assert river.listings[0].url in result.text


def test_price_validity_and_not_confirmed_note(found):
    result, _ = compose(found)
    assert "Prices valid until 21.11.2026" in result.text
    assert "not a confirmed booking" in result.text


def test_labels_follow_client_language(found):
    ru = REQUEST.model_copy(update={"language": "ru"})
    text = {"greeting": "Здравствуйте, Анна!", "intro": "Спасибо за запрос.", "closing": "Команда Apart World"}
    result, _ = compose(found, text=text, request=ru)
    assert "итого" in result.text and "Цены действительны до" in result.text


# --- the model never sees or writes prices and links ---

def test_llm_gets_no_prices_or_links(found):
    _, llm = compose(found)
    system_prompt, user_text = llm.calls[0]
    assert "CZK" not in user_text and "http" not in user_text
    assert "River Deluxe" not in user_text


@pytest.mark.parametrize("bad_intro, problem", [
    ("Our best option costs 4 500 CZK.", "currency"),
    ("Prices start from 1500 per night.", "number"),
    ("Book here: https://example.com", "link"),
    ("Call the owner at +420 777 000 000.", "phone"),
    ("Write to owner@example.com.", "email"),
])
def test_invented_details_go_to_manual_review(found, bad_intro, problem):
    result, _ = compose(found, text={**GOOD_TEXT, "intro": bad_intro})
    assert result.status == ReplyStatus.MANUAL_REVIEW
    assert problem in result.reason
    assert result.offers  # the person still sees the verified options


def test_dates_and_years_are_allowed():
    assert find_forbidden("Your stay 10.03.2027 – 13.03.2027 for 4 guests in 2027.") == []
    assert find_forbidden("From 2027-03-10 to 2027-03-13.") == []


def test_too_long_text_goes_to_manual_review(found):
    result, _ = compose(found, text={**GOOD_TEXT, "intro": "a" * 1000})
    assert result.status == ReplyStatus.MANUAL_REVIEW
    assert result.reason == "text_too_long"


def test_llm_failure_goes_to_manual_review(found):
    llm = FakeLlmClient(error=LlmError("503"))
    result = compose_reply(REQUEST, found, llm, TODAY)
    assert result.status == ReplyStatus.MANUAL_REVIEW
    assert result.reason.startswith("llm_error")


# --- nothing to offer ---

def test_only_unverified_options_go_to_a_person():
    unverified = CheckResult(unit_id="u1", verdict=Verdict.MANUAL_REVIEW, reason="stale_calendar")
    empty = SearchResult(available=[], manual_review=[unverified], rejected=[])
    llm = FakeLlmClient(answer=GOOD_TEXT)
    result = compose_reply(REQUEST, empty, llm, TODAY)
    assert result.status == ReplyStatus.MANUAL_REVIEW
    assert result.reason == "only_unverified_options"
    assert llm.calls == []  # no LLM call needed


def test_nothing_available_gives_polite_no_options_reply():
    empty = SearchResult(available=[], manual_review=[], rejected=[])
    text = {"greeting": "Dear Anna,", "intro": "Unfortunately nothing is free on these dates.",
            "closing": "We would be happy to check other dates. Apart World team"}
    result = compose_reply(REQUEST, empty, FakeLlmClient(answer=text), TODAY)
    assert result.status == ReplyStatus.NO_OPTIONS
    assert "nothing is free" in result.text
    assert "CZK" not in result.text


def test_requested_district_offered_first(found):
    smichov = REQUEST.model_copy(update={"preferences": ["near Smíchov"]})
    result, _ = compose(found, request=smichov)
    # two available units are in Smíchov: they lead, the cheapest other one follows
    districts = [o.unit.district for o in result.offers]
    assert ["Smíchov" in d for d in districts] == [True, True, False]
