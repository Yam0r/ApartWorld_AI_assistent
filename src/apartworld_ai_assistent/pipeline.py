"""Scenario 2 end to end: incoming email -> reply draft or a task for a person."""

import uuid
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

from apartworld_ai_assistent.intake import BookingRequest, IntakeStatus, parse_email
from apartworld_ai_assistent.inventory import Inventory
from apartworld_ai_assistent.llm import LlmClient
from apartworld_ai_assistent.reply import ReplyStatus, compose_reply
from apartworld_ai_assistent.search import Offer, search


class RequestStatus(str, Enum):
    READY = "ready"                  # reply with verified options
    NEEDS_INFO = "needs_info"        # reply asks the client for missing details
    NO_OPTIONS = "no_options"        # reply says nothing is free
    MANUAL_REVIEW = "manual_review"  # a person has to handle it


class OfferSummary(BaseModel):
    unit_id: str
    name: str
    district: str
    total_price_czk: int
    url: str | None
    proximity: str | None


class ProcessedRequest(BaseModel):
    id: str
    received_at: datetime
    email: str
    status: RequestStatus
    reply_text: str | None = None
    reason: str | None = None
    missing: list[str] = Field(default_factory=list)
    request: BookingRequest | None = None
    offers: list[OfferSummary] = Field(default_factory=list)


ASK_FOR = {
    "en": ("Thank you for your message! To check availability, please tell us: {items}.",
           {"dates": "your arrival and departure dates", "guests": "the number of guests"}, "Apart World team"),
    "ru": ("Спасибо за ваше сообщение! Чтобы проверить наличие, пожалуйста, уточните: {items}.",
           {"dates": "даты заезда и выезда", "guests": "количество гостей"}, "Команда Apart World"),
    "cs": ("Děkujeme za Vaši zprávu! Abychom mohli ověřit dostupnost, napište nám prosím: {items}.",
           {"dates": "datum příjezdu a odjezdu", "guests": "počet hostů"}, "Tým Apart World"),
    "de": ("Vielen Dank für Ihre Nachricht! Um die Verfügbarkeit zu prüfen, nennen Sie uns bitte: {items}.",
           {"dates": "Ihr An- und Abreisedatum", "guests": "die Anzahl der Gäste"}, "Ihr Apart World Team"),
}


def ask_for_missing(missing: list[str], language: str) -> str:
    """Fixed, translated text: no LLM needed to ask for dates or guests."""
    template, names, signature = ASK_FOR.get(language, ASK_FOR["en"])
    items = []
    if {"check_in", "check_out"} & set(missing):
        items.append(names["dates"])
    if "guests" in missing:
        items.append(names["guests"])
    return template.format(items=", ".join(items)) + f"\n\n{signature}"


def _summary(offer: Offer) -> OfferSummary:
    return OfferSummary(
        unit_id=offer.unit.id,
        name=offer.unit.name,
        district=offer.unit.district,
        total_price_czk=offer.total_price_czk,
        url=next((l.url for l in offer.listings if l.url), None),
        proximity=offer.proximity,
    )


def handle_email(
    email: str,
    llm: LlmClient,
    inventory: Inventory,
    today: date,
    now: datetime,
) -> ProcessedRequest:
    base = dict(id=uuid.uuid4().hex[:12], received_at=now, email=email)

    intake = parse_email(email, llm, today)
    if intake.status == IntakeStatus.MANUAL_REVIEW:
        return ProcessedRequest(**base, status=RequestStatus.MANUAL_REVIEW, reason=intake.reason)
    if intake.status == IntakeStatus.NEEDS_INFO:
        language = intake.extracted.language if intake.extracted else "en"
        return ProcessedRequest(
            **base, status=RequestStatus.NEEDS_INFO, missing=intake.missing,
            reply_text=ask_for_missing(intake.missing, language),
        )

    request = intake.request
    found = search(inventory, request.check_in, request.check_out, request.guests, now)
    reply = compose_reply(request, found, llm, today)

    status = {
        ReplyStatus.READY: RequestStatus.READY,
        ReplyStatus.NO_OPTIONS: RequestStatus.NO_OPTIONS,
        ReplyStatus.MANUAL_REVIEW: RequestStatus.MANUAL_REVIEW,
    }[reply.status]
    return ProcessedRequest(
        **base, status=status, request=request, reply_text=reply.text, reason=reply.reason,
        offers=[_summary(o) for o in reply.offers],
    )
