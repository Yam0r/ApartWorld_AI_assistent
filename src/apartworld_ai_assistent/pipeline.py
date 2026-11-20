"""Scenario 2 end to end: incoming email -> reply draft or a task for a person."""

import uuid
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from apartworld_ai_assistent.intake import BookingRequest, IntakeStatus, parse_email
from apartworld_ai_assistent.inventory import Inventory
from apartworld_ai_assistent.llm import LlmClient
from apartworld_ai_assistent.reply import ReplyStatus, compose_reply
from apartworld_ai_assistent.search import Offer, rank_by_preferences, search


class RequestStatus(StrEnum):
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
    max_guests: int | None = None


class ResolutionAction(StrEnum):
    REPLY_SENT = "reply_sent"  # staff answered the client (own text or the draft)
    DISMISSED = "dismissed"    # no reply needed: spam, duplicate, handled by phone...


class Resolution(BaseModel):
    action: ResolutionAction
    resolved_by: str
    resolved_at: datetime
    reply_text: str | None = None
    note: str | None = None


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
    staff_note: str | None = None  # short summary for the person handling the item
    resolution: Resolution | None = None  # set when a person closed the review item


ASK_FOR = {
    "en": ("Thank you for your message! To check availability, please tell us: {items}.",
           {"dates": "your arrival and departure dates", "guests": "the number of guests"},
           "Apart World team"),
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


GROUP_REPLY = {
    "en": ("Thank you for your request! For a group of {guests} guests from {check_in} to {check_out} "
           "we will put together a package of several apartments. Our manager will contact you "
           "with the options and prices.", "Best regards,\nApart World team"),
    "ru": ("Спасибо за запрос! Для группы из {guests} человек на даты {check_in} – {check_out} "
           "мы подберём пакет из нескольких апартаментов. Менеджер свяжется с вами "
           "с вариантами и ценами.", "С уважением,\nкоманда Apart World"),
    "cs": ("Děkujeme za Vaši poptávku! Pro skupinu {guests} osob v termínu {check_in} – {check_out} "
           "připravíme balíček několika apartmánů. Náš manažer Vás bude kontaktovat "
           "s možnostmi a cenami.", "S pozdravem,\ntým Apart World"),
    "de": ("Vielen Dank für Ihre Anfrage! Für eine Gruppe von {guests} Personen vom {check_in} bis "
           "{check_out} stellen wir ein Paket aus mehreren Apartments zusammen. Unser Manager meldet "
           "sich mit Angeboten und Preisen.", "Mit freundlichen Grüßen\nIhr Apart World Team"),
}


def group_reply(request: BookingRequest) -> str:
    """Fixed, translated acknowledgement for a group: a person sends it after a glance."""
    text, signature = GROUP_REPLY.get(request.language, GROUP_REPLY["en"])
    body = text.format(guests=request.guests, check_in=f"{request.check_in:%d.%m.%Y}",
                       check_out=f"{request.check_out:%d.%m.%Y}")
    return f"{body}\n\n{signature}"


def handle_group(request: BookingRequest, inventory: Inventory, now: datetime) -> dict:
    """Material for the person building a group package.

    Every unit free on the dates (any size), requested district first, and the
    total number of places, so it is clear at once whether the group fits.
    """
    found = search(inventory, request.check_in, request.check_out, guests=1, now=now)
    free = rank_by_preferences(found.available, request.preferences, keep_far=True)
    places = sum(o.unit.max_guests for o in free)
    verdict = "enough for the whole group" if places >= request.guests else "NOT enough for the whole group"
    note = (f"Group of {request.guests}: {len(free)} units free on these dates, "
            f"{places} places in total ({verdict}).")
    close = [o for o in free if o.proximity in ("requested", "nearby")]
    if any(o.proximity for o in free):
        close_places = sum(o.unit.max_guests for o in close)
        fits = "fits there" if close_places >= request.guests else "other districts are needed"
        note += f" In or near the requested district: {len(close)} units, {close_places} places ({fits})."
    note += " Build a package from the offers below."
    return dict(offers=[_summary(o) for o in free], staff_note=note, reply_text=group_reply(request))


def _summary(offer: Offer) -> OfferSummary:
    return OfferSummary(
        unit_id=offer.unit.id,
        name=offer.unit.name,
        district=offer.unit.district,
        total_price_czk=offer.total_price_czk,
        url=next((li.url for li in offer.listings if li.url), None),
        proximity=offer.proximity,
        max_guests=offer.unit.max_guests,
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
        if intake.reason == "large_group" and intake.request:
            return ProcessedRequest(
                **base, status=RequestStatus.MANUAL_REVIEW, reason=intake.reason,
                request=intake.request, **handle_group(intake.request, inventory, now),
            )
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
