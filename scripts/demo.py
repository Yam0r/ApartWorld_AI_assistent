"""End-to-end demo of scenario 2: incoming email -> reply draft.

    python -m uv run python scripts/demo.py                 # built-in sample email
    python -m uv run python scripts/demo.py "your email text"

Uses the real LLM from .env.
"""

import sys
from datetime import date, datetime

from apartworld_ai_assistent.intake import IntakeStatus, parse_email
from apartworld_ai_assistent.inventory import load_inventory
from apartworld_ai_assistent.llm import LlmError, OpenAiCompatibleClient
from apartworld_ai_assistent.reply import compose_reply
from apartworld_ai_assistent.search import search

SAMPLE = ("Hi, we are 4 colleagues coming to Prague for a project, 10-13 March 2027. "
          "Something near Smíchov would be great. Thanks, Anna")

# The test calendars in data/units.json were "updated" at this moment.
# Using it as "now" keeps the demo stable; with the real clock they become stale.
DEMO_NOW = datetime(2026, 11, 19, 12, 0)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    email = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    today = date.today()

    try:
        llm = OpenAiCompatibleClient.from_env()
    except LlmError as e:
        print(f"Config error: {e}")
        return 1

    print("=== 1. Incoming email ===\n" + email)

    intake = parse_email(email, llm, today)
    print(f"\n=== 2. Intake: {intake.status.value} ===")
    if intake.status != IntakeStatus.READY:
        print(f"reason: {intake.reason}  missing: {intake.missing}")
        return 0
    request = intake.request
    print(request.model_dump_json(indent=2))

    found = search(load_inventory(), request.check_in, request.check_out, request.guests, DEMO_NOW)
    print("\n=== 3. Search ===")
    print("available:     ", [o.unit.id for o in found.available])
    print("manual review: ", [f"{r.unit_id} ({r.reason})" for r in found.manual_review])
    print("rejected:      ", [f"{r.unit_id} ({r.reason})" for r in found.rejected])

    reply = compose_reply(request, found, llm, today)
    print(f"\n=== 4. Reply: {reply.status.value} ===")
    if reply.reason:
        print(f"reason: {reply.reason}")
    if reply.text:
        print("\n" + reply.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
