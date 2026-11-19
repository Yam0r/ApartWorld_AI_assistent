"""LLM access behind a small interface.

The rest of the code depends only on `LlmClient`, so the provider (or a fake
in tests) can be swapped without touching business logic.
"""

import json
import os
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LlmError(Exception):
    """The model could not be reached or returned unusable output."""


def _type_hint(prop: dict) -> str:
    if "anyOf" in prop:
        return " or ".join(_type_hint(p) for p in prop["anyOf"])
    if prop.get("format") == "date":
        return "YYYY-MM-DD"
    if prop.get("type") == "array":
        return f"list of {_type_hint(prop.get('items', {}))}"
    return prop.get("type", "any")


def output_shape(schema: type[BaseModel]) -> dict[str, str]:
    """{"check_in": "YYYY-MM-DD or null", "guests": "integer or null", ...}"""
    props = schema.model_json_schema()["properties"]
    return {name: _type_hint(prop) for name, prop in props.items()}


class LlmClient(Protocol):
    def extract(self, system_prompt: str, user_text: str, schema: type[T]) -> T:
        """Return an instance of `schema` extracted from `user_text`."""
        ...


class FakeLlmClient:
    """Returns prepared answers. Used in tests: no network, no API key."""

    def __init__(self, answer: dict | None = None, error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def extract(self, system_prompt: str, user_text: str, schema: type[T]) -> T:
        self.calls.append((system_prompt, user_text))
        if self.error:
            raise self.error
        try:
            return schema.model_validate(self.answer)
        except ValueError as e:
            raise LlmError(f"invalid model output: {e}") from e


# HTTP statuses worth retrying: overloaded, rate limited, temporary server errors.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Statuses where another model may still work (model missing or refused).
TRY_NEXT_MODEL_STATUS = RETRYABLE_STATUS | {403, 404}


def _status_of(error: Exception) -> int | None:
    return getattr(error, "status_code", None)


def _is_retryable(error: Exception) -> bool:
    from openai import APIConnectionError, APITimeoutError

    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return True
    return _status_of(error) in RETRYABLE_STATUS


class OpenAiCompatibleClient:
    """Works with any OpenAI-compatible API: OpenAI, Gemini, Groq, Ollama...

    On temporary errors (503, 429, timeouts) the request is retried with a
    growing pause; if the main model keeps failing, the fallback model is tried.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        fallback_model: str | None = None,
        attempts: int = 3,
        backoff_seconds: float = 1.0,
        sleep=time.sleep,
        client=None,
    ):
        if client is None:
            from openai import OpenAI

            # Retries are done here, not by the SDK (its default waits up to 10 min).
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=20, max_retries=0)
        self._client = client
        self.models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])
        self._attempts = attempts
        self._backoff = backoff_seconds
        self._sleep = sleep

    @classmethod
    def from_env(cls) -> OpenAiCompatibleClient:
        from dotenv import load_dotenv

        load_dotenv()
        try:
            return cls(
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"],
                model=os.environ["LLM_MODEL"],
                fallback_model=os.environ.get("LLM_FALLBACK_MODEL") or None,
            )
        except KeyError as e:
            raise LlmError(f"missing setting {e} in .env (see .env.example)") from e

    def _create(self, model: str, messages: list[dict]):
        """One model: retry temporary errors with 1 s, 2 s, 4 s... pauses."""
        for attempt in range(self._attempts):
            try:
                return self._client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            except Exception as e:
                if not _is_retryable(e) or attempt == self._attempts - 1:
                    raise
                self._sleep(self._backoff * 2 ** attempt)

    def extract(self, system_prompt: str, user_text: str, schema: type[T]) -> T:
        # A flat example is easier for small models than a full JSON schema,
        # which they tend to echo back ({"properties": {...}}).
        shape = json.dumps(output_shape(schema), ensure_ascii=False, indent=2)
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nReply with ONE flat JSON object with "
                           f"exactly these keys (values show the expected type):\n{shape}",
            },
            {"role": "user", "content": user_text},
        ]

        errors = []
        for model in self.models:
            try:
                response = self._create(model, messages)
            except Exception as e:  # network, auth, rate limit...
                errors.append(f"{model}: {e}")
                if _is_retryable(e) or _status_of(e) in TRY_NEXT_MODEL_STATUS:
                    continue
                break  # e.g. 400/401: another model will not help
            content = response.choices[0].message.content or ""
            try:
                return schema.model_validate_json(content)
            except ValueError as e:
                raise LlmError(f"invalid model output: {e}") from e

        raise LlmError("LLM request failed: " + "; ".join(errors))
