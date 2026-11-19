"""LLM access behind a small interface.

The rest of the code depends only on `LlmClient`, so the provider (or a fake
in tests) can be swapped without touching business logic.
"""

import json
import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LlmError(Exception):
    """The model could not be reached or returned unusable output."""


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


class OpenAiCompatibleClient:
    """Works with any OpenAI-compatible API: OpenAI, Gemini, Groq, Ollama..."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    @classmethod
    def from_env(cls) -> "OpenAiCompatibleClient":
        from dotenv import load_dotenv

        load_dotenv()
        try:
            return cls(
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"],
                model=os.environ["LLM_MODEL"],
            )
        except KeyError as e:
            raise LlmError(f"missing setting {e} in .env (see .env.example)") from e

    def extract(self, system_prompt: str, user_text: str, schema: type[T]) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt}\n\nReply with a single JSON object "
                                   f"matching this JSON schema:\n{schema_json}",
                    },
                    {"role": "user", "content": user_text},
                ],
            )
        except Exception as e:  # network, auth, rate limit...
            raise LlmError(f"LLM request failed: {e}") from e

        content = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(content)
        except ValueError as e:
            raise LlmError(f"invalid model output: {e}") from e
