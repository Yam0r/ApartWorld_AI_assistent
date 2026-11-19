"""OpenAiCompatibleClient retry and fallback logic, without network."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from apartworld_ai_assistent.llm import LlmError, OpenAiCompatibleClient


class Answer(BaseModel):
    value: int


class ApiError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


def ok(value: int = 1):
    message = SimpleNamespace(content=f'{{"value": {value}}}')
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedApi:
    """Plays back a list of outcomes per model: an exception or a response."""

    def __init__(self, script: dict[str, list]):
        self.script = {m: list(outcomes) for m, outcomes in script.items()}
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, **kwargs):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(script, fallback="backup", attempts=3):
    api = ScriptedApi(script)
    pauses: list[float] = []
    client = OpenAiCompatibleClient(
        base_url="x", api_key="x", model="main", fallback_model=fallback,
        attempts=attempts, backoff_seconds=1.0, sleep=pauses.append, client=api,
    )
    return client, api, pauses


def extract(client):
    return client.extract("system", "user", Answer)


def test_success_on_first_try():
    client, api, pauses = make_client({"main": [ok(7)]})
    assert extract(client).value == 7
    assert api.calls == ["main"] and pauses == []


def test_temporary_503_is_retried_with_growing_pause():
    client, api, pauses = make_client({"main": [ApiError(503), ApiError(503), ok(5)]})
    assert extract(client).value == 5
    assert api.calls == ["main", "main", "main"]
    assert pauses == [1.0, 2.0]


def test_rate_limit_429_is_retried():
    client, api, _ = make_client({"main": [ApiError(429), ok()]})
    assert extract(client).value == 1


def test_fallback_model_used_when_main_keeps_failing():
    client, api, _ = make_client({
        "main": [ApiError(503)] * 3,
        "backup": [ok(9)],
    })
    assert extract(client).value == 9
    assert api.calls == ["main"] * 3 + ["backup"]


def test_missing_model_404_goes_straight_to_fallback():
    client, api, pauses = make_client({"main": [ApiError(404)], "backup": [ok(3)]})
    assert extract(client).value == 3
    assert api.calls == ["main", "backup"] and pauses == []


def test_bad_key_401_fails_immediately_without_fallback():
    client, api, _ = make_client({"main": [ApiError(401)], "backup": [ok()]})
    with pytest.raises(LlmError, match="401"):
        extract(client)
    assert api.calls == ["main"]


def test_all_models_failing_raises_with_both_errors():
    client, _, _ = make_client({"main": [ApiError(503)] * 3, "backup": [ApiError(503)] * 3})
    with pytest.raises(LlmError) as err:
        extract(client)
    assert "main" in str(err.value) and "backup" in str(err.value)


def test_works_without_fallback_model():
    client, api, _ = make_client({"main": [ApiError(503)] * 3}, fallback=None)
    with pytest.raises(LlmError):
        extract(client)
    assert api.calls == ["main"] * 3


def test_invalid_json_is_an_error_not_a_retry():
    bad = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))])
    client, api, _ = make_client({"main": [bad], "backup": [ok()]})
    with pytest.raises(LlmError, match="invalid model output"):
        extract(client)
    assert api.calls == ["main"]
