import asyncio
from types import SimpleNamespace

import pytest
import rlm.clients.gemini as vendor_gemini_client

import gatekg.api.gemini_rlm_client as gemini_rlm_client


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = None
        self.candidates = []


class _FakeModels:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def _next_response(self):
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def generate_content(self, **kwargs):
        return self._next_response()


class _FakeAsyncModels(_FakeModels):
    async def generate_content(self, **kwargs):
        return self._next_response()


class _FakeGenAIClient:
    def __init__(self, sync_responses, async_responses=None):
        self.models = _FakeModels(sync_responses)
        self.aio = SimpleNamespace(
            models=_FakeAsyncModels(
                async_responses if async_responses is not None else sync_responses
            )
        )


def test_resilient_gemini_client_returns_empty_string_for_no_text_response(monkeypatch):
    fake_client = _FakeGenAIClient(
        sync_responses=[_FakeResponse(None), _FakeResponse("hello")],
    )
    monkeypatch.setattr(vendor_gemini_client.genai, "Client", lambda **kwargs: fake_client)
    sleep_calls = []
    monkeypatch.setattr(gemini_rlm_client.time, "sleep", lambda delay: sleep_calls.append(delay))

    client = gemini_rlm_client.ResilientGeminiRLMClient(
        api_key="test-key",
        model_name="gemini-test",
    )

    assert client.completion("prompt") == ""
    assert fake_client.models.calls == 1
    assert sleep_calls == []


def test_resilient_gemini_client_retries_request_exception(monkeypatch):
    fake_client = _FakeGenAIClient(
        sync_responses=[RuntimeError("503 UNAVAILABLE"), _FakeResponse("hello")],
    )
    monkeypatch.setattr(vendor_gemini_client.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(gemini_rlm_client.time, "sleep", lambda _: None)

    client = gemini_rlm_client.ResilientGeminiRLMClient(
        api_key="test-key",
        model_name="gemini-test",
    )

    assert client.completion("prompt") == "hello"
    assert fake_client.models.calls == 2


def test_resilient_gemini_client_async_returns_empty_string_for_no_text_response(monkeypatch):
    fake_client = _FakeGenAIClient(
        sync_responses=[_FakeResponse("unused")],
        async_responses=[_FakeResponse(None), _FakeResponse("hello")],
    )
    monkeypatch.setattr(vendor_gemini_client.genai, "Client", lambda **kwargs: fake_client)

    sleep_calls = []

    async def no_sleep(delay):
        sleep_calls.append(delay)
        return None

    monkeypatch.setattr(gemini_rlm_client.asyncio, "sleep", no_sleep)

    client = gemini_rlm_client.ResilientGeminiRLMClient(
        api_key="test-key",
        model_name="gemini-test",
    )

    assert asyncio.run(client.acompletion("prompt")) == ""
    assert fake_client.aio.models.calls == 1
    assert sleep_calls == []


def test_resilient_gemini_client_raises_after_exhausting_retries(monkeypatch):
    fake_client = _FakeGenAIClient(
        sync_responses=[
            RuntimeError("503 UNAVAILABLE"),
            RuntimeError("503 UNAVAILABLE"),
            RuntimeError("503 UNAVAILABLE"),
        ],
    )
    monkeypatch.setattr(vendor_gemini_client.genai, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(gemini_rlm_client.time, "sleep", lambda _: None)
    monkeypatch.setattr(gemini_rlm_client, "_MAX_REQUEST_ATTEMPTS", 3)

    client = gemini_rlm_client.ResilientGeminiRLMClient(
        api_key="test-key",
        model_name="gemini-test",
    )

    with pytest.raises(RuntimeError, match="503 UNAVAILABLE"):
        client.completion("prompt")


def test_resilient_gemini_client_can_initialize_vertex_ai(monkeypatch):
    captured_kwargs = {}
    fake_client = _FakeGenAIClient(sync_responses=[_FakeResponse("hello")])

    def fake_client_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setattr(vendor_gemini_client.genai, "Client", fake_client_factory)

    client = gemini_rlm_client.ResilientGeminiRLMClient(
        api_key="vertex-key",
        model_name="gemini-test",
        google_provider="vertex_ai",
        timeout=17,
    )

    assert client.completion("prompt") == "hello"
    assert captured_kwargs["vertexai"] is True
    assert captured_kwargs["api_key"] == "vertex-key"
    assert captured_kwargs["http_options"].timeout == 17000
