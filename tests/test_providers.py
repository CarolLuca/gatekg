import asyncio
from types import SimpleNamespace

import pytest
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.gemini import GeminiEmbedder
from graphiti_core.prompts.models import Message

import gatekg.api.providers as providers
from gatekg.utils.config import (
    Config,
    GoogleConfig,
    GraphitiRuntimeConfig,
    LoggingConfig,
    OllamaConfig,
    VllmConfig,
)
from gatekg.utils.prompt_truncation import APPROX_MAX_INPUT_CHARS


def test_build_rlm_client_registers_custom_backends(monkeypatch):
    calls: list[str] = []

    class DummyRLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        providers, "_ensure_rlm_backends_registered", lambda: calls.append("patched")
    )
    monkeypatch.setattr(providers, "RLM", DummyRLM)

    client = providers.build_rlm_client(
        custom_system_prompt="system",
        verbose=False,
    )

    assert calls == ["patched"]
    assert isinstance(client, DummyRLM)
    assert client.kwargs["backend_kwargs"]["temperature"] == providers.config.temperature
    assert (
        client.kwargs["backend_kwargs"]["timeout"] == providers.config.llm_runtime.timeout_seconds
    )
    assert client.kwargs["max_timeout"] == providers.config.llm_runtime.timeout_seconds


def test_truncating_graphiti_client_mixin_caps_messages_per_request():
    seen: dict[str, list[Message]] = {}

    class DummyBase:
        async def _generate_response(self, messages, *args, **kwargs):
            seen["messages"] = messages
            return {"ok": True}

    class DummyClient(providers._TruncatingGraphitiClientMixin, DummyBase):
        pass

    client = DummyClient()
    user_content = "a" * (APPROX_MAX_INPUT_CHARS + 64)
    result = asyncio.run(
        client._generate_response(
            [
                Message(role="system", content="sys"),
                Message(role="user", content=user_content),
            ]
        )
    )

    assert result == {"ok": True}
    truncated_messages = seen["messages"]
    assert truncated_messages[0].content == "sys"
    assert sum(len(message.content) for message in truncated_messages) == APPROX_MAX_INPUT_CHARS
    assert truncated_messages[1].content == user_content[-(APPROX_MAX_INPUT_CHARS - len("sys")) :]


def test_truncating_gemini_reranker_caps_scoring_prompt():
    recorded_calls: list[dict] = []

    class FakeModels:
        async def generate_content(self, **kwargs):
            recorded_calls.append(kwargs)
            return SimpleNamespace(text="87")

    reranker = providers._TruncatingGeminiRerankerClient(
        config=SimpleNamespace(model="gemini-reranker"),
        client=SimpleNamespace(aio=SimpleNamespace(models=FakeModels())),
    )
    long_passage = "p" * (APPROX_MAX_INPUT_CHARS + 256)

    results = asyncio.run(reranker.rank("query", [long_passage, "short"]))

    assert isinstance(reranker, CrossEncoderClient)
    assert len(recorded_calls) == 2
    first_prompt = recorded_calls[0]["contents"]
    assert len(first_prompt) <= APPROX_MAX_INPUT_CHARS
    assert first_prompt.endswith(
        "Provide only a number between 0 and 100 (no explanation, just the number):"
    )
    assert results == [(long_passage, 0.87), ("short", 0.87)]


def test_build_graphiti_embedder_uses_embedding_provider(monkeypatch):
    monkeypatch.setattr(
        providers,
        "config",
        Config(
            provider="ollama",
            embedding_provider="google",
            google=GoogleConfig(
                llm_model_name="gemini-2.5-pro",
                emb_model_name="gemini-embedding-001",
                embedding_dim=1024,
                api_key="test-google-key",
            ),
            ollama=OllamaConfig(
                llm_model_name="qwen3.5:35b",
                emb_model_name="nomic-embed-text:latest",
                embedding_dim=768,
                base_url="http://localhost:11435",
            ),
            logging=LoggingConfig(),
            graphiti=GraphitiRuntimeConfig(),
            neo4j_uri=None,
            neo4j_username=None,
            neo4j_password=None,
            neo4j_database=None,
        ),
    )

    embedder = providers.build_graphiti_embedder()

    assert isinstance(embedder, GeminiEmbedder)
    assert embedder.config.embedding_model == "gemini-embedding-001"
    assert embedder.config.embedding_dim == 1024
    assert embedder.config.api_key == "test-google-key"


def test_rlm_backend_for_vllm_uses_openai_compatible_server(monkeypatch):
    monkeypatch.setattr(
        providers,
        "config",
        Config(
            provider="vllm",
            embedding_provider="google",
            google=GoogleConfig(
                llm_model_name="gemini-2.5-pro",
                emb_model_name="gemini-embedding-001",
                embedding_dim=1024,
                api_key="test-google-key",
            ),
            ollama=OllamaConfig(
                llm_model_name="qwen3.5:35b",
                emb_model_name="nomic-embed-text:latest",
                embedding_dim=768,
                base_url="http://localhost:11435",
            ),
            vllm=VllmConfig(
                llm_model_name="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
                base_url="http://127.0.0.1:8000",
                temperature=0.1,
            ),
            logging=LoggingConfig(),
            graphiti=GraphitiRuntimeConfig(),
            neo4j_uri=None,
            neo4j_username=None,
            neo4j_password=None,
            neo4j_database=None,
        ),
    )

    backend, backend_kwargs = providers._rlm_backend_for_model(
        "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    )

    assert backend == "vllm"
    assert backend_kwargs == {
        "model_name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        "api_key": "vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "temperature": 0.1,
        "timeout": providers.config.llm_runtime.timeout_seconds,
    }


def test_rlm_backend_for_google_vertex_uses_selected_google_provider(monkeypatch):
    monkeypatch.setattr(
        providers,
        "config",
        Config(
            provider="google",
            embedding_provider="google",
            google=GoogleConfig(
                provider="vertex_ai",
                llm_model_name="gemini-2.5-pro",
                emb_model_name="gemini-embedding-001",
                embedding_dim=1024,
                api_key="test-vertex-key",
            ),
            ollama=OllamaConfig(
                llm_model_name="qwen3.5:35b",
                emb_model_name="nomic-embed-text:latest",
                embedding_dim=768,
                base_url="http://localhost:11435",
            ),
            logging=LoggingConfig(),
            graphiti=GraphitiRuntimeConfig(),
            neo4j_uri=None,
            neo4j_username=None,
            neo4j_password=None,
            neo4j_database=None,
        ),
    )

    backend, backend_kwargs = providers._rlm_backend_for_model("gemini-2.5-pro")

    assert backend == "gemini"
    assert backend_kwargs == {
        "model_name": "gemini-2.5-pro",
        "api_key": "test-vertex-key",
        "google_provider": "vertex_ai",
        "temperature": 0.0,
        "timeout": providers.config.llm_runtime.timeout_seconds,
    }


@pytest.mark.parametrize("provider", ["ollama", "vllm", "google"])
def test_real_graph_client_construction(monkeypatch, provider):
    from dataclasses import replace

    from graphiti_core import Graphiti

    import gatekg.api.kg as kg

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GRAPHITI_TELEMETRY_ENABLED", "false")
    config = replace(
        providers.config,
        provider=provider,
        neo4j_uri="bolt://localhost:7687",
        embedding_provider="google" if provider == "google" else "ollama",
        google=replace(
            providers.config.google, api_key="offline-test-key" if provider == "google" else None
        ),
    )
    monkeypatch.setattr(providers, "config", config)
    monkeypatch.setattr(kg, "config", config)
    graph = kg.GraphConnection()
    try:
        assert isinstance(graph.client, Graphiti)
        if provider == "google":
            assert isinstance(graph.client.cross_encoder, providers._TruncatingGeminiRerankerClient)
        else:
            assert graph.default_search_config == providers.COMBINED_HYBRID_SEARCH_RRF
            assert asyncio.run(graph.client.cross_encoder.rank("q", ["one", "two"])) == [
                ("one", 1.0),
                ("two", 0.5),
            ]
    finally:
        asyncio.run(graph.close())
