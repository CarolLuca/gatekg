import gatekg.api.google_client as google_client
from gatekg.utils.config import (
    Config,
    GoogleConfig,
    GraphitiRuntimeConfig,
    LoggingConfig,
    OllamaConfig,
)


def _config(*, google_provider: str, api_key: str) -> Config:
    return Config(
        provider="google",
        embedding_provider="google",
        google=GoogleConfig(
            provider=google_provider,
            llm_model_name="gemini-2.5-pro",
            emb_model_name="gemini-embedding-001",
            api_key=api_key,
        ),
        ollama=OllamaConfig(
            llm_model_name="qwen3.5:35b",
            emb_model_name="nomic-embed-text:latest",
        ),
        logging=LoggingConfig(),
        graphiti=GraphitiRuntimeConfig(),
        neo4j_uri=None,
        neo4j_username=None,
        neo4j_password=None,
        neo4j_database=None,
    )


def test_build_google_genai_client_uses_gemini_api_by_default(monkeypatch):
    captured_kwargs = {}
    sentinel = object()

    def fake_client(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(google_client.genai, "Client", fake_client)

    client = google_client.build_google_genai_client(
        config=_config(google_provider="gemini_api", api_key="gemini-key"),
        timeout_seconds=12.5,
    )

    assert client is sentinel
    assert captured_kwargs["api_key"] == "gemini-key"
    assert "vertexai" not in captured_kwargs
    assert captured_kwargs["http_options"].timeout == 12500


def test_build_google_genai_client_uses_vertex_ai_transport(monkeypatch):
    captured_kwargs = {}
    sentinel = object()

    def fake_client(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(google_client.genai, "Client", fake_client)

    client = google_client.build_google_genai_client(
        config=_config(google_provider="vertex_ai", api_key="vertex-key"),
        timeout_seconds=12.5,
    )

    assert client is sentinel
    assert captured_kwargs["vertexai"] is True
    assert captured_kwargs["api_key"] == "vertex-key"
    assert captured_kwargs["http_options"].timeout == 12500
