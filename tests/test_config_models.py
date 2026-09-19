import pytest

from gatekg.utils.config import (
    DEFAULT_INPUT_TOKEN_CAPS,
    Config,
    GoogleConfig,
    GraphitiRuntimeConfig,
    InputTokenCapsConfig,
    LLMRuntimeConfig,
    LoggingConfig,
    OllamaConfig,
    VllmConfig,
    _build_google_config,
    _build_input_token_caps_config,
    _build_llm_runtime_config,
    _build_ollama_config,
    _build_vllm_config,
)


def test_google_config_uses_small_model_as_canonical_llm_and_separate_reranker():
    config = _build_google_config(
        {
            "emb_model_name": "gemini-embedding-001",
            "temperature": 0.0,
            "google": {
                "llm_model_name": "gemini-2.5-pro",
                "kg_llm_model_name": "gemini-3-flash-preview",
                "reranker_model_name": "gemini-3-flash-preview",
                "judge_model_name": "gemini-3.1-pro-preview",
                "emb_model_name": "gemini-embedding-001",
            },
        }
    )

    assert config.llm_model_name == "gemini-2.5-pro"
    assert config.provider == "gemini_api"
    assert config.kg_llm_model_name == "gemini-3-flash-preview"
    assert config.reranker_model_name == "gemini-3-flash-preview"
    assert config.judge_model_name == "gemini-3.1-pro-preview"
    assert config.temperature == 0.0


def test_google_config_can_use_vertex_ai_provider_and_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-key")
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")

    config = _build_google_config(
        {
            "google": {
                "provider": "vertex_ai",
                "llm_model_name": "gemini-2.5-pro",
                "emb_model_name": "gemini-embedding-001",
            }
        }
    )

    assert config.provider == "vertex_ai"
    assert config.api_key == "vertex-key"
    assert config.gemini_api_key == "gemini-key"
    assert config.vertex_api_key == "vertex-key"


def test_google_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported `api.google.provider`"):
        _build_google_config(
            {
                "google": {
                    "provider": "not_real",
                    "llm_model_name": "gemini-2.5-pro",
                    "emb_model_name": "gemini-embedding-001",
                }
            }
        )


def test_ollama_config_uses_small_model_as_canonical_llm():
    config = _build_ollama_config(
        {
            "temperature": 0.0,
            "ollama": {
                "llm_model_name": "qwen3.5:35b",
                "emb_model_name": "nomic-embed-text:latest",
            },
        }
    )

    assert config.llm_model_name == "qwen3.5:35b"
    assert config.temperature == 0.0


def test_vllm_config_uses_default_qwen_profile():
    config = _build_vllm_config(
        {
            "temperature": 0.0,
            "vllm": {},
        }
    )

    assert config.llm_model_name == "Qwen/Qwen3.5-9B"
    assert config.openai_base_url == "http://127.0.0.1:8000/v1"
    assert config.openai_api_key == "vllm"
    assert config.max_model_len == 32768
    assert config.tensor_parallel_size == 1
    assert config.quantization is None
    assert config.enable_thinking is False


def test_config_exposes_google_kg_model_key_when_dedicated_model_is_set():
    config = Config(
        provider="ollama",
        google=GoogleConfig(
            llm_model_name="gemini-2.5-pro",
            kg_llm_model_name="gemini-3-flash-preview",
            emb_model_name="gemini-embedding-001",
            temperature=0.0,
        ),
        ollama=OllamaConfig(
            llm_model_name="qwen3.5:35b",
            emb_model_name="nomic-embed-text:latest",
            temperature=0.0,
        ),
        logging=LoggingConfig(),
        graphiti=GraphitiRuntimeConfig(),
        neo4j_uri=None,
        neo4j_username=None,
        neo4j_password=None,
        neo4j_database=None,
    )

    assert config.kg_provider == "google"
    assert config.kg_llm_model_name == "gemini-3-flash-preview"
    assert config.google_kg_llm_model_name == "gemini-3-flash-preview"
    assert config.kg_model_key == "google:gemini-3-flash-preview"
    assert config.kg_temperature == 0.0


def test_config_can_use_google_embeddings_with_ollama_llm():
    config = Config(
        provider="ollama",
        embedding_provider="google",
        google=GoogleConfig(
            llm_model_name="gemini-2.5-pro",
            emb_model_name="gemini-embedding-001",
            embedding_dim=1024,
            temperature=0.0,
        ),
        ollama=OllamaConfig(
            llm_model_name="qwen3.5:35b",
            emb_model_name="nomic-embed-text:latest",
            embedding_dim=768,
            temperature=0.2,
        ),
        logging=LoggingConfig(),
        graphiti=GraphitiRuntimeConfig(),
        neo4j_uri=None,
        neo4j_username=None,
        neo4j_password=None,
        neo4j_database=None,
    )

    assert config.provider == "ollama"
    assert config.embedding_provider == "google"
    assert config.llm_model_name == "qwen3.5:35b"
    assert config.temperature == 0.2
    assert config.emb_model_name == "gemini-embedding-001"
    assert config.embedding_dim == 1024


def test_config_can_default_vllm_to_google_embeddings():
    config = Config(
        provider="vllm",
        google=GoogleConfig(
            llm_model_name="gemini-2.5-pro",
            kg_llm_model_name="gemini-3-flash-preview",
            emb_model_name="gemini-embedding-001",
            embedding_dim=1024,
            temperature=0.0,
        ),
        ollama=OllamaConfig(
            llm_model_name="qwen3.5:35b",
            emb_model_name="nomic-embed-text:latest",
            embedding_dim=768,
            temperature=0.2,
        ),
        vllm=VllmConfig(
            llm_model_name="Qwen/Qwen3.5-9B",
            temperature=0.0,
        ),
        logging=LoggingConfig(),
        graphiti=GraphitiRuntimeConfig(),
        neo4j_uri=None,
        neo4j_username=None,
        neo4j_password=None,
        neo4j_database=None,
    )

    assert config.provider == "vllm"
    assert config.embedding_provider == "google"
    assert config.llm_model_name == "Qwen/Qwen3.5-9B"
    assert config.active_model_key == "vllm:Qwen/Qwen3.5-9B"
    assert config.emb_model_name == "gemini-embedding-001"


def test_input_token_caps_config_can_override_provider_specific_limits():
    caps = _build_input_token_caps_config(
        {
            "input_token_caps": {
                "google": 1_500_000,
                "ollama": 262_144,
                "vllm": 32_768,
            }
        }
    )

    assert caps.google == 1_500_000
    assert caps.ollama == 262_144
    assert caps.vllm == 32_768
    assert caps.for_provider("google") == 1_500_000
    assert caps.for_provider("ollama") == 262_144
    assert caps.for_provider("vllm") == 32_768


def test_config_defaults_input_token_caps_when_not_configured():
    config = Config(
        provider="google",
        google=GoogleConfig(
            llm_model_name="gemini-2.5-pro",
            emb_model_name="gemini-embedding-001",
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

    assert isinstance(config.input_token_caps, InputTokenCapsConfig)
    assert config.input_token_cap == DEFAULT_INPUT_TOKEN_CAPS["google"]
    assert config.input_token_cap_for("ollama") == DEFAULT_INPUT_TOKEN_CAPS["ollama"]
    assert config.input_token_cap_for("vllm") == DEFAULT_INPUT_TOKEN_CAPS["vllm"]


def test_config_defaults_embedding_provider_to_llm_provider():
    config = Config(
        provider="ollama",
        google=GoogleConfig(
            llm_model_name="gemini-2.5-pro",
            emb_model_name="gemini-embedding-001",
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

    assert config.embedding_provider == "ollama"
    assert config.emb_model_name == "nomic-embed-text:latest"


def test_llm_runtime_config_can_override_timeout():
    runtime = _build_llm_runtime_config(
        {
            "llm_runtime": {
                "timeout_seconds": 750,
            }
        }
    )

    assert runtime.timeout_seconds == 750.0


def test_config_defaults_llm_runtime_timeout_when_not_configured():
    config = Config(
        provider="google",
        google=GoogleConfig(
            llm_model_name="gemini-2.5-pro",
            emb_model_name="gemini-embedding-001",
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

    assert isinstance(config.llm_runtime, LLMRuntimeConfig)
    assert config.llm_runtime.timeout_seconds == 750.0
