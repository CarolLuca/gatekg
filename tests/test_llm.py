import gatekg.api.llm as llm_module
from gatekg.utils.config import (
    Config,
    GoogleConfig,
    GraphitiRuntimeConfig,
    LoggingConfig,
    OllamaConfig,
    VllmConfig,
)


def test_llm_uses_vllm_provider(monkeypatch):
    class DummyVLLM:
        def __init__(self):
            self.kind = "vllm"

    monkeypatch.setattr(
        llm_module,
        "config",
        Config(
            provider="vllm",
            embedding_provider="google",
            google=GoogleConfig(
                llm_model_name="gemini-2.5-pro",
                emb_model_name="gemini-embedding-001",
            ),
            ollama=OllamaConfig(
                llm_model_name="qwen3.5:35b",
                emb_model_name="nomic-embed-text:latest",
            ),
            vllm=VllmConfig(
                llm_model_name="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
            ),
            logging=LoggingConfig(),
            graphiti=GraphitiRuntimeConfig(),
            neo4j_uri=None,
            neo4j_username=None,
            neo4j_password=None,
            neo4j_database=None,
        ),
    )
    monkeypatch.setattr(llm_module, "VLLMLLM", DummyVLLM)

    client = llm_module.LLM()

    assert isinstance(client.client, DummyVLLM)
    assert client.model_key == "vllm:Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
