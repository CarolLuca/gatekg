from gatekg.utils.config import Config, load_config
from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import truncate_text_tail
from gatekg.utils.retry import call_with_retries

from .google_client import build_google_genai_client
from .ollama_llm import OllamaLLM
from .vllm_llm import VLLMLLM

config: Config = load_config()


class GoogleLLM:
    def __init__(self):
        self.model_name = config.google.llm_model_name
        from google.genai import types

        self.client = build_google_genai_client(
            config=config,
            timeout_seconds=config.llm_runtime.timeout_seconds,
        )
        self._generation_config = types.GenerateContentConfig(
            temperature=config.google.temperature,
        )

    def unsafe_predict(self, text: str) -> str:
        with LLM_REQUEST_LIMITER.sync_slot():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=truncate_text_tail(text, provider="google"),
                config=self._generation_config,
            )
        if not response.text:
            raise ValueError("Google returned an empty response")
        return response.text

    def predict(self, text: str) -> str:
        return call_with_retries(
            lambda: self.unsafe_predict(text),
            attempts=config.llm_runtime.max_attempts,
            delay_seconds=config.llm_runtime.retry_delay_seconds,
        )


class LLM:
    def __init__(self):
        self.client: GoogleLLM | OllamaLLM | VLLMLLM
        self.model_key = config.active_model_key
        if config.provider == "google":
            self.client = GoogleLLM()
        elif config.provider == "ollama":
            self.client = OllamaLLM()
        elif config.provider == "vllm":
            self.client = VLLMLLM()
        else:
            raise ValueError(f"Unsupported provider `{config.provider}`")

    def unsafe_predict(self, text: str) -> str:
        return self.client.unsafe_predict(text)

    def predict(self, text: str) -> str:
        return self.client.predict(text)
