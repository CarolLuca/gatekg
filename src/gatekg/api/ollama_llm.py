from openai import OpenAI

from gatekg.utils.config import Config, load_config
from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import truncate_text_tail
from gatekg.utils.retry import call_with_retries

config: Config = load_config()


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


class OllamaLLM:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name or config.ollama.llm_model_name
        self.context_length = config.ollama.context_length
        self.temperature = config.ollama.temperature
        self.client = OpenAI(
            api_key=api_key or config.ollama_openai_api_key,
            base_url=_normalize_openai_base_url(base_url or config.ollama_base_url),
            timeout=config.llm_runtime.timeout_seconds,
        )

    def unsafe_predict(self, text: str) -> str:
        with LLM_REQUEST_LIMITER.sync_slot():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": truncate_text_tail(text, provider="ollama")}],
                extra_body={
                    "options": {
                        "num_ctx": self.context_length,
                        "temperature": self.temperature,
                    }
                },
            )
        output = response.choices[0].message.content
        if output is None:
            raise ValueError("Ollama returned an empty response.")
        return output

    def predict(self, text: str) -> str:
        return call_with_retries(
            lambda: self.unsafe_predict(text),
            attempts=config.llm_runtime.max_attempts,
            delay_seconds=config.llm_runtime.retry_delay_seconds,
        )
