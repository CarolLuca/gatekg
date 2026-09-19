from typing import Any

from openai import OpenAI

from gatekg.utils.config import Config, load_config
from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import truncate_text_tail
from gatekg.utils.retry import call_with_retries

config: Config = load_config()


class VLLMLLM:
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name or config.vllm.llm_model_name
        self.temperature = config.vllm.temperature
        self.enable_thinking = config.vllm.enable_thinking
        self.client = OpenAI(
            api_key=api_key or config.vllm_openai_api_key,
            base_url=base_url or config.vllm_openai_base_url,
            timeout=config.llm_runtime.timeout_seconds,
        )

    def unsafe_predict(self, text: str) -> str:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": truncate_text_tail(text, provider="vllm"),
                }
            ],
        }
        if self.enable_thinking is not None:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            }

        with LLM_REQUEST_LIMITER.sync_slot():
            response = self.client.chat.completions.create(**request_kwargs)
        output = response.choices[0].message.content
        if output is None:
            raise ValueError("vLLM returned an empty response.")
        return output

    def predict(self, text: str) -> str:
        return call_with_retries(
            lambda: self.unsafe_predict(text),
            attempts=config.llm_runtime.max_attempts,
            delay_seconds=config.llm_runtime.retry_delay_seconds,
        )
