from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

import rlm.clients.gemini as vendor_gemini_client
from google.genai import types
from rlm.clients.base_lm import BaseLM
from rlm.clients.gemini import GeminiClient

from gatekg.utils.config import Config, load_config
from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import truncate_prompt_payload

from .google_client import google_genai_client_kwargs

logger = logging.getLogger(__name__)
config: Config = load_config()

_MAX_REQUEST_ATTEMPTS = 10
_BASE_RETRY_DELAY_SECONDS = config.llm_runtime.retry_delay_seconds


def _retry_delay_seconds(attempt_index: int) -> float:
    return _BASE_RETRY_DELAY_SECONDS * (attempt_index + 1)


class ResilientGeminiRLMClient(GeminiClient):
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        google_provider: str = "gemini_api",
        **kwargs,
    ):
        model_name = model_name or load_config().google.llm_model_name
        BaseLM.__init__(self, model_name=model_name, **kwargs)
        self.client = vendor_gemini_client.genai.Client(
            **google_genai_client_kwargs(
                google_provider=google_provider,
                api_key=api_key,
                timeout_seconds=self.timeout,
            )
        )
        self.model_name = model_name

        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def _build_config(self, system_instruction: str | None) -> types.GenerateContentConfig | None:
        temperature = self.kwargs.get("temperature")
        if not system_instruction and temperature is None:
            return None
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

    def _response_text_or_none(self, response: Any) -> str | None:
        response_text = response.text
        if response_text is None:
            return None
        if not response_text.strip():
            return None
        return response_text

    def _response_part_types(self, response: Any) -> list[str]:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        part_types: list[str] = []
        for part in parts:
            if isinstance(getattr(part, "text", None), str):
                part_types.append("text")
            if hasattr(part, "model_dump"):
                dumped = part.model_dump(exclude={"text", "thought", "thought_signature"})
                part_types.extend(sorted(key for key, value in dumped.items() if value is not None))
        return part_types

    def _log_empty_response(self, model: str, response: Any) -> None:
        logger.warning(
            "Gemini returned no text response. Using empty string. model=%s part_types=%s",
            model,
            self._response_part_types(response),
        )

    def _log_retry(
        self,
        *,
        model: str,
        attempt: int,
        error: Exception,
    ) -> None:
        logger.warning(
            "Gemini RLM request failed. model=%s attempt=%d/%d error=%s",
            model,
            attempt,
            _MAX_REQUEST_ATTEMPTS,
            error,
        )

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        contents, system_instruction = self._prepare_contents(
            truncate_prompt_payload(prompt, provider="google")
        )
        resolved_model = model or self.model_name
        if not resolved_model:
            raise ValueError("Model name is required for Gemini client.")

        config = self._build_config(system_instruction)
        last_error: Exception | None = None

        for attempt in range(1, _MAX_REQUEST_ATTEMPTS + 1):
            try:
                with LLM_REQUEST_LIMITER.sync_slot():
                    response = self.client.models.generate_content(
                        model=resolved_model,
                        contents=contents,
                        config=config,
                    )
                self._track_cost(response, resolved_model)
                response_text = self._response_text_or_none(response)
                if response_text is not None:
                    return response_text
                self._log_empty_response(resolved_model, response)
                return ""
            except Exception as exc:
                last_error = exc

            if attempt < _MAX_REQUEST_ATTEMPTS:
                assert last_error is not None
                self._log_retry(model=resolved_model, attempt=attempt, error=last_error)
                time.sleep(_retry_delay_seconds(attempt - 1))

        if last_error is not None:
            raise last_error
        raise ValueError(
            f"Gemini request failed without a captured exception. model={resolved_model}"
        )

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        contents, system_instruction = self._prepare_contents(
            truncate_prompt_payload(prompt, provider="google")
        )
        resolved_model = model or self.model_name
        if not resolved_model:
            raise ValueError("Model name is required for Gemini client.")

        config = self._build_config(system_instruction)
        last_error: Exception | None = None

        for attempt in range(1, _MAX_REQUEST_ATTEMPTS + 1):
            try:
                async with LLM_REQUEST_LIMITER.async_slot():
                    response = await self.client.aio.models.generate_content(
                        model=resolved_model,
                        contents=contents,
                        config=config,
                    )
                self._track_cost(response, resolved_model)
                response_text = self._response_text_or_none(response)
                if response_text is not None:
                    return response_text
                self._log_empty_response(resolved_model, response)
                return ""
            except Exception as exc:
                last_error = exc

            if attempt < _MAX_REQUEST_ATTEMPTS:
                assert last_error is not None
                self._log_retry(model=resolved_model, attempt=attempt, error=last_error)
                await asyncio.sleep(_retry_delay_seconds(attempt - 1))

        if last_error is not None:
            raise last_error
        raise ValueError(
            f"Gemini request failed without a captured exception. model={resolved_model}"
        )
