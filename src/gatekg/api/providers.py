from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.gemini_reranker_client import (
    DEFAULT_MODEL as DEFAULT_GEMINI_RERANKER_MODEL,
)
from graphiti_core.cross_encoder.gemini_reranker_client import (
    GeminiRerankerClient,
)
from graphiti_core.prompts.models import Message as GraphitiMessage
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    COMBINED_HYBRID_SEARCH_RRF,
)
from openai import AsyncOpenAI
from rlm import RLM

from gatekg.utils.config import Config, load_config
from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import truncate_messages_tail, truncate_text_tail

from .google_client import build_google_genai_client

config: Config = load_config()
_RLM_BACKENDS_PATCHED = False
logger = logging.getLogger(__name__)


def _should_use_gemini_reranker() -> bool:
    if config.provider == "google":
        return True
    return bool(config.google_api_key and config.google_reranker_model_name)


def _ollama_extra_body(context_length: int, temperature: float) -> dict[str, Any]:
    return {
        "options": {
            "num_ctx": context_length,
            "temperature": temperature,
        }
    }


def _merge_extra_body(
    default_extra_body: dict[str, Any],
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in default_extra_body.items()
    }
    if not extra_body:
        return merged

    for key, value in extra_body.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


class _AsyncOllamaChatCompletionsProxy:
    def __init__(self, completions: Any, default_extra_body: dict[str, Any]):
        self._completions = completions
        self._default_extra_body = default_extra_body

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = truncate_messages_tail(kwargs["messages"], provider="ollama")
        kwargs["extra_body"] = _merge_extra_body(
            self._default_extra_body,
            kwargs.get("extra_body"),
        )
        return await self._completions.create(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _AsyncOllamaChatProxy:
    def __init__(self, chat: Any, default_extra_body: dict[str, Any]):
        self._chat = chat
        self.completions = _AsyncOllamaChatCompletionsProxy(
            chat.completions,
            default_extra_body,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _AsyncOllamaOpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        context_length: int,
        temperature: float,
        *,
        timeout: float | None = None,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.chat = _AsyncOllamaChatProxy(
            self._client.chat,
            _ollama_extra_body(context_length, temperature),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _TruncatingGraphitiClientMixin:
    truncation_provider = "google"

    async def _generate_response(self, messages: list[GraphitiMessage], *args: Any, **kwargs: Any):
        truncated_messages = truncate_messages_tail(
            messages,
            provider=self.truncation_provider,
        )
        return await super()._generate_response(truncated_messages, *args, **kwargs)  # type: ignore[misc]


class _TruncatingGeminiRerankerClient(GeminiRerankerClient):
    _SYSTEM_INSTRUCTION = (
        "You are an expert at rating passage relevance. Respond with only a number from 0-100."
    )

    def _scoring_prompt(self, query: str, passage: str) -> str:
        return truncate_text_tail(
            f"""Rate how well this passage answers or relates to the query. Use a scale from 0 to 100.

Query: {query}

Passage: {passage}

Provide only a number between 0 and 100 (no explanation, just the number):""",
            provider="google",
        )

    async def _score_prompt(self, prompt: str) -> Any:
        from google.genai import types

        async with LLM_REQUEST_LIMITER.async_slot():
            return await self.client.aio.models.generate_content(
                model=self.config.model or DEFAULT_GEMINI_RERANKER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._SYSTEM_INSTRUCTION,
                    temperature=0.0,
                ),
            )

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if len(passages) <= 1:
            return [(passage, 1.0) for passage in passages]

        scoring_prompts = [self._scoring_prompt(query, passage) for passage in passages]

        try:
            responses = await asyncio.gather(
                *(self._score_prompt(prompt) for prompt in scoring_prompts)
            )
        except Exception as exc:
            error_message = str(exc).lower()
            if (
                "rate limit" in error_message
                or "quota" in error_message
                or "resource_exhausted" in error_message
                or "429" in str(exc)
            ):
                from graphiti_core.llm_client import RateLimitError

                raise RateLimitError from exc

            logger.error("Error in generating reranker response: %s", exc)
            raise

        results: list[tuple[str, float]] = []
        for passage, response in zip(passages, responses, strict=True):
            try:
                response_text = getattr(response, "text", None)
                if response_text:
                    score_match = re.search(r"\b(\d{1,3})\b", response_text.strip())
                    if score_match:
                        score = float(score_match.group(1))
                        results.append((passage, max(0.0, min(1.0, score / 100.0))))
                        continue
                    logger.warning(
                        "Could not extract numeric score from reranker response: %s",
                        response_text,
                    )
                else:
                    logger.warning("Empty response from Gemini reranker for passage scoring")
            except (ValueError, AttributeError) as exc:
                logger.warning("Error parsing reranker score: %s", exc)
            results.append((passage, 0.0))

        results.sort(reverse=True, key=lambda item: item[1])
        return results


def _ensure_rlm_backends_registered() -> None:
    global _RLM_BACKENDS_PATCHED
    if _RLM_BACKENDS_PATCHED:
        return

    import rlm.clients as rlm_clients
    import rlm.core.rlm as rlm_core_module

    original_get_client = rlm_clients.get_client

    def patched_get_client(backend: str, backend_kwargs: dict[str, Any]):
        if backend == "gemini":
            from .gemini_rlm_client import ResilientGeminiRLMClient

            return ResilientGeminiRLMClient(**backend_kwargs)
        if backend == "ollama_native":
            from .ollama_rlm_client import OllamaNativeRLMClient

            return OllamaNativeRLMClient(**backend_kwargs)
        return original_get_client(backend, backend_kwargs)

    rlm_clients.get_client = patched_get_client
    rlm_core_module.get_client = patched_get_client
    _RLM_BACKENDS_PATCHED = True


def build_graphiti_llm_client():
    from graphiti_core.llm_client.config import LLMConfig

    if config.kg_provider == "google":
        from graphiti_core.llm_client.gemini_client import GeminiClient

        class TruncatingGraphitiGeminiClient(_TruncatingGraphitiClientMixin, GeminiClient):
            truncation_provider = "google"
            pass

        return TruncatingGraphitiGeminiClient(
            config=LLMConfig(
                api_key=config.google_api_key,
                model=config.kg_llm_model_name,
                temperature=config.kg_temperature,
            ),
            client=build_google_genai_client(config=config),
        )

    if config.kg_provider == "ollama":
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        class TruncatingGraphitiOpenAIClient(
            _TruncatingGraphitiClientMixin,
            OpenAIGenericClient,
        ):
            truncation_provider = "ollama"
            pass

        return TruncatingGraphitiOpenAIClient(
            config=LLMConfig(
                api_key=config.ollama_openai_api_key,
                base_url=config.ollama_openai_base_url,
                model=config.kg_llm_model_name,
                temperature=config.kg_temperature,
            ),
            client=_AsyncOllamaOpenAIClient(
                api_key=config.ollama_openai_api_key,
                base_url=config.ollama_openai_base_url,
                context_length=config.ollama.context_length,
                temperature=config.kg_temperature,
                timeout=config.graphiti.llm_timeout_seconds,
            ),
        )

    raise ValueError(f"Unsupported provider `{config.provider}`")


def build_graphiti_embedder():
    if config.embedding_provider == "google":
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

        return GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=config.google_api_key,
                embedding_model=config.emb_model_name,
                embedding_dim=config.embedding_dim,
            ),
            client=build_google_genai_client(config=config),
        )

    if config.embedding_provider == "ollama":
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        return OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=config.ollama_openai_api_key,
                base_url=config.ollama_openai_base_url,
                embedding_model=config.emb_model_name,
                embedding_dim=config.embedding_dim,
            )
        )

    raise ValueError(f"Unsupported embedding provider `{config.embedding_provider}`")


class _LocalRankOrder(CrossEncoderClient):
    """Preserve candidate order if a custom recipe requests a cross-encoder locally."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 1.0 / index) for index, passage in enumerate(passages, start=1)]


def build_graphiti_cross_encoder():
    from graphiti_core.llm_client.config import LLMConfig

    if _should_use_gemini_reranker():
        return _TruncatingGeminiRerankerClient(
            config=LLMConfig(
                api_key=config.google_api_key,
                model=config.google_reranker_model_name,
                temperature=config.google.temperature,
            ),
            client=build_google_genai_client(config=config),
        )

    if config.provider in {"ollama", "vllm"}:
        return _LocalRankOrder()

    raise ValueError(f"Unsupported provider `{config.provider}`")


def default_graph_search_config() -> SearchConfig:
    if _should_use_gemini_reranker():
        return COMBINED_HYBRID_SEARCH_CROSS_ENCODER
    if config.provider in {"ollama", "vllm"}:
        return COMBINED_HYBRID_SEARCH_RRF
    raise ValueError(f"Unsupported provider `{config.provider}`")


def _rlm_backend_for_model(model_name: str) -> tuple[str, dict[str, Any]]:
    if config.provider == "google":
        return "gemini", {
            "model_name": model_name,
            "api_key": config.google_api_key,
            "google_provider": config.google_provider,
            "temperature": config.google.temperature,
            "timeout": config.llm_runtime.timeout_seconds,
        }

    if config.provider == "ollama":
        return "ollama_native", {
            "model_name": model_name,
            "api_key": config.ollama_openai_api_key,
            "base_url": config.ollama_base_url,
            "context_length": config.ollama.context_length,
            "temperature": config.ollama.temperature,
            "timeout": config.llm_runtime.timeout_seconds,
        }

    if config.provider == "vllm":
        return "vllm", {
            "model_name": model_name,
            "api_key": config.vllm_openai_api_key,
            "base_url": config.vllm_openai_base_url,
            "temperature": config.vllm.temperature,
            "timeout": config.llm_runtime.timeout_seconds,
        }

    raise ValueError(f"Unsupported provider `{config.provider}`")


def build_rlm_client(
    *,
    custom_system_prompt: str,
    verbose: bool,
    custom_tools: dict[str, Any] | None = None,
) -> RLM:
    _ensure_rlm_backends_registered()
    backend, backend_kwargs = _rlm_backend_for_model(config.llm_model_name)

    return RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        max_iterations=15,
        max_timeout=config.llm_runtime.timeout_seconds,
        custom_tools=custom_tools,
        custom_system_prompt=custom_system_prompt,
        environment="local",
        verbose=verbose,
    )
