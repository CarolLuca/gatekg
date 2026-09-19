from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import requests
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import truncate_prompt_payload


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def _format_messages_as_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, dict):
        prompt = [prompt]

    if not isinstance(prompt, list):
        return str(prompt)

    lines: list[str] = []
    for message in prompt:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue

        role = str(message.get("role", "user")).upper()
        if role == "SYSTEM":
            continue
        content = _stringify_content(message.get("content"))
        if role == "TOOL":
            tool_name = message.get("tool_name")
            role = f"TOOL[{tool_name}]" if tool_name else "TOOL"

        if content:
            lines.append(f"{role}:\n{content}")
        else:
            lines.append(f"{role}:")

    # Ending with ASSISTANT cues the model to continue with the next reply only.
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _extract_system_prompt(prompt: Any) -> str | None:
    if not isinstance(prompt, list):
        return None

    system_parts: list[str] = []
    for message in prompt:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "system":
            continue
        content = _stringify_content(message.get("content"))
        if content:
            system_parts.append(content)

    if not system_parts:
        return None
    return "\n\n".join(system_parts)


class OllamaNativeRLMClient(BaseLM):
    """
    RLM client for Ollama using the native `/api/generate` endpoint.

    This bypasses Ollama's OpenAI-compatible chat tool-call parsing, which is
    currently brittle for agentic/code-generation prompts with GPT-OSS models.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        context_length: int | None = None,
        temperature: float | None = None,
        timeout: float = 300.0,
        think: bool | str | None = None,
        raw: bool = False,
        **kwargs,
    ):
        super().__init__(model_name=model_name, timeout=timeout, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.context_length = context_length
        self.temperature = temperature
        self.think = False if think is None else think
        self.raw = raw

        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        options: dict[str, Any] = {}
        if self.context_length is not None:
            options["num_ctx"] = self.context_length
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.raw:
            payload["raw"] = True
        if self.think is not None:
            payload["think"] = self.think
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        return payload

    def _track_usage(self, data: dict[str, Any]) -> None:
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        total_tokens = prompt_tokens + completion_tokens

        self.model_call_counts[self.model_name] += 1
        self.model_input_tokens[self.model_name] += prompt_tokens
        self.model_output_tokens[self.model_name] += completion_tokens
        self.model_total_tokens[self.model_name] += total_tokens

        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        if model is not None and model != self.model_name:
            raise ValueError(
                f"OllamaNativeRLMClient instance is bound to `{self.model_name}`, got `{model}`."
            )

        truncated_prompt = truncate_prompt_payload(prompt, provider="ollama")
        prompt_text = _format_messages_as_prompt(truncated_prompt)
        system_prompt = _extract_system_prompt(truncated_prompt)
        with LLM_REQUEST_LIMITER.sync_slot():
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self._headers(),
                json=self._payload(prompt_text, system=system_prompt),
                timeout=self.timeout,
            )
        response.raise_for_status()
        data = response.json()
        self._track_usage(data)
        return data.get("response", "")

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        return await asyncio.to_thread(self.completion, prompt, model)

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
