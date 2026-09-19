from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from gatekg.utils.config import Config, load_config

config: Config = load_config()


APPROX_MAX_INPUT_TOKENS = config.input_token_cap_for("google")
APPROX_CHARS_PER_TOKEN = 4
APPROX_MAX_INPUT_CHARS = APPROX_MAX_INPUT_TOKENS * APPROX_CHARS_PER_TOKEN


def _resolve_max_input_tokens(
    *,
    provider: str | None,
    max_input_tokens: int | None,
) -> int:
    if max_input_tokens is not None:
        return max_input_tokens
    if provider is None:
        return config.input_token_cap
    return config.input_token_cap_for(provider)


def truncate_text_tail(
    text: str,
    *,
    provider: str | None = None,
    max_input_tokens: int | None = None,
    chars_per_token: int = APPROX_CHARS_PER_TOKEN,
) -> str:
    normalized_text = str(text)
    max_chars = (
        _resolve_max_input_tokens(
            provider=provider,
            max_input_tokens=max_input_tokens,
        )
        * chars_per_token
    )
    if len(normalized_text) <= max_chars:
        return normalized_text

    start = len(normalized_text) - max_chars
    window = normalized_text[start : min(len(normalized_text), start + 128)]
    for index, char in enumerate(window):
        if char.isspace():
            start += index + 1
            break
    return normalized_text[start:]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return str(content)
    return str(content)


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
        return None if role is None else str(role)
    role = getattr(message, "role", None)
    return None if role is None else str(role)


def _message_content_text(message: Any) -> str:
    if isinstance(message, dict):
        return _content_to_text(message.get("content"))
    return _content_to_text(getattr(message, "content", ""))


def _clone_message_with_content(message: Any, content: str) -> Any:
    if isinstance(message, dict):
        cloned = dict(message)
        cloned["content"] = content
        return cloned
    message_type = type(message)
    return message_type(role=message.role, content=content)


def _truncate_message_sequence_tail(messages: Sequence[Any], max_chars: int) -> list[Any]:
    if max_chars <= 0:
        return []

    remaining = max_chars
    kept_reversed: list[Any] = []
    for message in reversed(messages):
        if remaining <= 0:
            break

        content_text = _message_content_text(message)
        if not content_text:
            kept_reversed.append(_clone_message_with_content(message, ""))
            continue

        if len(content_text) <= remaining:
            kept_reversed.append(_clone_message_with_content(message, content_text))
            remaining -= len(content_text)
            continue

        kept_reversed.append(
            _clone_message_with_content(
                message,
                truncate_text_tail(content_text, max_input_tokens=remaining, chars_per_token=1),
            )
        )
        remaining = 0

    return list(reversed(kept_reversed))


def truncate_messages_tail(
    messages: Sequence[Any],
    *,
    provider: str | None = None,
    max_input_tokens: int | None = None,
    chars_per_token: int = APPROX_CHARS_PER_TOKEN,
) -> list[Any]:
    max_chars = (
        _resolve_max_input_tokens(
            provider=provider,
            max_input_tokens=max_input_tokens,
        )
        * chars_per_token
    )
    total_chars = sum(len(_message_content_text(message)) for message in messages)
    if total_chars <= max_chars:
        return [
            _clone_message_with_content(message, _message_content_text(message))
            for message in messages
        ]

    prefix_count = 0
    for message in messages:
        if _message_role(message) != "system":
            break
        prefix_count += 1

    if prefix_count == 0:
        return _truncate_message_sequence_tail(messages, max_chars)

    prefix = [
        _clone_message_with_content(message, _message_content_text(message))
        for message in messages[:prefix_count]
    ]
    prefix_chars = sum(len(_message_content_text(message)) for message in prefix)
    if prefix_chars >= max_chars:
        return _truncate_message_sequence_tail(prefix, max_chars)

    suffix = _truncate_message_sequence_tail(messages[prefix_count:], max_chars - prefix_chars)
    return prefix + suffix


def truncate_prompt_payload(
    payload: str | list[dict[str, Any]],
    *,
    provider: str | None = None,
    max_input_tokens: int | None = None,
    chars_per_token: int = APPROX_CHARS_PER_TOKEN,
) -> str | list[dict[str, Any]]:
    if isinstance(payload, str):
        return truncate_text_tail(
            payload,
            provider=provider,
            max_input_tokens=max_input_tokens,
            chars_per_token=chars_per_token,
        )
    return truncate_messages_tail(
        payload,
        provider=provider,
        max_input_tokens=max_input_tokens,
        chars_per_token=chars_per_token,
    )


def fit_judge_context(
    *, client: Any, model: str, limit: int, context: str, render: Callable[[str], str]
) -> tuple[str, int]:
    """Keep required fields intact and measure each candidate with the judge tokenizer."""

    def fits(prompt: str) -> bool:
        count = client.models.count_tokens(model=model, contents=prompt).total_tokens
        if count is None:
            raise ValueError("Judge tokenizer returned no token count")
        return count <= limit

    prompt = render(context)
    if fits(prompt):
        return prompt, 0
    prompt = render("")
    if not fits(prompt):
        raise ValueError("Required judge fields exceed the configured token budget")
    low, high = 0, len(context)
    while low + 1 < high:
        mid = (low + high) // 2
        candidate = render(context[:mid])
        if fits(candidate):
            low, prompt = mid, candidate
        else:
            high = mid
    return prompt, len(context) - low
