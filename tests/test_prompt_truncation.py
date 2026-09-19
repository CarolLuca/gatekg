from graphiti_core.prompts.models import Message

from gatekg.utils.prompt_truncation import (
    APPROX_CHARS_PER_TOKEN,
    APPROX_MAX_INPUT_CHARS,
    truncate_messages_tail,
    truncate_text_tail,
)


def test_truncate_text_tail_keeps_tail_within_limit():
    source = "a" * (APPROX_MAX_INPUT_CHARS + 100)

    truncated = truncate_text_tail(source, provider="google")

    assert len(truncated) == APPROX_MAX_INPUT_CHARS
    assert truncated == source[-APPROX_MAX_INPUT_CHARS:]


def test_truncate_messages_tail_preserves_leading_system_message():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "a" * (APPROX_MAX_INPUT_CHARS + 50)},
    ]

    truncated = truncate_messages_tail(messages, provider="google")

    assert truncated[0]["role"] == "system"
    assert truncated[0]["content"] == "system prompt"
    assert (
        truncated[1]["content"]
        == messages[1]["content"][-(APPROX_MAX_INPUT_CHARS - len("system prompt")) :]
    )


def test_truncate_messages_tail_supports_graphiti_messages():
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="b" * (APPROX_MAX_INPUT_CHARS + 25)),
    ]

    truncated = truncate_messages_tail(messages, provider="google")

    assert isinstance(truncated[0], Message)
    assert truncated[0].content == "sys"
    assert len(truncated[1].content) == APPROX_MAX_INPUT_CHARS - len("sys")


def test_truncate_text_tail_supports_provider_specific_cap():
    from gatekg.utils.prompt_truncation import config

    ollama_chars = config.input_token_cap_for("ollama") * APPROX_CHARS_PER_TOKEN
    source = "b" * (ollama_chars + 100)

    truncated = truncate_text_tail(source, provider="ollama")

    assert len(truncated) == ollama_chars
    assert truncated == source[-ollama_chars:]


def test_truncate_text_tail_supports_vllm_cap():
    from gatekg.utils.prompt_truncation import config

    vllm_chars = config.input_token_cap_for("vllm") * APPROX_CHARS_PER_TOKEN
    source = "c" * (vllm_chars + 100)

    truncated = truncate_text_tail(source, provider="vllm")

    assert len(truncated) == vllm_chars
    assert truncated == source[-vllm_chars:]
