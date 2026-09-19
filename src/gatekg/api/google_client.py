from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from gatekg.utils.config import SUPPORTED_GOOGLE_PROVIDERS, Config, load_config


def google_genai_client_kwargs(
    *,
    google_provider: str,
    api_key: str | None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if google_provider not in SUPPORTED_GOOGLE_PROVIDERS:
        supported = ", ".join(SUPPORTED_GOOGLE_PROVIDERS)
        raise ValueError(
            f"Unsupported google provider `{google_provider}`. Expected one of: {supported}"
        )

    client_kwargs: dict[str, Any] = {}
    if google_provider == "vertex_ai":
        client_kwargs["vertexai"] = True
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if timeout_seconds is not None:
        client_kwargs["http_options"] = types.HttpOptions(timeout=int(timeout_seconds * 1000))
    return client_kwargs


def build_google_genai_client(
    *,
    config: Config | None = None,
    timeout_seconds: float | None = None,
) -> genai.Client:
    resolved_config = config or load_config()
    return genai.Client(
        **google_genai_client_kwargs(
            google_provider=resolved_config.google_provider,
            api_key=resolved_config.google_api_key,
            timeout_seconds=timeout_seconds,
        )
    )
