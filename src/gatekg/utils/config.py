import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from yaml import safe_load

SUPPORTED_PROVIDERS = ("google", "ollama", "vllm")
SUPPORTED_GOOGLE_PROVIDERS = ("gemini_api", "vertex_ai")
SUPPORTED_EMBEDDING_PROVIDERS = ("google", "ollama")
DEFAULT_INPUT_TOKEN_CAPS = {
    "google": 1_048_576,
    "ollama": 32_768,
    "vllm": 32_768,
}
DEFAULT_LLM_TIMEOUT_SECONDS = 750.0
PACKAGED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "default_config.yaml"
CONFIG_ENV_VAR = "GATEKG_CONFIG"


def _normalize_openai_compatible_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


@dataclass(frozen=True)
class ProviderConfig:
    llm_model_name: str
    emb_model_name: str = ""
    embedding_dim: int = 1024
    temperature: float = 0.0


@dataclass(frozen=True)
class GoogleConfig(ProviderConfig):
    provider: Literal["gemini_api", "vertex_ai"] = "gemini_api"
    api_key: str | None = None
    gemini_api_key: str | None = None
    vertex_api_key: str | None = None
    kg_llm_model_name: str | None = None
    judge_model_name: str | None = None
    reranker_model_name: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_GOOGLE_PROVIDERS:
            supported = ", ".join(SUPPORTED_GOOGLE_PROVIDERS)
            raise ValueError(
                f"Unsupported google provider `{self.provider}`. Expected one of: {supported}"
            )


@dataclass(frozen=True)
class OllamaConfig(ProviderConfig):
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    context_length: int = 32768
    kg_llm_model_name: str | None = None

    @property
    def openai_base_url(self) -> str:
        return _normalize_openai_compatible_base_url(self.base_url)

    @property
    def openai_api_key(self) -> str:
        # Ollama's OpenAI-compatible API accepts any non-empty key.
        return self.api_key or "ollama"


@dataclass(frozen=True)
class VllmConfig(ProviderConfig):
    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    cuda_visible_devices: str = "0"
    quantization: str | None = None
    language_model_only: bool = True
    enable_thinking: bool = False

    @property
    def openai_base_url(self) -> str:
        return _normalize_openai_compatible_base_url(self.base_url)

    @property
    def openai_api_key(self) -> str:
        # vLLM's OpenAI-compatible API accepts any non-empty key by default.
        return self.api_key or "vllm"


def _default_vllm_config() -> VllmConfig:
    return VllmConfig(llm_model_name="Qwen/Qwen3.5-9B")


@dataclass(frozen=True)
class LoggingConfig:
    root_level: str = "INFO"
    neo4j_level: str = "INFO"
    graphiti_level: str = "INFO"
    rlm_level: str = "INFO"
    http_level: str = "WARNING"
    openai_level: str = "WARNING"
    google_level: str = "WARNING"


@dataclass(frozen=True)
class GraphitiRuntimeConfig:
    max_coroutines: int | None = None
    llm_timeout_seconds: float = 600.0


@dataclass(frozen=True)
class LLMRuntimeConfig:
    max_attempts: int = 5
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    retry_delay_seconds: float = 5.0


@dataclass(frozen=True)
class InputTokenCapsConfig:
    google: int = DEFAULT_INPUT_TOKEN_CAPS["google"]
    ollama: int = DEFAULT_INPUT_TOKEN_CAPS["ollama"]
    vllm: int = DEFAULT_INPUT_TOKEN_CAPS["vllm"]

    def for_provider(self, provider: Literal["google", "ollama", "vllm"]) -> int:
        if provider == "google":
            return self.google
        if provider == "ollama":
            return self.ollama
        return self.vllm


@dataclass(frozen=True)
class Config:
    provider: Literal["google", "ollama", "vllm"]
    google: GoogleConfig
    ollama: OllamaConfig
    logging: LoggingConfig
    graphiti: GraphitiRuntimeConfig
    neo4j_uri: str | None
    neo4j_username: str | None
    neo4j_password: str | None
    neo4j_database: str | None
    vllm: VllmConfig = field(default_factory=_default_vllm_config)
    embedding_provider: Literal["google", "ollama"] | None = None
    kg_backend: Literal["google", "ollama"] | None = None
    input_token_caps: InputTokenCapsConfig = field(default_factory=InputTokenCapsConfig)
    llm_runtime: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)

    def __post_init__(self) -> None:
        if self.kg_backend not in (None, "google", "ollama"):
            raise ValueError("KG provider must be google or ollama")
        if self.provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(
                f"Unsupported provider `{self.provider}`. Expected one of: {supported}"
            )

        if self.embedding_provider is None and self.provider == "vllm":
            resolved_embedding_provider = "google"
        else:
            resolved_embedding_provider = (
                self.provider if self.embedding_provider is None else self.embedding_provider
            )
        if resolved_embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            supported = ", ".join(SUPPORTED_EMBEDDING_PROVIDERS)
            raise ValueError(
                "Unsupported embedding provider "
                f"`{resolved_embedding_provider}`. Expected one of: {supported}"
            )
        object.__setattr__(self, "embedding_provider", resolved_embedding_provider)

    @property
    def active_provider(self) -> ProviderConfig:
        if self.provider == "google":
            return self.google
        if self.provider == "ollama":
            return self.ollama
        return self.vllm

    @property
    def active_embedding_provider(self) -> GoogleConfig | OllamaConfig:
        if self.embedding_provider == "google":
            return self.google
        return self.ollama

    @property
    def llm_model_name(self) -> str:
        return self.active_provider.llm_model_name

    @property
    def emb_model_name(self) -> str:
        return self.active_embedding_provider.emb_model_name

    @property
    def embedding_dim(self) -> int:
        return self.active_embedding_provider.embedding_dim

    @property
    def temperature(self) -> float:
        return self.active_provider.temperature

    @property
    def google_api_key(self) -> str | None:
        return self.google.api_key

    @property
    def google_provider(self) -> Literal["gemini_api", "vertex_ai"]:
        return self.google.provider

    @property
    def google_judge_model_name(self) -> str:
        return self.google.judge_model_name or self.google.llm_model_name

    @property
    def google_kg_llm_model_name(self) -> str:
        return self.google.kg_llm_model_name or self.google.llm_model_name

    @property
    def google_reranker_model_name(self) -> str:
        return self.google.reranker_model_name or self.google.llm_model_name

    @property
    def kg_provider(self) -> Literal["google", "ollama"]:
        if self.kg_backend is not None:
            return self.kg_backend
        if self.provider == "google" or self.google.kg_llm_model_name:
            return "google"
        return "ollama"

    @property
    def kg_llm_model_name(self) -> str:
        if self.kg_provider == "google":
            return self.google_kg_llm_model_name
        return self.ollama.kg_llm_model_name or self.ollama.llm_model_name

    @property
    def active_model_key(self) -> str:
        return f"{self.provider}:{self.llm_model_name}"

    @property
    def kg_model_key(self) -> str:
        return f"{self.kg_provider}:{self.kg_llm_model_name}"

    @property
    def kg_temperature(self) -> float:
        if self.kg_provider == "google":
            return self.google.temperature
        return self.ollama.temperature

    def input_token_cap_for(self, provider: str) -> int:
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        return self.input_token_caps.for_provider(
            cast(Literal["google", "ollama", "vllm"], provider)
        )

    @property
    def input_token_cap(self) -> int:
        return self.input_token_cap_for(self.provider)

    @property
    def ollama_base_url(self) -> str:
        return self.ollama.base_url

    @property
    def ollama_openai_base_url(self) -> str:
        return self.ollama.openai_base_url

    @property
    def ollama_api_key(self) -> str | None:
        return self.ollama.api_key

    @property
    def ollama_openai_api_key(self) -> str:
        return self.ollama.openai_api_key

    @property
    def vllm_base_url(self) -> str:
        return self.vllm.base_url

    @property
    def vllm_openai_base_url(self) -> str:
        return self.vllm.openai_base_url

    @property
    def vllm_api_key(self) -> str | None:
        return self.vllm.api_key

    @property
    def vllm_openai_api_key(self) -> str:
        return self.vllm.openai_api_key


def _build_google_config(api_config: dict) -> GoogleConfig:
    google_config = api_config.get("google", {}) or {}

    configured_model_name = google_config.get("llm_model_name", "gemini-3.8-flash")
    emb_model_name = google_config.get("emb_model_name", "gemini-embedding-001")
    kg_llm_model_name = google_config.get("kg_llm_model_name")
    judge_model_name = google_config.get("judge_model_name")
    reranker_model_name = google_config.get("reranker_model_name")
    google_provider = str(google_config.get("provider", "gemini_api")).strip().lower()

    if google_provider not in SUPPORTED_GOOGLE_PROVIDERS:
        supported = ", ".join(SUPPORTED_GOOGLE_PROVIDERS)
        raise ValueError(
            f"Unsupported `api.google.provider` `{google_provider}`. Expected one of: {supported}"
        )

    gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    vertex_api_key = os.getenv("VERTEX_API_KEY")

    return GoogleConfig(
        provider=cast(Literal["gemini_api", "vertex_ai"], google_provider),
        llm_model_name=configured_model_name,
        emb_model_name=emb_model_name,
        embedding_dim=int(google_config.get("embedding_dim", 1024)),
        temperature=float(google_config.get("temperature", api_config.get("temperature", 0.0))),
        api_key=vertex_api_key if google_provider == "vertex_ai" else gemini_api_key,
        gemini_api_key=gemini_api_key,
        vertex_api_key=vertex_api_key,
        kg_llm_model_name=kg_llm_model_name,
        judge_model_name=judge_model_name or configured_model_name,
        reranker_model_name=reranker_model_name or configured_model_name,
    )


def _build_ollama_config(api_config: dict) -> OllamaConfig:
    ollama_config = api_config.get("ollama", {}) or {}
    default_temperature = api_config.get("temperature")
    configured_model_name = ollama_config.get("llm_model_name", "qwen3.5:9b")
    emb_model_name = ollama_config.get("emb_model_name", "nomic-embed-text:latest")
    context_length = ollama_config.get("context_length", 32768)

    return OllamaConfig(
        llm_model_name=configured_model_name,
        emb_model_name=emb_model_name,
        embedding_dim=int(ollama_config.get("embedding_dim", 768)),
        temperature=float(ollama_config.get("temperature", default_temperature or 0.0)),
        base_url=os.getenv(
            "OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434")
        ),
        api_key=os.getenv("OLLAMA_API_KEY", ollama_config.get("api_key")),
        context_length=int(context_length),
        kg_llm_model_name=ollama_config.get("kg_llm_model_name"),
    )


def _build_vllm_config(api_config: dict) -> VllmConfig:
    vllm_config = api_config.get("vllm", {}) or {}
    default_temperature = api_config.get("temperature")

    return VllmConfig(
        llm_model_name=vllm_config.get(
            "llm_model_name",
            "Qwen/Qwen3.5-9B",
        ),
        temperature=float(vllm_config.get("temperature", default_temperature or 0.0)),
        base_url=os.getenv("VLLM_BASE_URL", vllm_config.get("base_url", "http://127.0.0.1:8000")),
        api_key=os.getenv("VLLM_API_KEY", vllm_config.get("api_key")),
        max_model_len=int(vllm_config.get("max_model_len", 32768)),
        gpu_memory_utilization=float(vllm_config.get("gpu_memory_utilization", 0.90)),
        tensor_parallel_size=int(vllm_config.get("tensor_parallel_size", 1)),
        cuda_visible_devices=str(vllm_config.get("cuda_visible_devices", "0")),
        quantization=vllm_config.get("quantization"),
        language_model_only=bool(vllm_config.get("language_model_only", True)),
        enable_thinking=bool(vllm_config.get("enable_thinking", False)),
    )


def _build_logging_config(config_data: dict) -> LoggingConfig:
    logging_config = config_data.get("logging", {}) or {}
    return LoggingConfig(
        root_level=str(logging_config.get("root_level", "INFO")).upper(),
        neo4j_level=str(logging_config.get("neo4j_level", "INFO")).upper(),
        graphiti_level=str(logging_config.get("graphiti_level", "INFO")).upper(),
        rlm_level=str(logging_config.get("rlm_level", "INFO")).upper(),
        http_level=str(logging_config.get("http_level", "WARNING")).upper(),
        openai_level=str(logging_config.get("openai_level", "WARNING")).upper(),
        google_level=str(logging_config.get("google_level", "WARNING")).upper(),
    )


def _build_graphiti_runtime_config(
    config_data: dict,
    *,
    provider: Literal["google", "ollama", "vllm"],
) -> GraphitiRuntimeConfig:
    graphiti_config = config_data.get("graphiti", {}) or {}
    max_coroutines = graphiti_config.get("max_coroutines")
    if max_coroutines is None and provider == "ollama":
        # Local Ollama backends usually behave best with effectively serialized Graphiti LLM calls.
        # Users with multiple GPUs can opt into 2 explicitly in config.
        max_coroutines = 1

    return GraphitiRuntimeConfig(
        max_coroutines=int(max_coroutines) if max_coroutines is not None else None,
        llm_timeout_seconds=float(graphiti_config.get("llm_timeout_seconds", 600.0)),
    )


def _build_llm_runtime_config(config_data: dict) -> LLMRuntimeConfig:
    llm_runtime_config = config_data.get("llm_runtime", {}) or {}
    return LLMRuntimeConfig(
        max_attempts=int(llm_runtime_config.get("max_attempts", 5)),
        timeout_seconds=float(
            llm_runtime_config.get("timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS)
        ),
        retry_delay_seconds=float(llm_runtime_config.get("retry_delay_seconds", 5.0)),
    )


def _build_input_token_caps_config(api_config: dict) -> InputTokenCapsConfig:
    input_token_caps = api_config.get("input_token_caps", {}) or {}
    return InputTokenCapsConfig(
        google=int(input_token_caps.get("google", DEFAULT_INPUT_TOKEN_CAPS["google"])),
        ollama=int(input_token_caps.get("ollama", DEFAULT_INPUT_TOKEN_CAPS["ollama"])),
        vllm=int(input_token_caps.get("vllm", DEFAULT_INPUT_TOKEN_CAPS["vllm"])),
    )


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve an explicit, environment-provided, local, or packaged config."""

    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
    elif os.getenv(CONFIG_ENV_VAR):
        resolved = Path(os.environ[CONFIG_ENV_VAR]).expanduser().resolve()
    elif (Path.cwd() / "config.yaml").is_file():
        resolved = (Path.cwd() / "config.yaml").resolve()
    elif (Path.cwd() / "configs" / "default.yaml").is_file():
        resolved = (Path.cwd() / "configs" / "default.yaml").resolve()
    else:
        resolved = PACKAGED_CONFIG_PATH

    if not resolved.is_file():
        raise FileNotFoundError(
            f"GateKG config does not exist: {resolved}. "
            f"Set {CONFIG_ENV_VAR} or pass load_config(path)."
        )
    return resolved


def _load_environment(config_path: Path) -> None:
    configured_env = os.getenv("GATEKG_ENV_FILE")
    candidates = [
        Path(configured_env).expanduser() if configured_env else None,
        Path.cwd() / ".env",
        config_path.parent / ".env",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            load_dotenv(candidate, override=False)


def _apply_environment_overrides(config: dict[str, Any]) -> None:
    api_config = config.setdefault("api", {})
    if provider := os.getenv("GATEKG_PROVIDER"):
        api_config["provider"] = provider.strip().lower()
    if embedding_provider := os.getenv("GATEKG_EMBEDDING_PROVIDER"):
        api_config["embedding_provider"] = embedding_provider.strip().lower()
    if model_name := os.getenv("GATEKG_LLM_MODEL"):
        provider = str(api_config.get("provider", "google"))
        api_config.setdefault(provider, {})["llm_model_name"] = model_name
    if kg_provider := os.getenv("GATEKG_KG_PROVIDER"):
        api_config["kg_provider"] = kg_provider.strip().lower()
    if kg_model_name := os.getenv("GATEKG_KG_MODEL"):
        kg_provider = api_config.get("kg_provider") or (
            "google" if api_config.get("provider") == "google" else "ollama"
        )
        if kg_provider == "google":
            api_config.setdefault("google", {})["kg_llm_model_name"] = kg_model_name
        elif kg_provider == "ollama":
            api_config.setdefault("ollama", {})["kg_llm_model_name"] = kg_model_name
        else:
            raise ValueError("GATEKG_KG_PROVIDER must be `google` or `ollama`")
    if judge_model_name := os.getenv("GATEKG_JUDGE_MODEL"):
        api_config.setdefault("google", {})["judge_model_name"] = judge_model_name


def load_config(config_path: str | Path | None = None) -> Config:
    """Load validated runtime configuration.

    Precedence is: explicit function argument, ``GATEKG_CONFIG``, a local
    ``config.yaml``, ``configs/default.yaml``, then the packaged default.
    Secrets and provider endpoints are read from environment variables and are
    never required in YAML.
    """

    resolved_path = resolve_config_path(config_path)
    _load_environment(resolved_path)
    if config_path is None and os.getenv(CONFIG_ENV_VAR):
        resolved_path = resolve_config_path()

    with resolved_path.open("r", encoding="utf-8") as config_file:
        config = safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"GateKG config must be a YAML mapping: {resolved_path}")
    if not isinstance(config.get("api"), dict):
        raise ValueError(f"GateKG config is missing an `api` mapping: {resolved_path}")

    _apply_environment_overrides(config)
    api_config = config["api"]
    provider = str(api_config.get("provider", "google")).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported provider `{provider}`. Expected one of: {supported}")
    embedding_provider = api_config.get(
        "embedding_provider",
        "google" if provider == "vllm" else provider,
    )
    if embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        supported = ", ".join(SUPPORTED_EMBEDDING_PROVIDERS)
        raise ValueError(
            f"Unsupported embedding provider `{embedding_provider}`. Expected one of: {supported}"
        )

    typed_provider = cast(Literal["google", "ollama", "vllm"], provider)
    return Config(
        provider=typed_provider,
        google=_build_google_config(api_config),
        ollama=_build_ollama_config(api_config),
        vllm=_build_vllm_config(api_config),
        logging=_build_logging_config(config),
        graphiti=_build_graphiti_runtime_config(config, provider=typed_provider),
        input_token_caps=_build_input_token_caps_config(api_config),
        llm_runtime=_build_llm_runtime_config(config),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        neo4j_database=os.getenv("NEO4J_DATABASE"),
        embedding_provider=embedding_provider,
        kg_backend=api_config.get("kg_provider"),
    )
