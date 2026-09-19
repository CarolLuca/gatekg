from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any

import nest_asyncio

from gatekg.health.tracking import ensure_baseline_health_snapshot, record_run_health_snapshot
from gatekg.utils.config import Config, load_config
from gatekg.utils.context_cache import (
    build_context_group_id,
    build_context_work_group_id,
    clear_context_group_manifest,
    has_context_group_manifest,
    mark_context_group_ready,
    normalize_context_chunks,
)
from gatekg.utils.prompts import (
    AGGRESSIVE_WRITE_RLM_SYSTEM_PROMPT,
    BASE_RLM_SYSTEM_PROMPT,
    CUSTOM_BASE_RLM_SYSTEM_PROMPT,
    READ_ONLY_RLM_SYSTEM_PROMPT,
    READ_WRITE_RLM_SYSTEM_PROMPT,
)
from gatekg.utils.trajectory import capture_value

from .context import Context
from .mcp_kg import MCP_KG
from .providers import build_rlm_client

config: Config = load_config()
logger = logging.getLogger(__name__)

_MUTATING_KG_METHODS = {
    "add_episode_bulk",
    "add_memory",
    "add_triplet",
    "build_communities",
    "build_indices_and_constraints",
    "clear_graph",
    "close",
    "delete_entity_edge",
    "delete_entity_edges",
    "delete_episode",
    "delete_episodes",
    "init_knowledge",
    "initialize",
}

_NON_TOOL_KG_METHODS = {
    "initialize",
    "build_indices_and_constraints",
    "build_communities",
    "clear_active_group",
    "close",
    "has_group_data",
    "set_active_group",
    "set_active_groups",
}


def _run_async_in_current_loop(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)


def _kg_method_names() -> list[str]:
    methods = []
    for name, _value in inspect.getmembers(MCP_KG, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        methods.append(name)
    return methods


def _format_signature_without_self(method: Any) -> str:
    signature = inspect.signature(method)
    params = list(signature.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    cleaned_signature = inspect.Signature(
        parameters=params,
        return_annotation=signature.return_annotation,
    )
    return str(cleaned_signature)


def _tool_description(name: str) -> str:
    method = getattr(MCP_KG, name)
    signature = _format_signature_without_self(method)
    return f"Proxy to `MCP_KG.{name}{signature}`. Use `help({name})` in REPL for details."


def _normalize_add_memory_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    alias_map = {
        "memory_name": "name",
        "memory_body": "episode_body",
        "memory_description": "source_description",
    }
    for old_name, new_name in alias_map.items():
        if old_name in normalized and new_name not in normalized:
            normalized[new_name] = normalized.pop(old_name)
    return normalized


def _normalize_kg_call(
    method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    normalized_args = args
    normalized_kwargs = dict(kwargs)

    if method_name == "add_memory":
        normalized_kwargs = _normalize_add_memory_kwargs(normalized_kwargs)
        if (
            len(normalized_args) >= 3
            and "group_id" not in normalized_kwargs
            and "source_description" not in normalized_kwargs
        ):
            normalized_args = (normalized_args[0], normalized_args[1])
            normalized_kwargs["source_description"] = args[2]
            if len(args) > 3:
                normalized_args += args[3:]

    if method_name == "search_nodes":
        if (
            len(normalized_args) >= 2
            and isinstance(normalized_args[1], int)
            and "group_ids" not in normalized_kwargs
            and "max_nodes" not in normalized_kwargs
        ):
            normalized_args = (normalized_args[0],)
            normalized_kwargs["max_nodes"] = args[1]
            if len(args) > 2:
                normalized_args += args[2:]

    if method_name == "search_memory_facts":
        if (
            len(normalized_args) >= 2
            and isinstance(normalized_args[1], int)
            and "group_ids" not in normalized_kwargs
            and "max_facts" not in normalized_kwargs
        ):
            normalized_args = (normalized_args[0],)
            normalized_kwargs["max_facts"] = args[1]
            if len(args) > 2:
                normalized_args += args[2:]

    return normalized_args, normalized_kwargs


def _make_kg_proxy(method_name: str):
    kg_method = getattr(MCP_KG, method_name)

    @functools.wraps(kg_method)
    def proxy(self, *args, **kwargs):
        self._check_tool_scope(method_name)
        args, kwargs = _normalize_kg_call(method_name, args, kwargs)
        return self._execute_tool(method_name, args, kwargs)

    proxy.__doc__ = _tool_description(method_name)
    return proxy


ALL_KG_METHOD_NAMES = _kg_method_names()
READ_ONLY_KG_TOOL_NAMES = [
    name
    for name in ALL_KG_METHOD_NAMES
    if name not in _MUTATING_KG_METHODS and name not in _NON_TOOL_KG_METHODS
]
READ_WRITE_KG_TOOL_NAMES = [
    name for name in ALL_KG_METHOD_NAMES if name not in _NON_TOOL_KG_METHODS
]


def _supporting_facts_tool(context: Context) -> dict[str, Any]:
    return {
        "tool": context.get_supporting_facts,
        "description": "Return the current supporting facts provided to the RLM.",
    }


class _KGMethodMixin:
    kg: MCP_KG

    _trajectory_attempts: list[dict[str, Any]] | None = None

    def _record_proposal(self, name, args, kwargs, gate_event=None):
        if self._trajectory_attempts is None:
            return None
        record = {
            "tool": name,
            "args": capture_value(args),
            "kwargs": capture_value(kwargs),
            "gate_event": gate_event,
            "status": "pending",
        }
        self._trajectory_attempts.append(record)
        return record

    def _execute_tool(self, name, args, kwargs, *, gate_event=None):
        record = self._record_proposal(name, args, kwargs, gate_event)
        logger.info("KG tool call: %s args=%d kwargs=%s", name, len(args), list(kwargs)[:8])
        try:
            result = _run_async_in_current_loop(getattr(self.kg, name)(*args, **kwargs))
        except Exception as exc:
            if record is not None:
                record.update(status="error", error_type=type(exc).__name__)
            raise
        if record is not None:
            record.update(status="returned", result=capture_value(result))
        logger.info("KG tool result: %s -> %s", name, type(result).__name__)
        return result

    def _check_tool_scope(self, method_name: str) -> None:
        if method_name in READ_WRITE_KG_TOOL_NAMES and method_name in _MUTATING_KG_METHODS:
            if not self.kg.active_group_id or self.kg.active_group_id != getattr(
                self, "write_group_id", None
            ):
                raise ValueError("Graph writes require a prepared work group")

    def _build_kg_custom_tools(self, writable: bool) -> dict[str, Any]:
        tool_names = READ_WRITE_KG_TOOL_NAMES if writable else READ_ONLY_KG_TOOL_NAMES
        return {
            name: {
                "tool": getattr(self, name),
                "description": _tool_description(name),
            }
            for name in tool_names
        }

    async def init_kg(self) -> None:
        await self.kg.initialize()

    async def close_kg(self) -> None:
        await self.kg.close()

    async def clear_cached_group(self, group_id: str | None = None) -> None:
        resolved_group_id = group_id or self.kg.active_group_id
        if resolved_group_id is None:
            raise ValueError("No active group is set.")
        await self.kg.clear_graph(group_ids=[resolved_group_id])

    async def clear_all_cached_groups(self) -> None:
        # Passing an empty list bypasses the active-group default and clears the full DB.
        await self.kg.clear_graph(group_ids=[])


for _method_name in ALL_KG_METHOD_NAMES:
    setattr(_KGMethodMixin, _method_name, _make_kg_proxy(_method_name))


class _BaseMCPRLM:
    def __init__(
        self,
        *,
        custom_system_prompt: str,
        verbose: bool,
        custom_tools: dict[str, Any] | None = None,
    ):
        self.model_key = config.active_model_key
        self.kg_model_key = config.kg_model_key
        self.client = build_rlm_client(
            custom_tools=custom_tools,
            custom_system_prompt=custom_system_prompt,
            verbose=verbose,
        )

    def unsafe_predict(self, text: str, question: str | None = None) -> str:
        answer = self.client.completion(prompt=text, root_prompt=question)
        return answer.response

    def predict(self, text: str, question: str | None = None) -> str:
        return self.unsafe_predict(text, question)


class BaseRLM(_BaseMCPRLM):
    def __init__(self, verbose: bool = False):
        super().__init__(
            custom_system_prompt=BASE_RLM_SYSTEM_PROMPT,
            verbose=verbose,
        )


class _SupportingFactsMCPRLM(_BaseMCPRLM):
    def __init__(
        self,
        *,
        custom_system_prompt: str,
        verbose: bool,
        custom_tools: dict[str, Any] | None = None,
    ):
        self.context = Context()
        merged_custom_tools = dict(custom_tools or {})
        merged_custom_tools["get_supporting_facts"] = _supporting_facts_tool(self.context)
        super().__init__(
            custom_system_prompt=custom_system_prompt,
            verbose=verbose,
            custom_tools=merged_custom_tools,
        )

    def set_context(self, texts: list[str]) -> None:
        self.context.set_supporting_facts(texts)


class CustomBaseRLM(_SupportingFactsMCPRLM):
    def __init__(self, verbose: bool = False):
        super().__init__(
            custom_system_prompt=CUSTOM_BASE_RLM_SYSTEM_PROMPT,
            verbose=verbose,
        )


class _KGEnabledMCPRLM(_SupportingFactsMCPRLM, _KGMethodMixin):
    run_predict_in_worker_thread = False

    def __init__(
        self,
        *,
        custom_system_prompt: str,
        verbose: bool,
        writable_tools: bool,
    ):
        self.kg = MCP_KG()
        self.writable_tools = writable_tools
        self.base_group_id: str | None = None
        self.write_group_id: str | None = None
        self._ephemeral_group_ids: set[str] = set()
        super().__init__(
            custom_system_prompt=custom_system_prompt,
            verbose=verbose,
            custom_tools=self._build_kg_custom_tools(writable=writable_tools),
        )

    async def close_kg(self) -> None:
        try:
            for group_id in list(self._ephemeral_group_ids):
                logger.info("Clearing ephemeral KG group before close: %s", group_id)
                await self.kg.clear_graph(group_ids=[group_id])
                self._ephemeral_group_ids.discard(group_id)
        finally:
            logger.info("Closing KG connection")
            await self.kg.close()

    async def prepare_context(
        self,
        context_chunks: list[str],
        *,
        kg_context_chunks: list[str] | None = None,
        cache_model_key: str | None = None,
        rebuild: bool = False,
    ) -> str:
        resolved_cache_model_key = cache_model_key or self.kg_model_key
        normalized_chunks = normalize_context_chunks(context_chunks)
        normalized_kg_chunks = normalize_context_chunks(
            kg_context_chunks if kg_context_chunks is not None else context_chunks
        )
        base_group_id = build_context_group_id(
            cache_model_key=resolved_cache_model_key,
            context_chunks=normalized_kg_chunks,
        )
        logger.info(
            (
                "Preparing context: runtime_chunks=%d kg_chunks=%d cache_model_key=%s "
                "base_group_id=%s writable=%s rebuild=%s"
            ),
            len(normalized_chunks),
            len(normalized_kg_chunks),
            resolved_cache_model_key,
            base_group_id,
            self.writable_tools,
            rebuild,
        )
        self.set_context(normalized_chunks)

        if self.write_group_id is not None:
            logger.info("Clearing previous write group: %s", self.write_group_id)
            await self.kg.clear_graph(group_ids=[self.write_group_id])
            self._ephemeral_group_ids.discard(self.write_group_id)
            self.write_group_id = None

        self.kg.set_active_group(None)
        self.kg.set_active_groups([])
        self.base_group_id = base_group_id

        await self.init_kg()
        if rebuild:
            logger.info("Rebuilding cached base group: %s", base_group_id)
            clear_context_group_manifest(base_group_id)
            await self.clear_cached_group(base_group_id)

        has_manifest = has_context_group_manifest(base_group_id)
        has_data = await self.kg.has_group_data(base_group_id)

        if has_manifest and has_data:
            logger.info("Cache hit for base group %s. Reusing persisted KG.", base_group_id)
        else:
            if has_manifest and not has_data:
                logger.info(
                    "Cache manifest exists without KG data for base group %s. Rebuilding.",
                    base_group_id,
                )
                clear_context_group_manifest(base_group_id)
            elif has_data and not has_manifest:
                logger.info(
                    "KG data exists without ready manifest for base group %s. Clearing stale partial cache.",
                    base_group_id,
                )
                await self.clear_cached_group(base_group_id)
            else:
                logger.info(
                    "Cache miss for base group %s. Initializing KG from context.",
                    base_group_id,
                )

            await self.kg.init_knowledge(
                normalized_kg_chunks,
                group_id=base_group_id,
            )
            mark_context_group_ready(
                group_id=base_group_id,
                cache_model_key=resolved_cache_model_key,
                context_chunks=normalized_kg_chunks,
            )
            logger.info("Marked base group %s as ready.", base_group_id)

        await ensure_baseline_health_snapshot(
            kg=self.kg,
            base_group_id=base_group_id,
            cache_model_key=resolved_cache_model_key,
            context_chunk_count=len(normalized_kg_chunks),
            force=rebuild,
            metadata={"rlm_class": type(self).__name__},
        )

        if self.writable_tools:
            self.write_group_id = build_context_work_group_id(
                cache_model_key=resolved_cache_model_key,
                context_chunks=normalized_kg_chunks,
            )
            self._ephemeral_group_ids.add(self.write_group_id)
            self.kg.set_active_group(self.write_group_id)
            self.kg.set_active_groups([self.write_group_id, base_group_id])
            logger.info(
                "Writable run configured with write_group_id=%s search_groups=%s",
                self.write_group_id,
                [self.write_group_id, base_group_id],
            )
        else:
            self.kg.set_active_group(base_group_id)
            self.kg.set_active_groups([base_group_id])
            logger.info("Read-only run configured with search_groups=%s", [base_group_id])

        return base_group_id

    async def after_predict(
        self,
        *,
        prompt: str,
        question: str | None,
        prediction: str,
    ) -> None:
        if not self.writable_tools:
            return
        if self.base_group_id is None or self.write_group_id is None:
            return

        await record_run_health_snapshot(
            kg=self.kg,
            base_group_id=self.base_group_id,
            write_group_id=self.write_group_id,
            model_key=self.model_key,
            kg_model_key=self.kg_model_key,
            question=question,
            prompt=prompt,
            prediction=prediction,
            metadata={"rlm_class": type(self).__name__},
        )


class ReadOnlyRLM(_KGEnabledMCPRLM):
    def __init__(self, verbose: bool = False):
        super().__init__(
            custom_system_prompt=READ_ONLY_RLM_SYSTEM_PROMPT,
            verbose=verbose,
            writable_tools=False,
        )


class ReadWriteRLM(_KGEnabledMCPRLM):
    def __init__(self, verbose: bool = False):
        super().__init__(
            custom_system_prompt=READ_WRITE_RLM_SYSTEM_PROMPT,
            verbose=verbose,
            writable_tools=True,
        )


class AggressiveWriteRLM(_KGEnabledMCPRLM):
    def __init__(self, verbose: bool = False):
        super().__init__(
            custom_system_prompt=AGGRESSIVE_WRITE_RLM_SYSTEM_PROMPT,
            verbose=verbose,
            writable_tools=True,
        )
