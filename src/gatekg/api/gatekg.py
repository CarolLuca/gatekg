from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gatekg.health.store import KGHealthSnapshotStore
from gatekg.health.tracking import ensure_baseline_health_snapshot, record_run_health_snapshot
from gatekg.paths import policy_artifact_dir
from gatekg.training.features import (
    gate_score_from_row,
    load_inference_bundle,
    normalize_variant_name,
    selector_scores_from_row,
    tool_is_mutating,
    writable_variant,
)
from gatekg.utils.context_cache import (
    build_context_group_id,
    build_context_work_group_id,
    clear_context_group_manifest,
    has_context_group_manifest,
    mark_context_group_ready,
    normalize_context_chunks,
)
from gatekg.utils.mcp_prompts import (
    AGGRESSIVE_WRITE_RLM_PROMPT_FUNC,
    READ_ONLY_RLM_SYSTEM_PROMPT_FUNC,
    READ_WRITE_RLM_SYSTEM_PROMPT_FUNC,
)
from gatekg.utils.prompts import (
    AGGRESSIVE_WRITE_RLM_SYSTEM_PROMPT,
    READ_ONLY_RLM_SYSTEM_PROMPT,
    READ_WRITE_RLM_SYSTEM_PROMPT,
)

from .mcp_rlm import (
    READ_ONLY_KG_TOOL_NAMES,
    READ_WRITE_KG_TOOL_NAMES,
    _KGEnabledMCPRLM,
    _normalize_kg_call,
    _supporting_facts_tool,
    _tool_description,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPTS = {
    "read_only_rlm": READ_ONLY_RLM_SYSTEM_PROMPT,
    "read_write_rlm": READ_WRITE_RLM_SYSTEM_PROMPT,
    "aggressive_write_rlm": AGGRESSIVE_WRITE_RLM_SYSTEM_PROMPT,
}
_QUESTION_PROMPTS: dict[str, Callable[[str], str]] = {
    "read_only_rlm": READ_ONLY_RLM_SYSTEM_PROMPT_FUNC,
    "read_write_rlm": READ_WRITE_RLM_SYSTEM_PROMPT_FUNC,
    "aggressive_write_rlm": AGGRESSIVE_WRITE_RLM_PROMPT_FUNC,
}


class GateKG(_KGEnabledMCPRLM):
    run_predict_in_worker_thread = False

    def __init__(
        self,
        *,
        dataset_variant: str,
        artifact_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.dataset_variant = str(dataset_variant)
        self.artifact_dir = policy_artifact_dir(artifact_dir)
        self._inference_bundle = load_inference_bundle(self.artifact_dir)
        self._health_store = KGHealthSnapshotStore()
        self.selected_variant = "read_only_rlm"
        self.selector_scores: dict[str, float] = {}
        self._baseline_snapshot: dict[str, Any] | None = None
        self._current_question: str | None = None
        self._current_context_chunk_count = 0
        self._current_kg_context_chunk_count = 0
        self._gate_events: list[dict[str, Any]] = []
        self._attempted_tool_counter: Counter[str] = Counter()
        self._attempted_tool_count = 0
        self._attempted_mutating_tool_count = 0
        super().__init__(
            custom_system_prompt=READ_ONLY_RLM_SYSTEM_PROMPT,
            verbose=verbose,
            writable_tools=False,
        )
        self._refresh_client_configuration()

    def _build_kg_custom_tools(self, writable: bool) -> dict[str, Any]:
        tool_names = READ_WRITE_KG_TOOL_NAMES if writable else READ_ONLY_KG_TOOL_NAMES
        tools: dict[str, Any] = {}
        for tool_name in tool_names:
            tools[tool_name] = {
                "tool": self._make_gated_tool(tool_name),
                "description": _tool_description(tool_name),
            }
        return tools

    def _make_gated_tool(self, tool_name: str):
        def tool(*args, **kwargs):
            return self._invoke_gated_tool(tool_name, *args, **kwargs)

        tool.__name__ = tool_name
        tool.__doc__ = _tool_description(tool_name)
        return tool

    def _refresh_client_configuration(self) -> None:
        custom_tools = self._build_kg_custom_tools(self.writable_tools)
        custom_tools["get_supporting_facts"] = _supporting_facts_tool(self.context)
        self.client.custom_tools = custom_tools
        self.client.custom_sub_tools = custom_tools
        self.client.system_prompt = _SYSTEM_PROMPTS[self.selected_variant]

    def build_prompt(self, question: str) -> str:
        return _QUESTION_PROMPTS[self.selected_variant](question)

    def _selector_feature_row(self, question: str) -> dict[str, Any]:
        return {
            "question": question,
            "dataset_variant": self.dataset_variant,
            "model_key": self.model_key,
            "baseline_average_health": (
                self._baseline_snapshot.get("average_health_score", 0.0)
                if self._baseline_snapshot is not None
                else 0.0
            ),
            "baseline_metric_scores": (
                self._baseline_snapshot.get("metric_scores") or {}
                if self._baseline_snapshot is not None
                else {}
            ),
            "context_chunk_count": self._current_context_chunk_count,
            "kg_context_chunk_count": self._current_kg_context_chunk_count,
        }

    def _policy_feature_row(self) -> dict[str, Any]:
        question = self._current_question or ""
        return {
            "question": question,
            "dataset_variant": self.dataset_variant,
            "model_key": self.model_key,
            "variant": self.selected_variant,
            "baseline_average_health": (
                self._baseline_snapshot.get("average_health_score", 0.0)
                if self._baseline_snapshot is not None
                else 0.0
            ),
            "baseline_metric_scores": (
                self._baseline_snapshot.get("metric_scores") or {}
                if self._baseline_snapshot is not None
                else {}
            ),
            "context_chunk_count": self._current_context_chunk_count,
            "kg_context_chunk_count": self._current_kg_context_chunk_count,
        }

    def _predict_selector(self, question: str) -> dict[str, float]:
        return selector_scores_from_row(
            model=self._inference_bundle.selector_model,
            feature_config=self._inference_bundle.feature_config,
            label_maps=self._inference_bundle.label_maps,
            row=self._selector_feature_row(question),
        )

    def _predict_gate(
        self,
        *,
        tool_name: str,
        call_index: int,
        prior_call_count: int,
        prior_mutating_call_count: int,
        prior_same_tool_count: int,
        prior_unique_tool_count: int,
        writable_flag: int,
    ) -> float:
        return gate_score_from_row(
            model=self._inference_bundle.tool_gate_model,
            feature_config=self._inference_bundle.feature_config,
            label_maps=self._inference_bundle.label_maps,
            row=self._policy_feature_row(),
            tool_name=tool_name,
            call_index=call_index,
            prior_call_count=prior_call_count,
            prior_mutating_call_count=prior_mutating_call_count,
            prior_same_tool_count=prior_same_tool_count,
            prior_unique_tool_count=prior_unique_tool_count,
            writable_flag=writable_flag,
        )

    def _reset_policy_state(self) -> None:
        self._gate_events = []
        self._attempted_tool_counter = Counter()
        self._attempted_tool_count = 0
        self._attempted_mutating_tool_count = 0

    async def _apply_selected_variant(
        self,
        *,
        selected_variant: str,
        cache_model_key: str,
        kg_context_chunks: list[str],
        base_group_id: str,
    ) -> None:
        normalized_variant = normalize_variant_name(selected_variant)
        self.selected_variant = normalized_variant
        self.writable_tools = writable_variant(normalized_variant)
        self.write_group_id = None

        if self.writable_tools:
            self.write_group_id = build_context_work_group_id(
                cache_model_key=cache_model_key,
                context_chunks=kg_context_chunks,
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

        self._refresh_client_configuration()
        self._reset_policy_state()

    async def prepare_context(
        self,
        context_chunks: list[str],
        *,
        question: str = "",
        kg_context_chunks: list[str] | None = None,
        cache_model_key: str | None = None,
        rebuild: bool = False,
    ) -> str:
        if not question:
            raise ValueError("GateKG requires a question to select a graph policy")
        resolved_cache_model_key = cache_model_key or self.kg_model_key
        normalized_chunks = normalize_context_chunks(context_chunks)
        normalized_kg_chunks = normalize_context_chunks(
            kg_context_chunks if kg_context_chunks is not None else context_chunks
        )
        base_group_id = build_context_group_id(
            cache_model_key=resolved_cache_model_key,
            context_chunks=normalized_kg_chunks,
        )
        self.set_context(normalized_chunks)
        self._current_question = question
        self._current_context_chunk_count = len(normalized_chunks)
        self._current_kg_context_chunk_count = len(normalized_kg_chunks)

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

        self._baseline_snapshot = await ensure_baseline_health_snapshot(
            kg=self.kg,
            base_group_id=base_group_id,
            cache_model_key=resolved_cache_model_key,
            context_chunk_count=len(normalized_kg_chunks),
            force=rebuild,
            metadata={"rlm_class": type(self).__name__},
            store=self._health_store,
        )
        if self._baseline_snapshot is None:
            self._baseline_snapshot = self._health_store.get_baseline_snapshot(base_group_id)

        self.selector_scores = self._predict_selector(question)
        selected_variant = max(
            self.selector_scores.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        await self._apply_selected_variant(
            selected_variant=selected_variant,
            cache_model_key=resolved_cache_model_key,
            kg_context_chunks=normalized_kg_chunks,
            base_group_id=base_group_id,
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
        logger.info(
            "GateKG selection event: %s",
            json.dumps(
                {
                    "question": question,
                    "dataset_variant": self.dataset_variant,
                    "selected_variant": self.selected_variant,
                    "selector_scores": self.selector_scores,
                    "base_group_id": self.base_group_id,
                    "write_group_id": self.write_group_id,
                    "baseline_average_health": (
                        self._baseline_snapshot.get("average_health_score")
                        if self._baseline_snapshot is not None
                        else None
                    ),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
        return base_group_id

    def _record_gate_attempt(self, tool_name: str) -> None:
        self._attempted_tool_counter[tool_name] += 1
        self._attempted_tool_count += 1
        if tool_is_mutating(tool_name):
            self._attempted_mutating_tool_count += 1

    def _invoke_gated_tool(self, tool_name: str, *args, **kwargs):
        self._check_tool_scope(tool_name)
        normalized_args, normalized_kwargs = _normalize_kg_call(tool_name, args, kwargs)
        call_index = self._attempted_tool_count
        prior_same_tool_count = self._attempted_tool_counter.get(tool_name, 0)
        prior_unique_tool_count = len(self._attempted_tool_counter)
        score = self._predict_gate(
            tool_name=tool_name,
            call_index=call_index,
            prior_call_count=self._attempted_tool_count,
            prior_mutating_call_count=self._attempted_mutating_tool_count,
            prior_same_tool_count=prior_same_tool_count,
            prior_unique_tool_count=prior_unique_tool_count,
            writable_flag=int(self.writable_tools),
        )
        allow = score >= self._inference_bundle.feature_config.gate_threshold

        event = {
            "question": self._current_question,
            "dataset_variant": self.dataset_variant,
            "selected_variant": self.selected_variant,
            "tool": tool_name,
            "decision": "allow" if allow else "block",
            "score": score,
            "call_index": call_index,
            "prior_call_count": self._attempted_tool_count,
            "prior_mutating_call_count": self._attempted_mutating_tool_count,
            "prior_same_tool_count": prior_same_tool_count,
            "prior_unique_tool_count": prior_unique_tool_count,
            "writable_flag": int(self.writable_tools),
        }

        logger.info(
            "GateKG gate: %s tool=%s score=%.4f variant=%s",
            event["decision"],
            tool_name,
            score,
            self.selected_variant,
        )
        logger.info(
            "GateKG gate event: %s",
            json.dumps(event, ensure_ascii=True, sort_keys=True),
        )
        self._gate_events.append(event)
        self._record_gate_attempt(tool_name)

        if not allow:
            record = self._record_proposal(tool_name, normalized_args, normalized_kwargs, event)
            if record is not None:
                record["status"] = "blocked"
            return {
                "blocked": True,
                "tool": tool_name,
                "reason": "tool_gate_blocked",
                "score": score,
                "selected_variant": self.selected_variant,
            }

        return self._execute_tool(tool_name, normalized_args, normalized_kwargs, gate_event=event)

    async def after_predict(
        self,
        *,
        prompt: str,
        question: str | None,
        prediction: str,
    ) -> None:
        metadata = {
            "rlm_class": type(self).__name__,
            "gatekg_dataset_variant": self.dataset_variant,
            "gatekg_selected_variant": self.selected_variant,
            "gatekg_selector_scores": self.selector_scores,
            "gatekg_allowed_tool_count": sum(
                1 for event in self._gate_events if event["decision"] == "allow"
            ),
            "gatekg_blocked_tool_count": sum(
                1 for event in self._gate_events if event["decision"] == "block"
            ),
        }
        run_snapshot = None
        if self.writable_tools and self.base_group_id and self.write_group_id:
            run_snapshot = await record_run_health_snapshot(
                kg=self.kg,
                base_group_id=self.base_group_id,
                write_group_id=self.write_group_id,
                model_key=self.model_key,
                kg_model_key=self.kg_model_key,
                question=question,
                prompt=prompt,
                prediction=prediction,
                metadata=metadata,
                store=self._health_store,
            )

        logger.info(
            "GateKG run summary event: %s",
            json.dumps(
                {
                    "question": question,
                    "dataset_variant": self.dataset_variant,
                    "selected_variant": self.selected_variant,
                    "selector_scores": self.selector_scores,
                    "allowed_tool_count": metadata["gatekg_allowed_tool_count"],
                    "blocked_tool_count": metadata["gatekg_blocked_tool_count"],
                    "write_group_id": self.write_group_id,
                    "run_snapshot_id": (
                        run_snapshot.get("snapshot_id") if run_snapshot is not None else None
                    ),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
