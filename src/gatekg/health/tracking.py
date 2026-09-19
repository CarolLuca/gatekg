from __future__ import annotations

import logging
from typing import Any

from .evaluator import evaluate_kg_health
from .store import HEALTH_SCHEMA_VERSION, KGHealthSnapshotStore

logger = logging.getLogger(__name__)


def _supports_health_tracking(kg: Any) -> bool:
    if hasattr(kg, "get_entity_nodes_by_group_ids") and hasattr(
        kg, "get_entity_edges_by_group_ids"
    ):
        return True
    return hasattr(getattr(kg, "client", None), "driver")


def _has_current_evaluator_version(snapshot: dict[str, Any]) -> bool:
    return snapshot.get("evaluator_version") == HEALTH_SCHEMA_VERSION


async def ensure_baseline_health_snapshot(
    *,
    kg: Any,
    base_group_id: str,
    cache_model_key: str,
    context_chunk_count: int,
    force: bool = False,
    metadata: dict[str, Any] | None = None,
    store: KGHealthSnapshotStore | None = None,
) -> dict[str, Any] | None:
    if not _supports_health_tracking(kg):
        return None

    resolved_store = store or KGHealthSnapshotStore()
    try:
        if not force:
            existing_snapshot = resolved_store.get_baseline_snapshot(base_group_id)
            if existing_snapshot is not None:
                if _has_current_evaluator_version(existing_snapshot):
                    return existing_snapshot
                logger.info(
                    (
                        "Recomputing baseline KG health due to evaluator version change. "
                        "base_group_id=%s stored_version=%s current_version=%s"
                    ),
                    base_group_id,
                    existing_snapshot.get("evaluator_version"),
                    HEALTH_SCHEMA_VERSION,
                )

        report = await evaluate_kg_health(kg, group_ids=[base_group_id])
        snapshot = resolved_store.upsert_baseline_snapshot(
            base_group_id=base_group_id,
            group_ids=[base_group_id],
            report=report,
            metadata={
                "cache_model_key": cache_model_key,
                "context_chunk_count": context_chunk_count,
                **(metadata or {}),
            },
        )
        logger.info(
            "Recorded baseline KG health. base_group_id=%s average_health_score=%.4f metric_scores=%s",
            base_group_id,
            snapshot["average_health_score"],
            snapshot["metric_scores"],
        )
        return snapshot
    except Exception:
        logger.exception("Failed to record baseline KG health for %s", base_group_id)
        return None


async def record_run_health_snapshot(
    *,
    kg: Any,
    base_group_id: str,
    write_group_id: str,
    model_key: str,
    kg_model_key: str,
    question: str | None = None,
    prompt: str | None = None,
    prediction: str | None = None,
    metadata: dict[str, Any] | None = None,
    store: KGHealthSnapshotStore | None = None,
) -> dict[str, Any] | None:
    if not _supports_health_tracking(kg):
        return None

    resolved_store = store or KGHealthSnapshotStore()
    try:
        group_ids = [group_id for group_id in (write_group_id, base_group_id) if group_id]
        report = await evaluate_kg_health(kg, group_ids=group_ids)
        snapshot = resolved_store.insert_run_snapshot(
            base_group_id=base_group_id,
            write_group_id=write_group_id,
            group_ids=group_ids,
            report=report,
            metadata={
                "model_key": model_key,
                "kg_model_key": kg_model_key,
                "question": question,
                "prompt_char_count": len(prompt or ""),
                "prediction_char_count": len(prediction or ""),
                **(metadata or {}),
            },
        )
        logger.info(
            (
                "Recorded run KG health. base_group_id=%s write_group_id=%s "
                "average_health_score=%.4f metric_scores=%s"
            ),
            base_group_id,
            write_group_id,
            snapshot["average_health_score"],
            snapshot["metric_scores"],
        )
        return snapshot
    except Exception:
        logger.exception(
            "Failed to record run KG health for base_group_id=%s write_group_id=%s",
            base_group_id,
            write_group_id,
        )
        return None
