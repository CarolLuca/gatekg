from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from gatekg.datasets import load_entries, resolve_dataset
from gatekg.utils.config import Config, load_config
from gatekg.utils.context_cache import build_context_group_id, context_cache_manifest_path
from gatekg.utils.eval_runner import ExperimentEntry


@dataclass(frozen=True)
class TaskCacheSelection:
    task_name: str
    cache_model_key: str
    entry_count: int
    group_ids: tuple[str, ...]


@dataclass(frozen=True)
class CacheCleanupPlan:
    task_filters: tuple[str, ...]
    cache_model_keys: tuple[str, ...]
    selections: tuple[TaskCacheSelection, ...]

    @property
    def group_ids(self) -> tuple[str, ...]:
        unique_group_ids = {
            group_id for selection in self.selections for group_id in selection.group_ids
        }
        return tuple(sorted(unique_group_ids))

    @property
    def manifest_paths(self) -> tuple[Path, ...]:
        return tuple(context_cache_manifest_path(group_id) for group_id in self.group_ids)


def _split_filter_values(values: Sequence[str]) -> tuple[str, ...]:
    expanded_values: list[str] = []
    for value in values:
        for part in value.split(","):
            normalized_part = part.strip()
            if normalized_part:
                expanded_values.append(normalized_part)
    return tuple(expanded_values)


def _normalize_task_name(task_name: str) -> str:
    return task_name.strip()


def _resolve_cache_model_key(raw_llm: str, config: Config) -> str:
    normalized_llm = raw_llm.strip()
    if not normalized_llm:
        raise ValueError("LLM filter must not be empty.")

    provider_prefixes = ("google:", "ollama:", "vllm:")
    if normalized_llm.startswith(provider_prefixes):
        return normalized_llm

    return f"{config.kg_provider}:{normalized_llm}"


def _load_entries_for_task(task_name: str, entry_limit: int = 0) -> list[ExperimentEntry]:
    return load_entries(resolve_dataset(task_name), limit=entry_limit)


def build_cache_cleanup_plan(
    *,
    tasks: Sequence[str],
    llms: Sequence[str],
    entry_limit: int = 0,
    config: Config | None = None,
) -> CacheCleanupPlan:
    resolved_config = config or load_config()
    normalized_tasks = tuple(
        dict.fromkeys(_normalize_task_name(task) for task in _split_filter_values(tasks))
    )
    if not normalized_tasks:
        raise ValueError("At least one task must be provided.")

    cache_model_keys = tuple(
        dict.fromkeys(
            _resolve_cache_model_key(llm, resolved_config) for llm in _split_filter_values(llms)
        )
    )
    if not cache_model_keys:
        raise ValueError("At least one llm must be provided.")

    entries_by_task = {
        task_name: _load_entries_for_task(task_name, entry_limit=entry_limit)
        for task_name in normalized_tasks
    }

    selections: list[TaskCacheSelection] = []
    for task_name, entries in entries_by_task.items():
        for cache_model_key in cache_model_keys:
            group_ids = {
                build_context_group_id(
                    cache_model_key=cache_model_key,
                    context_chunks=entry.kg_context_chunks or entry.context_chunks,
                )
                for entry in entries
            }
            selections.append(
                TaskCacheSelection(
                    task_name=task_name,
                    cache_model_key=cache_model_key,
                    entry_count=len(entries),
                    group_ids=tuple(sorted(group_ids)),
                )
            )

    return CacheCleanupPlan(
        task_filters=normalized_tasks,
        cache_model_keys=cache_model_keys,
        selections=tuple(selections),
    )


async def delete_cached_graph_groups(
    group_ids: Sequence[str], config: Config | None = None
) -> None:
    normalized_group_ids = [group_id for group_id in group_ids if group_id]
    if not normalized_group_ids:
        return

    resolved_config = config or load_config()
    if not resolved_config.neo4j_uri:
        raise ValueError("NEO4J_URI is required for graph cleanup")
    driver = Neo4jDriver(
        uri=resolved_config.neo4j_uri,
        user=resolved_config.neo4j_username,
        password=resolved_config.neo4j_password,
        database=resolved_config.neo4j_database or "neo4j",
    )
    try:
        await clear_data(driver, group_ids=normalized_group_ids)
    finally:
        await driver.close()


def delete_manifest_files(group_ids: Sequence[str]) -> list[Path]:
    deleted_paths: list[Path] = []
    for group_id in group_ids:
        manifest_path = context_cache_manifest_path(group_id)
        if manifest_path.exists():
            manifest_path.unlink()
            deleted_paths.append(manifest_path)
    return deleted_paths


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delete cached Neo4j context graphs and manifest files by rebuilding the "
            "deterministic cache group ids from task datasets and llm cache keys."
        )
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        required=True,
        help="Dataset name or JSONL path. Repeat to select multiple datasets.",
    )
    parser.add_argument(
        "--llm",
        dest="llms",
        action="append",
        required=True,
        help=(
            "LLM cache model name or cache model key. Examples: gemini-3.8-flash, "
            "google:gemini-3.8-flash"
        ),
    )
    parser.add_argument(
        "--entry-limit",
        type=int,
        default=0,
        help="Only consider the first N dataset entries per task. Default: all entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the cache groups that would be deleted without deleting anything.",
    )
    return parser


def _print_plan(plan: CacheCleanupPlan) -> None:
    print(f"Tasks: {', '.join(plan.task_filters)}")
    print(f"Cache model keys: {', '.join(plan.cache_model_keys)}")
    print(f"Selections: {len(plan.selections)}")
    for selection in plan.selections:
        print(
            f"- task={selection.task_name} cache_model_key={selection.cache_model_key} "
            f"entries={selection.entry_count} groups={len(selection.group_ids)}"
        )

    print(f"Unique cache groups: {len(plan.group_ids)}")
    for group_id in plan.group_ids:
        manifest_path = context_cache_manifest_path(group_id)
        manifest_status = "manifest-present" if manifest_path.exists() else "manifest-missing"
        print(f"- {group_id} [{manifest_status}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    plan = build_cache_cleanup_plan(
        tasks=args.tasks,
        llms=args.llms,
        entry_limit=args.entry_limit,
    )
    _print_plan(plan)

    if not plan.group_ids:
        print("No cached graph groups matched the provided filters.")
        return 1

    if args.dry_run:
        print("Dry run only. No graphs or manifests were deleted.")
        return 0

    asyncio.run(delete_cached_graph_groups(plan.group_ids))
    deleted_manifests = delete_manifest_files(plan.group_ids)
    print(
        f"Deleted {len(plan.group_ids)} cache group(s) from Neo4j and "
        f"{len(deleted_manifests)} manifest file(s)."
    )
    return 0
