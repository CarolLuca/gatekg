"""Training dataset construction."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from gatekg.api.judge_llm import _normalize_score
from gatekg.evaluation.hotpot import exact_match_score, f1_score
from gatekg.health.store import KGHealthSnapshotStore
from gatekg.records import read_run_manifest
from gatekg.training.features import (
    GATEKG_VARIANTS,
    normalize_variant_name,
    tool_is_mutating,
    writable_variant,
)
from gatekg.utils.eval_runner import ExperimentEntry
from gatekg.utils.trajectory import private_output, read_traces, trace_path

GATEKG_DATASET_PARSER_VERSION = "gatekg-v2"
ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TESTING_ENTRY_RE = re.compile(r"^Testing entry (\d+)/(\d+)$", re.MULTILINE)
PREPARE_CONTEXT_CURRENT_RE = re.compile(
    r"Preparing context: runtime_chunks=(?P<context>\d+) "
    r"kg_chunks=(?P<kg_context>\d+) "
    r"cache_model_key=(?P<cache_model_key>\S+) "
    r"base_group_id=(?P<base_group_id>mcp-cache_[A-Za-z0-9_]+) "
    r"writable=(?P<writable>True|False)"
)
WRITE_GROUP_RE = re.compile(
    r"Writable run configured with write_group_id="
    r"(?P<write_group_id>mcp-work_[A-Za-z0-9_]+)"
)
TOOL_CALL_RE = re.compile(
    r"KG tool call: (?P<tool>[A-Za-z0-9_]+) "
    r"args=(?P<args>.+?) kwargs=(?P<kwargs>.+)$"
)
LOG_TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \|")
TASK_HEADER_RE = re.compile(r"^Task:\s*(?P<value>.+)$", re.MULTILINE)
MODEL_HEADER_RE = re.compile(r"^Model:\s*(?P<value>.+)$", re.MULTILINE)
KG_MODEL_HEADER_RE = re.compile(r"^KG cache key:\s*(?P<value>.+)$", re.MULTILINE)
JUDGE_MODEL_HEADER_RE = re.compile(r"^Judge model:\s*(?P<value>.+)$", re.MULTILINE)
JUDGE_SCORE_RE = re.compile(r"^judge score:\s*(?P<score>.+)$")
JUDGE_EXPLANATION_PREFIX = "judge explanation:"


@dataclass(frozen=True)
class ExperimentSpec:
    task_name: str
    log_family: str
    variant: str
    dataset_variant: str
    dataset_path: Path


@dataclass(frozen=True)
class ParsedExecutionSegment:
    entry_position: int
    total_entries: int
    segment_text: str
    context_chunk_count: int | None
    kg_context_chunk_count: int | None
    base_group_id: str | None
    write_group_id: str | None
    cache_model_key: str | None
    timestamp: datetime | None
    tool_calls: list[dict[str, Any]]
    prediction: str | None


@dataclass(frozen=True)
class ParsedResultBlock:
    prediction: str | None
    judge_raw: str | None
    judge_binary: int | None
    judge_explanation: str | None


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _parse_header_value(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group("value").strip()


def resolve_experiment_spec(log_path: str | Path) -> ExperimentSpec:
    manifest = read_run_manifest(log_path)
    return ExperimentSpec(
        task_name=manifest["dataset"],
        log_family=manifest["runner"],
        variant=manifest["runner"],
        dataset_variant=manifest["dataset"],
        dataset_path=Path(manifest["dataset_path"]),
    )


def load_entries_for_log(
    log_path: str | Path,
    *,
    entry_limit: int = 0,
) -> tuple[ExperimentSpec, list[ExperimentEntry]]:
    manifest = read_run_manifest(log_path)
    entries = [ExperimentEntry(**entry) for entry in manifest["entries"]]
    return resolve_experiment_spec(log_path), entries[:entry_limit] if entry_limit else entries


def _parse_timestamp(segment_text: str) -> datetime | None:
    match = LOG_TIMESTAMP_RE.search(segment_text)
    if match is None:
        return None
    return datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def _parse_prepare_context(
    segment_text: str,
) -> tuple[int | None, int | None, str | None, str | None]:
    current_match = PREPARE_CONTEXT_CURRENT_RE.search(segment_text)
    if current_match is not None:
        return (
            int(current_match.group("context")),
            int(current_match.group("kg_context")),
            current_match.group("base_group_id"),
            current_match.group("cache_model_key"),
        )

    return None, None, None, None


def _parse_tool_calls(segment_text: str) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    lines = segment_text.splitlines()
    for line_index, line in enumerate(lines):
        match = TOOL_CALL_RE.search(line.strip())
        if match is None:
            continue
        tool_name = match.group("tool")
        context_start = max(0, line_index - 8)
        context_end = min(len(lines), line_index + 6)
        tool_calls.append(
            {
                "tool": tool_name,
                "args_summary": match.group("args").strip(),
                "kwargs_summary": match.group("kwargs").strip(),
                "is_mutating": tool_is_mutating(tool_name),
                "line_index": line_index,
                "nearby_context": "\n".join(lines[context_start:context_end]).strip(),
            }
        )
    return tool_calls


def _trajectory_calls(
    segment: ParsedExecutionSegment, trace: dict[str, Any] | None, *, gated: bool
):
    calls = segment.tool_calls
    if trace is None:
        events = [
            json.loads(line.split("GateKG gate event: ", 1)[1])
            for line in segment.segment_text.splitlines()
            if "GateKG gate event: " in line
        ]
        if gated:
            allowed = [event for event in events if event["decision"] == "allow"]
            if [event["tool"] for event in allowed] != [call["tool"] for call in calls]:
                raise ValueError("Policy log has unmatched gate decisions and KG tool calls")
            calls = [
                {**call, "gate_event": event} for call, event in zip(calls, allowed, strict=True)
            ]
        return calls, events

    attempts = trace["attempts"]
    events = [call["gate_event"] for call in attempts if call["gate_event"] is not None]
    if gated and len(events) != len(attempts):
        raise ValueError("Policy trajectory is missing runtime gate decisions")
    executed = [call for call in attempts if call["status"] != "blocked"]
    if any(call["status"] not in {"returned", "error"} for call in executed):
        raise ValueError("Incomplete tool execution in trajectory")
    # Console context is optional. Only attach it if all executed calls align.
    aligned = [call["tool"] for call in executed] == [call["tool"] for call in calls]
    return [
        {
            **call,
            "is_mutating": tool_is_mutating(call["tool"]),
            "nearby_context": calls[i]["nearby_context"] if aligned else "",
        }
        for i, call in enumerate(executed)
    ], events


def _parse_final_answer(segment_text: str) -> str | None:
    stripped_lines = [line.strip() for line in strip_ansi(segment_text).splitlines()]
    final_answer_start = None
    for index, line in enumerate(stripped_lines):
        if "Final Answer" in line:
            final_answer_start = index
    if final_answer_start is None:
        return None

    candidate_lines: list[str] = []
    for line in stripped_lines[final_answer_start + 1 :]:
        if not line:
            if candidate_lines:
                break
            continue
        if line.startswith("Testing entry "):
            break
        if line.startswith("judge score:"):
            break
        if line == "-----------------":
            if candidate_lines:
                break
            continue
        cleaned = line.strip("│").strip()
        if cleaned:
            candidate_lines.append(cleaned)
    if not candidate_lines:
        return None
    return "\n".join(candidate_lines).strip() or None


def split_execution_segments(log_text: str) -> list[ParsedExecutionSegment]:
    stripped = strip_ansi(log_text)
    matches = list(TESTING_ENTRY_RE.finditer(stripped))
    segments: list[ParsedExecutionSegment] = []
    for index, match in enumerate(matches):
        segment_start = match.start()
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        segment_text = stripped[segment_start:segment_end]
        (
            context_chunk_count,
            kg_context_chunk_count,
            base_group_id,
            cache_model_key,
        ) = _parse_prepare_context(segment_text)
        write_group_match = WRITE_GROUP_RE.search(segment_text)
        segments.append(
            ParsedExecutionSegment(
                entry_position=int(match.group(1)),
                total_entries=int(match.group(2)),
                segment_text=segment_text,
                context_chunk_count=context_chunk_count,
                kg_context_chunk_count=kg_context_chunk_count,
                base_group_id=base_group_id,
                write_group_id=(
                    write_group_match.group("write_group_id")
                    if write_group_match is not None
                    else None
                ),
                cache_model_key=cache_model_key,
                timestamp=_parse_timestamp(segment_text),
                tool_calls=_parse_tool_calls(segment_text),
                prediction=_parse_final_answer(segment_text),
            )
        )
    return segments


def extract_result_blocks(log_text: str) -> list[ParsedResultBlock]:
    stripped = strip_ansi(log_text)
    lines = stripped.splitlines()
    separator_indices = [
        index for index, line in enumerate(lines) if line.strip() == "-----------------"
    ]
    separator_set = set(separator_indices)
    result_blocks: list[ParsedResultBlock] = []

    for line_index, line in enumerate(lines):
        judge_match = JUDGE_SCORE_RE.match(line.strip())
        if judge_match is None:
            continue

        previous_separators = [
            separator for separator in separator_indices if separator < line_index
        ]
        if len(previous_separators) < 2:
            continue

        prediction_start = previous_separators[-2] + 1
        prediction_end = previous_separators[-1]
        prediction = "\n".join(lines[prediction_start:prediction_end]).strip() or None

        explanation_lines: list[str] = []
        for follow_index in range(line_index + 1, len(lines)):
            if follow_index in separator_set:
                break
            explanation_lines.append(lines[follow_index])

        explanation = None
        if explanation_lines:
            first_line = explanation_lines[0].strip()
            if first_line.startswith(JUDGE_EXPLANATION_PREFIX):
                explanation_lines[0] = first_line.split(JUDGE_EXPLANATION_PREFIX, 1)[1]
            explanation = "\n".join(line.rstrip() for line in explanation_lines).strip() or None

        judge_raw = judge_match.group("score").strip()
        judge_binary = None
        try:
            judge_binary = _normalize_score(judge_raw)
        except Exception:
            judge_binary = None

        result_blocks.append(
            ParsedResultBlock(
                prediction=prediction,
                judge_raw=judge_raw,
                judge_binary=judge_binary,
                judge_explanation=explanation,
            )
        )

    return result_blocks


def _normalize_judge_model_key(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if ":" in stripped:
        return stripped
    return f"google:{stripped}" if stripped.startswith("gemini-") else stripped


def _snapshot_average(snapshot: dict[str, Any] | None) -> float | None:
    if snapshot is None:
        return None
    return float(snapshot.get("average_health_score", 0.0))


def _snapshot_metrics(snapshot: dict[str, Any] | None) -> dict[str, float]:
    if snapshot is None:
        return {}
    return {
        str(metric_name): float(metric_value)
        for metric_name, metric_value in dict(snapshot.get("metric_scores") or {}).items()
    }


def _metric_deltas(
    baseline_scores: dict[str, float],
    run_scores: dict[str, float],
) -> dict[str, float]:
    metric_names = sorted(set(baseline_scores) | set(run_scores))
    return {
        metric_name: float(run_scores.get(metric_name, 0.0) - baseline_scores.get(metric_name, 0.0))
        for metric_name in metric_names
    }


def _list_all_snapshots(store: KGHealthSnapshotStore) -> list[dict[str, Any]]:
    return store.list_snapshots()


def _fallback_run_snapshot(
    store: KGHealthSnapshotStore,
    *,
    question: str,
    model_key: str | None,
    segment_timestamp: datetime | None,
) -> dict[str, Any] | None:
    candidates = [
        snapshot
        for snapshot in _list_all_snapshots(store)
        if snapshot.get("snapshot_kind") == "run"
    ]
    if not candidates:
        return None

    filtered = [
        snapshot
        for snapshot in candidates
        if (snapshot.get("metadata") or {}).get("question") == question
        and (model_key is None or (snapshot.get("metadata") or {}).get("model_key") == model_key)
    ]
    if not filtered:
        filtered = candidates

    if segment_timestamp is None:
        return filtered[-1] if filtered else None

    def distance(snapshot: dict[str, Any]) -> float:
        created_at = snapshot.get("created_at")
        if not created_at:
            return float("inf")
        parsed = datetime.fromisoformat(str(created_at))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return abs((parsed - segment_timestamp).total_seconds())

    return min(filtered, key=distance) if filtered else None


def _find_baseline_snapshot(
    store: KGHealthSnapshotStore,
    *,
    base_group_id: str | None,
    question: str,
    model_key: str | None,
    segment_timestamp: datetime | None,
) -> dict[str, Any] | None:
    if base_group_id:
        snapshot = store.get_baseline_snapshot(base_group_id)
        if snapshot is not None:
            return snapshot

    fallback_run_snapshot = _fallback_run_snapshot(
        store,
        question=question,
        model_key=model_key,
        segment_timestamp=segment_timestamp,
    )
    if fallback_run_snapshot is None:
        return None
    fallback_base_group_id = fallback_run_snapshot.get("base_group_id")
    if not fallback_base_group_id:
        return None
    return store.get_baseline_snapshot(str(fallback_base_group_id))


def _find_run_snapshot(
    store: KGHealthSnapshotStore,
    *,
    base_group_id: str | None,
    write_group_id: str | None,
    question: str,
    model_key: str | None,
    segment_timestamp: datetime | None,
) -> dict[str, Any] | None:
    if base_group_id and write_group_id:
        exact_matches = [
            snapshot
            for snapshot in store.list_snapshots(
                base_group_id=base_group_id,
                write_group_id=write_group_id,
            )
            if snapshot.get("snapshot_kind") == "run"
        ]
        if exact_matches:
            return exact_matches[-1]

    if base_group_id:
        base_matches = [
            snapshot
            for snapshot in store.list_snapshots(base_group_id=base_group_id)
            if snapshot.get("snapshot_kind") == "run"
        ]
        filtered = [
            snapshot
            for snapshot in base_matches
            if (snapshot.get("metadata") or {}).get("question") == question
            and (
                model_key is None or (snapshot.get("metadata") or {}).get("model_key") == model_key
            )
        ]
        if filtered:
            return filtered[-1]

    return _fallback_run_snapshot(
        store,
        question=question,
        model_key=model_key,
        segment_timestamp=segment_timestamp,
    )


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("run_log_path") or ""),
        str(row.get("dataset_variant") or ""),
        str(row.get("model_key") or ""),
        str(row.get("entry_id") or ""),
    )


def _artifact_quality(
    *,
    result_block: ParsedResultBlock | None,
    segment_prediction: str | None,
    baseline_snapshot: dict[str, Any] | None,
    run_snapshot: dict[str, Any] | None,
    writable: bool,
) -> str:
    reasons: list[str] = []
    if result_block is None:
        reasons.append("missing_result_block")
    if result_block is None and segment_prediction is not None:
        reasons.append("segment_prediction_fallback")
    if baseline_snapshot is None:
        reasons.append("missing_baseline_health")
    if writable and run_snapshot is None:
        reasons.append("missing_run_health")
    if not reasons:
        return "complete"
    return "partial:" + ",".join(reasons)


def parse_run_log(
    log_path: str | Path,
    *,
    store: KGHealthSnapshotStore | None = None,
) -> list[dict[str, Any]]:
    resolved_log_path = Path(log_path)
    log_text = resolved_log_path.read_text(encoding="utf-8", errors="ignore")
    stripped_log_text = strip_ansi(log_text)
    segments = split_execution_segments(stripped_log_text)
    if not segments:
        return []

    spec, entries = load_entries_for_log(log_path, entry_limit=len(segments))
    sidecar = trace_path(resolved_log_path)
    traces = read_traces(sidecar) if sidecar.exists() else None
    result_blocks = extract_result_blocks(stripped_log_text)
    health_store = store or KGHealthSnapshotStore()

    task_name = _parse_header_value(TASK_HEADER_RE, stripped_log_text) or spec.task_name
    model_key = _parse_header_value(MODEL_HEADER_RE, stripped_log_text)
    kg_model_key = _parse_header_value(KG_MODEL_HEADER_RE, stripped_log_text)
    judge_model_key = _normalize_judge_model_key(
        _parse_header_value(JUDGE_MODEL_HEADER_RE, stripped_log_text)
    )

    rows: list[dict[str, Any]] = []
    aligned_length = min(len(entries), len(segments))
    for index in range(aligned_length):
        entry = entries[index]
        segment = segments[index]
        result_block = result_blocks[index] if index < len(result_blocks) else None

        prediction = ""
        if result_block is not None and result_block.prediction is not None:
            prediction = result_block.prediction
        elif segment.prediction is not None:
            prediction = segment.prediction

        judge_raw = result_block.judge_raw if result_block is not None else None
        judge_binary = result_block.judge_binary if result_block is not None else None
        judge_explanation = result_block.judge_explanation if result_block is not None else None

        if judge_binary is None and judge_raw is not None:
            try:
                judge_binary = _normalize_score(judge_raw)
            except Exception:
                judge_binary = None

        baseline_snapshot = _find_baseline_snapshot(
            health_store,
            base_group_id=segment.base_group_id,
            question=entry.question,
            model_key=model_key,
            segment_timestamp=segment.timestamp,
        )
        run_snapshot = None
        if segment.write_group_id is not None:
            run_snapshot = _find_run_snapshot(
                health_store,
                base_group_id=segment.base_group_id,
                write_group_id=segment.write_group_id,
                question=entry.question,
                model_key=model_key,
                segment_timestamp=segment.timestamp,
            )

        baseline_average_health = _snapshot_average(baseline_snapshot)
        run_average_health = _snapshot_average(run_snapshot)
        baseline_metric_scores = _snapshot_metrics(baseline_snapshot)
        run_metric_scores = _snapshot_metrics(run_snapshot)

        if segment.write_group_id is None:
            run_average_health = baseline_average_health
            run_metric_scores = dict(baseline_metric_scores)

        average_health_delta = None
        if baseline_average_health is not None and run_average_health is not None:
            average_health_delta = float(run_average_health - baseline_average_health)
        elif segment.write_group_id is None:
            average_health_delta = 0.0

        metric_deltas = _metric_deltas(baseline_metric_scores, run_metric_scores)
        em = float(exact_match_score(prediction, entry.gold))
        f1, precision, recall = f1_score(prediction, entry.gold)

        variant = spec.variant
        if traces is not None and entry.entry_id not in traces:
            raise ValueError(f"Missing tool trajectory for {entry.entry_id!r} in {sidecar}")
        trace = traces[entry.entry_id] if traces is not None else None
        calls, gate_events = _trajectory_calls(segment, trace, gated=variant == "gatekg")
        if variant == "gatekg" and trace is not None:
            variant = normalize_variant_name(trace["selected_variant"])
        elif variant == "gatekg":
            marker = "GateKG selection event: "
            event = next(
                (
                    line.split(marker, 1)[1]
                    for line in segment.segment_text.splitlines()
                    if marker in line
                ),
                None,
            )
            if event is None:
                raise ValueError(f"Missing policy selection for {entry.entry_id} in {log_path}")
            variant = normalize_variant_name(json.loads(event)["selected_variant"])
        rows.append(
            {
                "runner": spec.variant,
                "run_log_path": str(resolved_log_path),
                "task_name": task_name,
                "dataset_variant": spec.dataset_variant,
                "entry_index": segment.entry_position - 1,
                "entry_id": entry.entry_id,
                "question": entry.question,
                "gold": entry.gold,
                "variant": variant,
                "model_key": model_key,
                "kg_model_key": kg_model_key,
                "judge_model_key": judge_model_key,
                "prediction": prediction,
                "judge_raw": judge_raw,
                "judge_binary": judge_binary,
                "judge_explanation": judge_explanation,
                "em": em,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "base_group_id": segment.base_group_id,
                "write_group_id": segment.write_group_id,
                "context_chunk_count": segment.context_chunk_count,
                "kg_context_chunk_count": segment.kg_context_chunk_count,
                "baseline_snapshot_id": (
                    baseline_snapshot.get("snapshot_id") if baseline_snapshot is not None else None
                ),
                "run_snapshot_id": (
                    run_snapshot.get("snapshot_id") if run_snapshot is not None else None
                ),
                "baseline_average_health": baseline_average_health,
                "run_average_health": run_average_health,
                "average_health_delta": average_health_delta,
                "baseline_metric_scores": baseline_metric_scores,
                "baseline_metric_raw_values": (baseline_snapshot or {}).get(
                    "metric_raw_values", {}
                ),
                "run_metric_scores": run_metric_scores,
                "metric_deltas": metric_deltas,
                "tool_calls": calls,
                "gate_events": gate_events,
                "tool_call_count": len(calls),
                "mutating_tool_call_count": sum(
                    1 for tool_call in calls if tool_call.get("is_mutating")
                ),
                "artifact_quality": _artifact_quality(
                    result_block=result_block,
                    segment_prediction=segment.prediction,
                    baseline_snapshot=baseline_snapshot,
                    run_snapshot=run_snapshot,
                    writable=segment.write_group_id is not None,
                ),
                "parser_version": GATEKG_DATASET_PARSER_VERSION,
                "cache_model_key": segment.cache_model_key,
                "segment_timestamp": (
                    segment.timestamp.isoformat() if segment.timestamp is not None else None
                ),
            }
        )

    return rows


def find_training_logs(log_root: str | Path) -> list[Path]:
    resolved_root = Path(log_root)
    matched_paths: list[Path] = []
    for log_path in sorted(resolved_root.rglob("run_*.log")):
        try:
            spec = resolve_experiment_spec(log_path)
        except FileNotFoundError:
            continue
        if spec.variant in (*GATEKG_VARIANTS, "gatekg"):
            matched_paths.append(log_path)
    return matched_paths


def build_dataset_rows(log_root: str | Path) -> list[dict[str, Any]]:
    store = KGHealthSnapshotStore()
    rows: list[dict[str, Any]] = []
    for log_path in find_training_logs(log_root):
        rows.extend(parse_run_log(log_path, store=store))
    return rows


def _count_by_key(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def summarize_dataset(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized_rows = list(rows)
    summary: dict[str, Any] = {
        "row_count": len(materialized_rows),
        "run_count": len({str(row.get("run_log_path") or "") for row in materialized_rows}),
        "variant_counts": _count_by_key(materialized_rows, "variant"),
        "dataset_variant_counts": _count_by_key(materialized_rows, "dataset_variant"),
        "model_key_counts": _count_by_key(materialized_rows, "model_key"),
        "artifact_quality_counts": _count_by_key(materialized_rows, "artifact_quality"),
    }
    if not materialized_rows:
        return summary

    summary["avg_tool_call_count"] = mean(
        float(row.get("tool_call_count") or 0.0) for row in materialized_rows
    )
    summary["avg_mutating_tool_call_count"] = mean(
        float(row.get("mutating_tool_call_count") or 0.0) for row in materialized_rows
    )
    if "composite_reward" in materialized_rows[0]:
        summary["reward_mean"] = mean(
            float(row.get("composite_reward") or 0.0) for row in materialized_rows
        )
    return summary


def write_dataset(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    summary_path: str | Path,
) -> None:
    resolved_output_path = Path(output_path)
    resolved_summary_path = Path(summary_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with private_output(resolved_output_path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_summary_path.write_text(
        json.dumps(summarize_dataset(rows), ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_dataset_rows(dataset_path: str | Path) -> list[dict[str, Any]]:
    resolved_dataset_path = Path(dataset_path)
    rows: list[dict[str, Any]] = []
    with resolved_dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def group_rows_by_identity(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("dataset_variant") or ""),
            str(row.get("model_key") or ""),
            str(row.get("entry_id") or ""),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def efficiency_raw(row: dict[str, Any]) -> float:
    return -(
        float(row.get("tool_call_count") or 0.0)
        + (0.1 * float(row.get("mutating_tool_call_count") or 0.0))
    )


def _cohort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset_variant") or ""),
        str(row.get("model_key") or ""),
        str(row.get("variant") or ""),
    )


def _normalize_by_cohort(
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
) -> dict[tuple[str, str, str, str], float]:
    values_by_cohort: dict[tuple[str, str, str], list[float]] = {}
    value_by_row: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        cohort_key = _cohort_key(row)
        value = float(row.get(value_key) or 0.0)
        values_by_cohort.setdefault(cohort_key, []).append(value)
        value_by_row[_row_identity(row)] = value

    normalized: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        cohort_key = _cohort_key(row)
        values = values_by_cohort[cohort_key]
        minimum = min(values)
        maximum = max(values)
        row_key = _row_identity(row)
        raw_value = value_by_row[row_key]
        if maximum == minimum:
            normalized[row_key] = 0.5
        else:
            normalized[row_key] = float((raw_value - minimum) / (maximum - minimum))
    return normalized


def annotate_rewards(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized_rows = [dict(row) for row in rows]
    for row in materialized_rows:
        row["average_health_delta"] = float(row.get("average_health_delta") or 0.0)
        if (
            row.get("judge_binary") not in (0, 1, "0", "1")
            or row.get("prediction") == "Server error"
            or str(row.get("judge_explanation") or "").startswith(
                "Judge request failed after retries."
            )
        ):
            raise ValueError(
                f"Missing or failed evaluation for entry {row.get('entry_id')!r} in {row.get('run_log_path')}"
            )
        row["judge_binary"] = int(row["judge_binary"])
        row["efficiency_raw"] = efficiency_raw(row)

    normalized_health = _normalize_by_cohort(
        materialized_rows,
        value_key="average_health_delta",
    )
    normalized_efficiency = _normalize_by_cohort(
        materialized_rows,
        value_key="efficiency_raw",
    )

    for row in materialized_rows:
        row_key = _row_identity(row)
        row["normalized_average_health_delta"] = normalized_health[row_key]
        row["normalized_efficiency"] = normalized_efficiency[row_key]
        row["composite_reward"] = (
            (0.45 * float(row.get("judge_binary") or 0.0))
            + (0.15 * float(row.get("f1") or 0.0))
            + (0.30 * float(row["normalized_average_health_delta"]))
            + (0.10 * float(row["normalized_efficiency"]))
        )
        row["reward"] = row["composite_reward"]
    return materialized_rows


def select_best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row.get("composite_reward", row.get("reward") or 0.0)),
            float(row.get("judge_binary") or 0.0),
            float(row.get("f1") or 0.0),
            float(row.get("average_health_delta") or 0.0),
            -float(row.get("tool_call_count") or 0.0),
            str(row.get("variant") or ""),
        ),
    )


def build_selector_targets(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated_rows = annotate_rewards(rows)
    targets: list[dict[str, Any]] = []
    for group_rows in group_rows_by_identity(annotated_rows).values():
        best_row = dict(select_best_row(group_rows))
        best_row["target_variant"] = best_row["variant"]
        best_row["sibling_count"] = len(group_rows)
        targets.append(best_row)
    return targets


def common_prefix_length(
    left_tools: Sequence[dict[str, Any]],
    right_tools: Sequence[dict[str, Any]],
) -> int:
    prefix_length = 0
    for left_call, right_call in zip(left_tools, right_tools, strict=False):
        if left_call.get("tool") != right_call.get("tool"):
            break
        prefix_length += 1
    return prefix_length


def normalize_trajectory_state(value: Any) -> int:
    state = int(value)
    if state not in {-1, 0, 1}:
        raise ValueError(f"Unsupported trajectory state: {value}")
    return state


def final_correct_state(row: dict[str, Any]) -> int:
    if int(row.get("judge_binary") or 0) == 1:
        return 1
    return -1


def right_fill_trajectory_states(
    raw_states: Sequence[int],
    *,
    fallback_state: int,
) -> list[int]:
    if fallback_state not in {-1, 1}:
        raise ValueError("Trajectory fallback state must be -1 or 1.")

    resolved_states: list[int] = [0 for _ in raw_states]
    next_non_zero: int | None = None
    for index in range(len(raw_states) - 1, -1, -1):
        state = normalize_trajectory_state(raw_states[index])
        if state == 0:
            resolved_states[index] = next_non_zero if next_non_zero is not None else fallback_state
        else:
            resolved_states[index] = state
            next_non_zero = state
    return resolved_states


def fill_row_trajectory_states(row: dict[str, Any]) -> dict[str, Any]:
    materialized_row = dict(row)
    tool_calls = [dict(tool_call) for tool_call in materialized_row.get("tool_calls") or []]
    if not tool_calls:
        materialized_row["tool_calls"] = tool_calls
        return materialized_row

    raw_states: list[int] = []
    for tool_call in tool_calls:
        if tool_call.get("trajectory_label_fallback"):
            raise ValueError("Failed trajectory judgment requires relabeling")
        state_value = tool_call.get("trajectory_state_raw", tool_call.get("trajectory_state"))
        if state_value is None:
            raise ValueError(
                "Cannot fill trajectory states because a tool call is missing "
                "`trajectory_state_raw`."
            )
        raw_states.append(normalize_trajectory_state(state_value))

    filled_states = right_fill_trajectory_states(
        raw_states,
        fallback_state=final_correct_state(materialized_row),
    )
    for tool_call, raw_state, filled_state in zip(
        tool_calls, raw_states, filled_states, strict=True
    ):
        tool_call["trajectory_state_raw"] = raw_state
        tool_call["trajectory_state"] = filled_state
    materialized_row["tool_calls"] = tool_calls
    return materialized_row


def _call_features(row: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    counts: dict[str, int] = {}
    mutations = 0
    for index, call in enumerate(row["tool_calls"]):
        tool = call["tool"]
        counters = {
            "call_index": index,
            "prior_call_count": index,
            "prior_mutating_call_count": mutations,
            "prior_same_tool_count": counts.get(tool, 0),
            "prior_unique_tool_count": len(counts),
            "writable_flag": int(writable_variant(row["variant"])),
        }
        event = call.get("gate_event")
        if event is not None:
            if event["tool"] != tool or event["decision"] != "allow":
                raise ValueError("Executed tool does not match its gate event")
            counters = {key: event[key] for key in counters}
        elif row.get("runner") == "gatekg":
            raise ValueError("Policy row lacks attempt counters. Rebuild it from the run log.")
        features = {
            "run_log_path": row["run_log_path"],
            "dataset_variant": row["dataset_variant"],
            "model_key": row["model_key"],
            "entry_id": row["entry_id"],
            "question": row["question"],
            "variant": row["variant"],
            "tool_name": tool,
            **counters,
            "baseline_average_health": row.get("baseline_average_health"),
            "baseline_metric_scores": row.get("baseline_metric_scores") or {},
            "baseline_metric_raw_values": row.get("baseline_metric_raw_values") or {},
            "context_chunk_count": row.get("context_chunk_count"),
            "kg_context_chunk_count": row.get("kg_context_chunk_count"),
            "composite_reward": row.get("composite_reward", row.get("reward")),
        }
        yield call, features
        counts[tool] = counts.get(tool, 0) + 1
        mutations += int(call.get("is_mutating", tool_is_mutating(tool)))


def build_gate_examples(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated_rows = annotate_rewards(rows)
    examples: list[dict[str, Any]] = []
    for group_rows in group_rows_by_identity(annotated_rows).values():
        if len(group_rows) < 2:
            continue

        best_row = select_best_row(group_rows)
        best_tools = list(best_row.get("tool_calls") or [])

        for _, features in _call_features(best_row):
            examples.append({**features, "label": 1})

        for losing_row in group_rows:
            if losing_row is best_row:
                continue
            divergence = common_prefix_length(best_tools, losing_row["tool_calls"])
            for position, (_, features) in enumerate(_call_features(losing_row)):
                if position >= divergence:
                    examples.append({**features, "label": 0})

    return examples


def build_trajectory_gate_examples(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated_rows = annotate_rewards(rows)
    examples: list[dict[str, Any]] = []
    missing_label_rows = 0
    for row in annotated_rows:
        tool_calls = list(row.get("tool_calls") or [])
        if not tool_calls:
            continue
        try:
            labeled_row = fill_row_trajectory_states(row)
        except ValueError:
            missing_label_rows += 1
            continue

        for tool_call, features in _call_features(labeled_row):
            trajectory_state = normalize_trajectory_state(tool_call["trajectory_state"])
            examples.append(
                {
                    **features,
                    "label": int(trajectory_state == 1),
                    "trajectory_state_raw": normalize_trajectory_state(
                        tool_call["trajectory_state_raw"]
                    ),
                    "trajectory_state": trajectory_state,
                    "trajectory_label_source": tool_call.get(
                        "trajectory_label_source",
                        "gatekg_llm_judge",
                    ),
                }
            )

    if missing_label_rows and not examples:
        raise ValueError(
            "No trajectory gate examples could be built because tool calls are "
            "missing trajectory labels."
        )
    return examples


def collect_tool_divergences(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated_rows = annotate_rewards(rows)
    counts: dict[tuple[str, str], int] = {}
    for group_rows in group_rows_by_identity(annotated_rows).values():
        if len(group_rows) < 2:
            continue
        best_row = select_best_row(group_rows)
        best_tools = list(best_row.get("tool_calls") or [])
        for losing_row in group_rows:
            if losing_row is best_row:
                continue
            losing_tools = list(losing_row.get("tool_calls") or [])
            divergence_index = common_prefix_length(best_tools, losing_tools)
            winning_tool = (
                str(best_tools[divergence_index].get("tool"))
                if divergence_index < len(best_tools)
                else "__end__"
            )
            losing_tool = (
                str(losing_tools[divergence_index].get("tool"))
                if divergence_index < len(losing_tools)
                else "__end__"
            )
            counts[(winning_tool, losing_tool)] = counts.get((winning_tool, losing_tool), 0) + 1

    return [
        {
            "winning_tool": winning_tool,
            "losing_tool": losing_tool,
            "count": count,
        }
        for (winning_tool, losing_tool), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]


def summarize_reward_distribution(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
) -> dict[str, dict[str, float]]:
    annotated_rows = annotate_rewards(rows)
    grouped: dict[str, list[float]] = {}
    for row in annotated_rows:
        grouped.setdefault(str(row.get(group_key) or ""), []).append(
            float(row.get("composite_reward") or 0.0)
        )
    return {
        key: {
            "count": float(len(values)),
            "mean": float(mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
        for key, values in sorted(grouped.items())
    }
