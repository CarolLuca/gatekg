"""Label tool trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Protocol

from google.genai import errors, types

from gatekg.paths import runtime_paths

PROJECT_ROOT = runtime_paths().root

from gatekg.api.google_client import build_google_genai_client
from gatekg.training.dataset import (
    fill_row_trajectory_states,
    normalize_trajectory_state,
    read_dataset_rows,
    write_dataset,
)
from gatekg.utils.config import Config, load_config
from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import fit_judge_context
from gatekg.utils.trajectory import private_output

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "gatekg" / "runs.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "gatekg" / "runs.labeled.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "gatekg" / "runs.labeled.summary.json"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "gatekg" / "trajectory_label_cache.json"
DEFAULT_JUDGE_RETRIES = 6
DEFAULT_JUDGE_RETRY_DELAY_SECONDS = 30.0
DEFAULT_PROGRESS_INTERVAL = 25
TRAJECTORY_LABEL_VERSION = "gatekg-trajectory-v3"
TRAJECTORY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {
            "type": "integer",
            "enum": [-1, 0, 1],
        },
        "explanation": {
            "type": "string",
        },
    },
    "required": ["state", "explanation"],
}


@dataclass(frozen=True)
class TrajectoryLabel:
    state: int
    explanation: str
    model_name: str
    fallback: bool = False
    context_chars_omitted: int = 0


class TrajectoryJudge(Protocol):
    model_name: str

    def judge_tool_call(
        self,
        *,
        row: dict[str, Any],
        tool_call: dict[str, Any],
        call_index: int,
    ) -> TrajectoryLabel: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-label GateKG per-tool-call trajectory states.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument(
        "--row-limit",
        type=int,
        default=0,
        help="Optional maximum number of rows to label.",
    )
    parser.add_argument(
        "--call-limit",
        type=int,
        default=0,
        help="Optional maximum number of unlabeled tool calls to judge.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Relabel calls even when trajectory_state_raw already exists.",
    )
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=DEFAULT_JUDGE_RETRIES,
        help="Maximum attempts per uncached judge request.",
    )
    parser.add_argument(
        "--judge-retry-delay",
        type=float,
        default=DEFAULT_JUDGE_RETRY_DELAY_SECONDS,
        help="Initial seconds to wait before retrying failed judge requests.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print progress after this many processed tool calls.",
    )
    return parser.parse_args(argv)


def _prompt_for_call(
    *,
    row: dict[str, Any],
    tool_call: dict[str, Any],
    call_index: int,
) -> str:
    event = tool_call.get("gate_event") or {}
    call_index = event.get("call_index", call_index)
    if "args" in tool_call:
        evidence = json.dumps(
            {
                key: tool_call[key]
                for key in ("args", "kwargs", "result", "status", "error_type")
                if key in tool_call
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        evidence = f"Legacy console summaries only. Args: {tool_call.get('args_summary', '')}; kwargs: {tool_call.get('kwargs_summary', '')}"
    return (
        "You are labeling a tool-use trajectory for a QA model.\n"
        "Return only JSON with keys `state` and `explanation`.\n"
        "`state` must be exactly one of -1, 0, 1.\n"
        "- 1 means this tool call or nearby reasoning is moving toward the gold answer.\n"
        "- 0 means neutral: it does not clearly support a right or wrong answer.\n"
        "- -1 means this tool call or nearby reasoning is moving toward a wrong answer.\n"
        "Judge the local trajectory state, not only final correctness. Keep the explanation short.\n\n"
        f"Question:\n{row.get('question') or ''}\n\n"
        f"Gold answer:\n{row.get('gold') or ''}\n\n"
        f"Final prediction:\n{row.get('prediction') or ''}\n\n"
        f"Final judge score:\n{row.get('judge_binary')}\n\n"
        f"Call index:\n{call_index}\n"
        f"Tool:\n{tool_call.get('tool') or ''}\n"
        f"Tool evidence:\n{evidence}\n\n"
        f"Nearby execution context:\n{tool_call.get('nearby_context') or ''}\n"
    )


def _is_retryable_judge_error(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        status_code = getattr(exc, "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True
        message = str(exc).upper()
        return "RESOURCE_EXHAUSTED" in message or "RATE_LIMIT" in message
    return False


class LLMTrajectoryStateJudge:
    def __init__(
        self,
        *,
        config: Config | None = None,
        max_retries: int = DEFAULT_JUDGE_RETRIES,
        retry_delay_seconds: float = DEFAULT_JUDGE_RETRY_DELAY_SECONDS,
    ) -> None:
        self.config = config or load_config()
        self.model_name = self.config.google_judge_model_name
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.client = build_google_genai_client(
            config=self.config,
            timeout_seconds=self.config.llm_runtime.timeout_seconds,
        )

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
            seed=0,
            response_mime_type="application/json",
            response_json_schema=TRAJECTORY_RESPONSE_SCHEMA,
        )

    def judge_tool_call(
        self,
        *,
        row: dict[str, Any],
        tool_call: dict[str, Any],
        call_index: int,
    ) -> TrajectoryLabel:
        for attempt in range(1, self.max_retries + 1):
            try:
                with LLM_REQUEST_LIMITER.sync_slot():
                    prompt, omitted = fit_judge_context(
                        client=self.client,
                        model=self.model_name,
                        limit=self.config.input_token_cap_for("google"),
                        context=str(tool_call.get("nearby_context") or ""),
                        render=lambda text: _prompt_for_call(
                            row=row,
                            tool_call={**tool_call, "nearby_context": text},
                            call_index=call_index,
                        ),
                    )
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=self._config(),
                    )
                if response.text is None or not str(response.text).strip():
                    raise ValueError("Trajectory judge returned an empty response.")
                payload = json.loads(response.text)
                return TrajectoryLabel(
                    state=normalize_trajectory_state(payload["state"]),
                    explanation=str(payload.get("explanation") or "").strip()
                    or "No explanation provided.",
                    model_name=self.model_name,
                    context_chars_omitted=omitted,
                )
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_delay_seconds * attempt
                retry_kind = (
                    "retryable API error"
                    if _is_retryable_judge_error(exc)
                    else "invalid/empty judge response"
                )
                print(
                    f"Trajectory judge {retry_kind}; retrying "
                    f"attempt={attempt + 1}/{self.max_retries} "
                    f"delay={delay:.1f}s error={exc}",
                    flush=True,
                )
                time.sleep(delay)

        raise AssertionError("Retry loop exhausted")


def _cache_key(
    *,
    row: dict[str, Any],
    tool_call: dict[str, Any],
    call_index: int,
) -> str:
    payload = {
        "version": TRAJECTORY_LABEL_VERSION,
        "run_log_path": row.get("run_log_path"),
        "dataset_variant": row.get("dataset_variant"),
        "model_key": row.get("model_key"),
        "entry_id": row.get("entry_id"),
        "prompt": _prompt_for_call(row=row, tool_call=tool_call, call_index=call_index),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return blake2b(encoded, digest_size=16).hexdigest()


def _read_cache(cache_path: str | Path) -> dict[str, dict[str, Any]]:
    resolved_cache_path = Path(cache_path)
    if not resolved_cache_path.is_file():
        return {}
    return json.loads(resolved_cache_path.read_text(encoding="utf-8"))


def _write_cache(cache_path: str | Path, cache: dict[str, dict[str, Any]]) -> None:
    resolved_cache_path = Path(cache_path)
    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with private_output(resolved_cache_path) as handle:
        json.dump(cache, handle, ensure_ascii=True, indent=2, sort_keys=True)


def _progress_bar(current: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    completed = min(width, int(width * min(current, total) / total))
    return "[" + ("#" * completed) + ("-" * (width - completed)) + "]"


def _print_progress(
    *,
    current: int,
    total: int,
    judged: int,
    cached: int,
    pre_labeled: int,
    skipped: int,
    final: bool = False,
) -> None:
    percent = (100.0 * current / total) if total else 100.0
    message = (
        "Trajectory labels "
        f"{_progress_bar(current, total)} "
        f"{current}/{total} ({percent:.1f}%) "
        f"judged={judged} cached={cached} "
        f"pre_labeled={pre_labeled} skipped={skipped}"
    )
    if sys.stdout.isatty() and not final:
        print("\r" + message, end="", flush=True)
    else:
        print(message, flush=True)
    if sys.stdout.isatty() and final:
        print(flush=True)


def label_rows(
    rows: Sequence[dict[str, Any]],
    *,
    judge: TrajectoryJudge,
    cache_path: str | Path,
    row_limit: int = 0,
    call_limit: int = 0,
    force: bool = False,
    progress_interval: int = 25,
) -> list[dict[str, Any]]:
    cache = _read_cache(cache_path)
    labeled_rows: list[dict[str, Any]] = []
    judged_call_count = 0
    cached_call_count = 0
    pre_labeled_call_count = 0
    skipped_call_count = 0
    processed_call_count = 0
    selected_rows = list(rows[:row_limit] if row_limit > 0 else rows)
    total_call_count = sum(len(row.get("tool_calls") or []) for row in selected_rows)

    print(
        "GateKG trajectory labeling: "
        f"rows={len(selected_rows)} tool_calls={total_call_count} "
        f"cache_entries={len(cache)} judge_model={judge.model_name}",
        flush=True,
    )
    _print_progress(
        current=0,
        total=total_call_count,
        judged=judged_call_count,
        cached=cached_call_count,
        pre_labeled=pre_labeled_call_count,
        skipped=skipped_call_count,
    )

    for row in selected_rows:
        labeled_row = dict(row)
        tool_calls = [dict(tool_call) for tool_call in row.get("tool_calls") or []]
        for call_index, tool_call in enumerate(tool_calls):
            processed_call_count += 1
            key = _cache_key(row=row, tool_call=tool_call, call_index=call_index)
            if (
                not force
                and tool_call.get("trajectory_state_raw") is not None
                and not tool_call.get("trajectory_label_fallback")
                and tool_call.get("trajectory_label_model") == judge.model_name
                and tool_call.get("trajectory_label_cache_key") == key
                and tool_call.get("trajectory_label_version") == TRAJECTORY_LABEL_VERSION
            ):
                pre_labeled_call_count += 1
                if progress_interval > 0 and processed_call_count % progress_interval == 0:
                    _print_progress(
                        current=processed_call_count,
                        total=total_call_count,
                        judged=judged_call_count,
                        cached=cached_call_count,
                        pre_labeled=pre_labeled_call_count,
                        skipped=skipped_call_count,
                    )
                continue

            cached_label = cache.get(key)
            if (
                force
                or cached_label is None
                or cached_label.get("fallback")
                or cached_label.get("model_name") != judge.model_name
            ):
                if call_limit > 0 and judged_call_count >= call_limit:
                    for field in list(tool_call):
                        if field.startswith("trajectory_"):
                            del tool_call[field]
                    skipped_call_count += 1
                    if progress_interval > 0 and processed_call_count % progress_interval == 0:
                        _print_progress(
                            current=processed_call_count,
                            total=total_call_count,
                            judged=judged_call_count,
                            cached=cached_call_count,
                            pre_labeled=pre_labeled_call_count,
                            skipped=skipped_call_count,
                        )
                    continue
                if sys.stdout.isatty():
                    print(
                        "\r"
                        f"Judging call {processed_call_count}/{total_call_count}: "
                        f"entry_id={row.get('entry_id')} tool={tool_call.get('tool')}" + " " * 12,
                        end="",
                        flush=True,
                    )
                label = judge.judge_tool_call(row=row, tool_call=tool_call, call_index=call_index)
                if label.fallback:
                    raise ValueError("Trajectory judge returned an infrastructure failure")
                judged_call_count += 1
                cached_label = {
                    "state": label.state,
                    "explanation": label.explanation,
                    "model_name": label.model_name,
                    "fallback": False,
                    "context_chars_omitted": label.context_chars_omitted,
                    "version": TRAJECTORY_LABEL_VERSION,
                }
                cache[key] = cached_label
                _write_cache(cache_path, cache)
            else:
                cached_call_count += 1

            tool_call["trajectory_state_raw"] = normalize_trajectory_state(cached_label["state"])
            tool_call["trajectory_label_explanation"] = str(cached_label.get("explanation") or "")
            tool_call["trajectory_label_model"] = str(
                cached_label.get("model_name") or judge.model_name
            )
            tool_call["trajectory_label_cache_key"] = key
            tool_call["trajectory_label_version"] = TRAJECTORY_LABEL_VERSION
            tool_call["trajectory_label_source"] = "gatekg_llm_judge"
            tool_call["trajectory_context_chars_omitted"] = cached_label.get(
                "context_chars_omitted", 0
            )
            tool_call["trajectory_label_fallback"] = bool(cached_label.get("fallback", False))
            if progress_interval > 0 and processed_call_count % progress_interval == 0:
                _print_progress(
                    current=processed_call_count,
                    total=total_call_count,
                    judged=judged_call_count,
                    cached=cached_call_count,
                    pre_labeled=pre_labeled_call_count,
                    skipped=skipped_call_count,
                )

        labeled_row["tool_calls"] = tool_calls
        if tool_calls and all(
            tool_call.get("trajectory_state_raw") is not None for tool_call in tool_calls
        ):
            labeled_row = fill_row_trajectory_states(labeled_row)
        labeled_rows.append(labeled_row)

    _write_cache(cache_path, cache)
    _print_progress(
        current=processed_call_count,
        total=total_call_count,
        judged=judged_call_count,
        cached=cached_call_count,
        pre_labeled=pre_labeled_call_count,
        skipped=skipped_call_count,
        final=True,
    )
    print(f"Trajectory cache written to {cache_path}", flush=True)
    return labeled_rows


def label_dataset(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    cache_path: str | Path,
    judge: TrajectoryJudge | None = None,
    row_limit: int = 0,
    call_limit: int = 0,
    force: bool = False,
    judge_retries: int = DEFAULT_JUDGE_RETRIES,
    judge_retry_delay: float = DEFAULT_JUDGE_RETRY_DELAY_SECONDS,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> list[dict[str, Any]]:
    rows = read_dataset_rows(dataset_path)
    labeled_rows = label_rows(
        rows,
        judge=judge
        or LLMTrajectoryStateJudge(
            max_retries=judge_retries,
            retry_delay_seconds=judge_retry_delay,
        ),
        cache_path=cache_path,
        row_limit=row_limit,
        call_limit=call_limit,
        force=force,
        progress_interval=progress_interval,
    )
    print(f"Writing labeled dataset to {output_path}", flush=True)
    write_dataset(labeled_rows, output_path=output_path, summary_path=summary_path)
    print(f"Writing labeled summary to {summary_path}", flush=True)
    return labeled_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labeled_rows = label_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        summary_path=args.summary,
        cache_path=args.cache,
        row_limit=args.row_limit,
        call_limit=args.call_limit,
        force=args.force,
        judge_retries=args.judge_retries,
        judge_retry_delay=args.judge_retry_delay,
        progress_interval=args.progress_interval,
    )
    print(f"Wrote {len(labeled_rows)} GateKG labeled rows to {args.output}")
    print(f"Wrote trajectory cache to {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
