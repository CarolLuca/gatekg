"""Inspect training datasets and run traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from gatekg.paths import runtime_paths

PROJECT_ROOT = runtime_paths().root

from gatekg.training.dataset import (
    annotate_rewards,
    build_selector_targets,
    collect_tool_divergences,
    group_rows_by_identity,
    parse_run_log,
    read_dataset_rows,
    select_best_row,
    strip_ansi,
    summarize_dataset,
    summarize_reward_distribution,
)

DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "gatekg" / "runs.jsonl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "artifacts" / "gatekg" / "reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret GateKG datasets or run logs.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Interpret the reference dataset.")
    dataset_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Canonical GateKG dataset JSONL path.",
    )
    dataset_parser.add_argument(
        "--output-prefix",
        default="",
        help="Report output prefix; writes <prefix>.json and <prefix>.md.",
    )

    run_parser = subparsers.add_parser("run", help="Interpret one GateKG run log.")
    run_parser.add_argument("--log", required=True, help="GateKG experiment log path.")
    run_parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Reference GateKG dataset JSONL path used for sibling comparisons.",
    )
    run_parser.add_argument(
        "--output-prefix",
        default="",
        help="Report output prefix; writes <prefix>.json and <prefix>.md.",
    )

    return parser.parse_args(argv)


def _resolve_output_prefix(mode: str, explicit_prefix: str, target_name: str) -> Path:
    if explicit_prefix:
        return Path(explicit_prefix)
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_REPORT_DIR / f"{mode}_{target_name}"


def _write_reports(prefix: Path, payload: dict, markdown: str) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")


def interpret_dataset_rows(rows: list[dict]) -> dict:
    annotated_rows = annotate_rewards(rows)
    selector_targets = build_selector_targets(annotated_rows)
    selector_target_distribution = Counter(
        str(row.get("target_variant") or "") for row in selector_targets
    )

    health_by_variant: dict[str, list[float]] = {}
    for row in annotated_rows:
        health_by_variant.setdefault(str(row.get("variant") or ""), []).append(
            float(row.get("average_health_delta") or 0.0)
        )

    return {
        "coverage": summarize_dataset(annotated_rows),
        "reward_by_variant": summarize_reward_distribution(
            annotated_rows,
            group_key="variant",
        ),
        "reward_by_dataset_variant": summarize_reward_distribution(
            annotated_rows,
            group_key="dataset_variant",
        ),
        "selector_target_distribution": dict(sorted(selector_target_distribution.items())),
        "tool_divergences": collect_tool_divergences(annotated_rows)[:20],
        "health_delta_summary": {
            variant: {
                "count": len(values),
                "mean": float(mean(values)) if values else 0.0,
                "min": float(min(values)) if values else 0.0,
                "max": float(max(values)) if values else 0.0,
            }
            for variant, values in sorted(health_by_variant.items())
        },
    }


def _dataset_markdown(payload: dict) -> str:
    coverage = payload["coverage"]
    lines = [
        "# GateKG Dataset Interpretation",
        "",
        f"- Rows: {coverage['row_count']}",
        f"- Runs: {coverage['run_count']}",
        f"- Variants: {coverage['variant_counts']}",
        f"- Dataset sizes: {coverage['dataset_variant_counts']}",
        "",
        "## Selector Targets",
        "",
    ]
    for variant, count in payload["selector_target_distribution"].items():
        lines.append(f"- {variant}: {count}")
    lines.extend(["", "## Top Divergences", ""])
    for item in payload["tool_divergences"][:10]:
        lines.append(
            f"- winning `{item['winning_tool']}` vs losing `{item['losing_tool']}`: {item['count']}"
        )
    return "\n".join(lines)


def _parse_policy_events(log_path: str | Path) -> dict[int, dict[str, list[dict] | dict | None]]:
    log_text = strip_ansi(Path(log_path).read_text(encoding="utf-8", errors="ignore"))
    current_entry_index: int | None = None
    events: dict[int, dict[str, list[dict] | dict | None]] = {}

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("Testing entry "):
            header = line.split()
            entry_position = int(header[2].split("/")[0])
            current_entry_index = entry_position - 1
            events.setdefault(current_entry_index, {"selection": None, "gate": [], "summary": None})
            continue

        if current_entry_index is None:
            continue

        if "GateKG selection event:" in line:
            payload = json.loads(line.split("GateKG selection event:", 1)[1].strip())
            events[current_entry_index]["selection"] = payload
        elif "GateKG gate event:" in line:
            payload = json.loads(line.split("GateKG gate event:", 1)[1].strip())
            gate_events = events[current_entry_index]["gate"]
            assert isinstance(gate_events, list)
            gate_events.append(payload)
        elif "GateKG run summary event:" in line:
            payload = json.loads(line.split("GateKG run summary event:", 1)[1].strip())
            events[current_entry_index]["summary"] = payload

    return events


def _reference_best_row(
    reference_rows: list[dict],
    *,
    dataset_variant: str,
    model_key: str | None,
    entry_id: str,
) -> dict | None:
    exact_group = group_rows_by_identity(reference_rows).get(
        (dataset_variant, str(model_key), entry_id)
    )
    if exact_group:
        return select_best_row(exact_group)

    fallback_rows = [
        row
        for row in reference_rows
        if str(row.get("dataset_variant") or "") == dataset_variant
        and str(row.get("entry_id") or "") == entry_id
    ]
    if not fallback_rows:
        return None
    return select_best_row(fallback_rows)


def interpret_run_log(log_path: str | Path, reference_rows: list[dict]) -> dict:
    annotated_run_rows = annotate_rewards(parse_run_log(log_path))
    policy_events = _parse_policy_events(log_path)

    per_question: list[dict] = []
    decision_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    selected_variant_counter: Counter[str] = Counter()

    for row in annotated_run_rows:
        entry_index = int(row.get("entry_index") or 0)
        entry_events = policy_events.get(
            entry_index, {"selection": None, "gate": [], "summary": None}
        )
        selection_event = entry_events.get("selection")
        gate_events = entry_events.get("gate") or []
        summary_event = entry_events.get("summary")

        selected_variant = None
        if isinstance(selection_event, dict):
            selected_variant = selection_event.get("selected_variant")
            if selected_variant is not None:
                selected_variant_counter[str(selected_variant)] += 1

        for event in gate_events:
            decision_counter[str(event.get("decision") or "")] += 1
            tool_counter[str(event.get("tool") or "")] += 1

        reference_best = _reference_best_row(
            reference_rows,
            dataset_variant=str(row.get("dataset_variant") or ""),
            model_key=row.get("model_key"),
            entry_id=str(row.get("entry_id") or ""),
        )

        per_question.append(
            {
                "entry_id": row.get("entry_id"),
                "entry_index": entry_index,
                "question": row.get("question"),
                "selected_variant": selected_variant,
                "prediction": row.get("prediction"),
                "judge_binary": row.get("judge_binary"),
                "f1": row.get("f1"),
                "average_health_delta": row.get("average_health_delta"),
                "tool_call_count": row.get("tool_call_count"),
                "mutating_tool_call_count": row.get("mutating_tool_call_count"),
                "gate_events": gate_events,
                "selection_event": selection_event,
                "summary_event": summary_event,
                "reference_best": (
                    {
                        "variant": reference_best.get("variant"),
                        "reward": reference_best.get("composite_reward"),
                        "judge_binary": reference_best.get("judge_binary"),
                        "f1": reference_best.get("f1"),
                        "average_health_delta": reference_best.get("average_health_delta"),
                        "tool_call_count": reference_best.get("tool_call_count"),
                    }
                    if reference_best is not None
                    else None
                ),
            }
        )

    return {
        "log_path": str(log_path),
        "entry_count": len(per_question),
        "selected_variant_counts": dict(sorted(selected_variant_counter.items())),
        "gate_decision_counts": dict(sorted(decision_counter.items())),
        "tool_frequency": dict(tool_counter.most_common(20)),
        "aggregate": {
            "judge_mean": (
                float(mean(float(row.get("judge_binary") or 0.0) for row in annotated_run_rows))
                if annotated_run_rows
                else 0.0
            ),
            "f1_mean": (
                float(mean(float(row.get("f1") or 0.0) for row in annotated_run_rows))
                if annotated_run_rows
                else 0.0
            ),
            "health_delta_mean": (
                float(
                    mean(
                        float(row.get("average_health_delta") or 0.0) for row in annotated_run_rows
                    )
                )
                if annotated_run_rows
                else 0.0
            ),
        },
        "per_question": per_question,
    }


def _run_markdown(payload: dict) -> str:
    lines = [
        "# GateKG Run Interpretation",
        "",
        f"- Log: `{payload['log_path']}`",
        f"- Entries: {payload['entry_count']}",
        f"- Selected variants: {payload['selected_variant_counts']}",
        f"- Gate decisions: {payload['gate_decision_counts']}",
        "",
        "## Per Question",
        "",
    ]
    for item in payload["per_question"][:20]:
        lines.append(
            f"- [{item['entry_index']}] variant={item['selected_variant']} "
            f"judge={item['judge_binary']} f1={float(item['f1'] or 0.0):.4f} "
            f"health_delta={float(item['average_health_delta'] or 0.0):.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "dataset":
        rows = annotate_rewards(read_dataset_rows(args.dataset))
        payload = interpret_dataset_rows(rows)
        prefix = _resolve_output_prefix("dataset", args.output_prefix, "summary")
        _write_reports(prefix, payload, _dataset_markdown(payload))
        print(f"Wrote dataset interpretation to {prefix.with_suffix('.json')}")
        print(f"Wrote dataset markdown report to {prefix.with_suffix('.md')}")
        return 0

    reference_rows = annotate_rewards(read_dataset_rows(args.dataset))
    payload = interpret_run_log(args.log, reference_rows)
    prefix = _resolve_output_prefix("run", args.output_prefix, Path(args.log).stem)
    _write_reports(prefix, payload, _run_markdown(payload))
    print(f"Wrote run interpretation to {prefix.with_suffix('.json')}")
    print(f"Wrote run markdown report to {prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
