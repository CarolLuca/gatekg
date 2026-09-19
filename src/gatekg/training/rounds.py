"""Collect traces, label trajectories, and train policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gatekg.paths import runtime_paths

PROJECT_ROOT = runtime_paths().root

from gatekg.health.store import KGHealthSnapshotStore
from gatekg.training.dataset import (
    annotate_rewards,
    build_dataset_rows,
    parse_run_log,
    write_dataset,
)
from gatekg.training.label import (
    DEFAULT_JUDGE_RETRIES,
    DEFAULT_JUDGE_RETRY_DELAY_SECONDS,
    DEFAULT_PROGRESS_INTERVAL,
    label_dataset,
)
from gatekg.training.train import train_gatekg

DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "gatekg" / "rounds"
DEFAULT_SELECTOR_EPOCHS = 50
DEFAULT_GATE_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_SELECTOR_DROPOUT = 0.2
DEFAULT_GATE_DROPOUT = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SEED = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advance GateKG using LLM-labeled trajectory gate supervision.",
    )
    parser.add_argument("--include-aggressive-write", action="store_true")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument(
        "--gate-label-source",
        choices=("reward_divergence", "trajectory_state"),
        default="reward_divergence",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Root directory containing experiment logs.",
    )
    parser.add_argument(
        "--log-path",
        action="append",
        default=[],
        help="Explicit log file to include. Can be passed multiple times.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--raw-dataset-output", default="")
    parser.add_argument("--dataset-output", default="")
    parser.add_argument("--raw-summary-output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--trajectory-cache", default="")
    parser.add_argument("--selector-epochs", type=int, default=DEFAULT_SELECTOR_EPOCHS)
    parser.add_argument("--gate-epochs", type=int, default=DEFAULT_GATE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--selector-dropout", type=float, default=DEFAULT_SELECTOR_DROPOUT)
    parser.add_argument("--gate-dropout", type=float, default=DEFAULT_GATE_DROPOUT)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--trajectory-row-limit",
        type=int,
        default=0,
        help="Optional row limit for trajectory labeling.",
    )
    parser.add_argument(
        "--trajectory-call-limit",
        type=int,
        default=0,
        help="Optional new-call limit for trajectory labeling.",
    )
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=DEFAULT_JUDGE_RETRIES,
        help="Maximum attempts per uncached trajectory judge request.",
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
        help="Print progress after this many processed trajectory tool calls.",
    )
    return parser.parse_args(argv)


def _resolve_round_directory(root: Path, round_number: int) -> Path:
    return root / f"round_{round_number:03d}"


def _collect_rows(*, log_root: Path, log_paths: list[Path]) -> list[dict[str, Any]]:
    if log_paths:
        store = KGHealthSnapshotStore()
        rows: list[dict[str, Any]] = []
        for log_path in log_paths:
            rows.extend(parse_run_log(log_path, store=store))
        return rows
    return build_dataset_rows(log_root)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.round < 1:
        raise ValueError("round must be positive")
    output_root = Path(args.output_root)
    round_dir = _resolve_round_directory(output_root, args.round)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else round_dir
    raw_dataset_path = (
        Path(args.raw_dataset_output)
        if args.raw_dataset_output
        else round_dir / "dataset.raw.jsonl"
    )
    dataset_path = Path(args.dataset_output) if args.dataset_output else round_dir / "dataset.jsonl"
    raw_summary_path = (
        Path(args.raw_summary_output)
        if args.raw_summary_output
        else round_dir / "dataset.raw.summary.json"
    )
    summary_path = (
        Path(args.summary_output) if args.summary_output else round_dir / "dataset.summary.json"
    )
    trajectory_cache_path = (
        Path(args.trajectory_cache)
        if args.trajectory_cache
        else round_dir / "trajectory_label_cache.json"
    )

    log_paths = [Path(path) for path in args.log_path]
    print(
        "GateKG round setup: "
        f"round={args.round} output_root={output_root} artifact_dir={artifact_dir}",
        flush=True,
    )
    print(
        "Collecting raw rows from "
        + (
            f"{len(log_paths)} explicit log path(s)"
            if log_paths
            else f"log root {Path(args.log_root)}"
        ),
        flush=True,
    )
    rows = annotate_rewards(
        _collect_rows(
            log_root=Path(args.log_root),
            log_paths=log_paths,
        )
    )
    if not rows:
        raise ValueError("No GateKG rows were found in the supplied logs.")
    print(
        "Collected raw dataset: "
        f"rows={len(rows)} tool_calls={sum(len(row.get('tool_calls') or []) for row in rows)}",
        flush=True,
    )

    print(f"Writing raw dataset to {raw_dataset_path}", flush=True)
    write_dataset(rows, output_path=raw_dataset_path, summary_path=raw_summary_path)
    print(f"Writing raw summary to {raw_summary_path}", flush=True)
    if args.gate_label_source == "trajectory_state":
        labeled_rows = label_dataset(
            dataset_path=raw_dataset_path,
            output_path=dataset_path,
            summary_path=summary_path,
            cache_path=trajectory_cache_path,
            row_limit=args.trajectory_row_limit,
            call_limit=args.trajectory_call_limit,
            judge_retries=args.judge_retries,
            judge_retry_delay=args.judge_retry_delay,
            progress_interval=args.progress_interval,
        )
    else:
        labeled_rows = rows
        write_dataset(rows, output_path=dataset_path, summary_path=summary_path)

    print(
        "Starting GateKG training: "
        f"selector_epochs={args.selector_epochs} gate_epochs={args.gate_epochs} "
        f"batch_size={args.batch_size} lr={args.learning_rate}",
        flush=True,
    )
    metrics = train_gatekg(
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        selector_epochs=args.selector_epochs,
        gate_epochs=args.gate_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        group_split=True,
        include_aggressive_write=args.include_aggressive_write,
        selector_dropout=args.selector_dropout,
        gate_dropout=args.gate_dropout,
        gate_label_source=args.gate_label_source,
    )

    manifest = {
        "round": args.round,
        "gate_label_source": args.gate_label_source,
        "row_count": len(labeled_rows),
        "log_root": str(Path(args.log_root)),
        "log_paths": [str(path) for path in log_paths],
        "raw_dataset_path": str(raw_dataset_path),
        "dataset_path": str(dataset_path),
        "raw_summary_path": str(raw_summary_path),
        "summary_path": str(summary_path),
        "trajectory_cache_path": str(trajectory_cache_path),
        "artifact_dir": str(artifact_dir),
        "raw_row_count": len(rows),
        "labeled_row_count": len(labeled_rows),
        "judge_retries": args.judge_retries,
        "judge_retry_delay": args.judge_retry_delay,
        "progress_interval": args.progress_interval,
        "dataset_summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "training_metrics": metrics,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing round manifest to {artifact_dir / 'round_manifest.json'}", flush=True)
    (artifact_dir / "round_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "Advanced GateKG round "
        f"{args.round}: rows={len(labeled_rows)} artifact_dir={artifact_dir} "
        f"dataset={dataset_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
