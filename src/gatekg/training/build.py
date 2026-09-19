"""Build training datasets from experiment runs."""

from __future__ import annotations

import argparse

from gatekg.paths import runtime_paths

PROJECT_ROOT = runtime_paths().root

from gatekg.training.dataset import annotate_rewards, build_dataset_rows, write_dataset

DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "gatekg" / "runs.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "gatekg" / "runs.summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the GateKG training dataset from reference logs.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Root directory containing reference experiment logs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSONL dataset path.",
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Output summary JSON path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = annotate_rewards(build_dataset_rows(args.log_root))
    if not rows:
        raise ValueError("No manifested graph-baseline runs found in the log directory.")
    write_dataset(
        rows,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(f"Wrote {len(rows)} GateKG rows to {args.output}")
    print(f"Wrote dataset summary to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
