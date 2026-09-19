#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean

from gatekg.paths import runtime_paths

DEFAULT_DB = runtime_paths().cache / "kg_health" / "snapshots.sqlite3"
DEFAULT_OUT_DIR = runtime_paths().outputs / "health"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_rows(db_path: Path) -> list[dict]:
    if not db_path.is_file():
        return []

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                snapshot_id,
                snapshot_kind,
                base_group_id,
                write_group_id,
                average_health_score,
                metric_scores_json,
                metric_raw_values_json,
                metadata_json,
                created_at
            FROM health_snapshots
            ORDER BY created_at ASC
            """
        ).fetchall()
    finally:
        connection.close()

    parsed: list[dict] = []
    for row in rows:
        parsed.append(
            {
                "snapshot_id": row["snapshot_id"],
                "snapshot_kind": row["snapshot_kind"],
                "base_group_id": row["base_group_id"],
                "write_group_id": row["write_group_id"],
                "average_health_score": float(row["average_health_score"]),
                "metric_scores": json.loads(row["metric_scores_json"] or "{}"),
                "metric_raw_values": json.loads(row["metric_raw_values_json"] or "{}"),
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": row["created_at"],
            }
        )
    return parsed


def _summarize(rows: list[dict]) -> dict:
    baselines_by_group: dict[str, dict] = {}
    runs_by_group: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        if row["snapshot_kind"] == "baseline":
            baselines_by_group[row["base_group_id"]] = row
        elif row["snapshot_kind"] == "run":
            runs_by_group[row["base_group_id"]].append(row)

    group_summary: list[dict] = []
    metric_summary: dict[str, list[float]] = defaultdict(list)
    model_summary: dict[str, list[float]] = defaultdict(list)

    for base_group_id, run_rows in runs_by_group.items():
        baseline_row = baselines_by_group.get(base_group_id)
        baseline_score = (
            float(baseline_row["average_health_score"]) if baseline_row is not None else None
        )
        run_scores = [float(item["average_health_score"]) for item in run_rows]
        run_mean = mean(run_scores) if run_scores else 0.0
        delta = run_mean - baseline_score if baseline_score is not None else None

        if delta is not None:
            model_keys = {
                str(item.get("metadata", {}).get("model_key") or "unknown") for item in run_rows
            }
            for model_key in model_keys:
                model_summary[model_key].append(delta)

        baseline_metrics = baseline_row.get("metric_scores", {}) if baseline_row else {}
        for run_row in run_rows:
            run_metrics = run_row.get("metric_scores", {})
            for metric_name in sorted(set(baseline_metrics) | set(run_metrics)):
                baseline_metric = float(baseline_metrics.get(metric_name, 0.0))
                run_metric = float(run_metrics.get(metric_name, 0.0))
                metric_summary[metric_name].append(run_metric - baseline_metric)

        group_summary.append(
            {
                "base_group_id": base_group_id,
                "run_count": len(run_rows),
                "baseline_score": baseline_score,
                "run_mean_score": run_mean,
                "delta_mean": delta,
                "first_run_at": run_rows[0]["created_at"],
                "last_run_at": run_rows[-1]["created_at"],
            }
        )

    group_summary.sort(key=lambda item: item["run_count"], reverse=True)

    metric_rows = [
        {
            "metric": metric,
            "count": len(values),
            "delta_mean": mean(values),
            "delta_min": min(values),
            "delta_max": max(values),
        }
        for metric, values in sorted(metric_summary.items())
        if values
    ]

    model_rows = [
        {
            "model_key": model_key,
            "group_count": len(values),
            "delta_mean": mean(values),
            "delta_min": min(values),
            "delta_max": max(values),
        }
        for model_key, values in sorted(model_summary.items())
        if values
    ]

    return {
        "snapshot_count": len(rows),
        "baseline_count": len(baselines_by_group),
        "run_count": sum(len(items) for items in runs_by_group.values()),
        "group_summary": group_summary,
        "metric_summary": metric_rows,
        "model_summary": model_rows,
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                values.append(str(value))
            handle.write(",".join(values) + "\n")


def _build_markdown(summary: dict, db_path: Path) -> str:
    lines = [
        "# KG Health Summary",
        "",
        f"- Source DB: {db_path}",
        f"- Total snapshots: {summary['snapshot_count']}",
        f"- Baselines: {summary['baseline_count']}",
        f"- Run snapshots: {summary['run_count']}",
        "",
        "## Mean Delta by Metric",
        "",
    ]

    if not summary["metric_summary"]:
        lines.append("No run-vs-baseline metric deltas are available yet.")
    else:
        lines.append("| Metric | Count | Mean Delta | Min Delta | Max Delta |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in summary["metric_summary"]:
            lines.append(
                f"| {row['metric']} | {row['count']} | {row['delta_mean']:.4f} | {row['delta_min']:.4f} | {row['delta_max']:.4f} |"
            )

    lines.extend(["", "## Mean Delta by Model", ""])
    if not summary["model_summary"]:
        lines.append("No model-level deltas are available yet.")
    else:
        lines.append("| Model Key | Groups | Mean Delta | Min Delta | Max Delta |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in summary["model_summary"]:
            lines.append(
                f"| {row['model_key']} | {row['group_count']} | {row['delta_mean']:.4f} | {row['delta_min']:.4f} | {row['delta_max']:.4f} |"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze KG health snapshots for EMNLP reporting.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = _load_rows(args.db)
    summary = _summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "kg_health_summary.json", summary)
    _write_csv(
        args.out_dir / "kg_health_groups.csv",
        summary["group_summary"],
        [
            "base_group_id",
            "run_count",
            "baseline_score",
            "run_mean_score",
            "delta_mean",
            "first_run_at",
            "last_run_at",
        ],
    )
    _write_csv(
        args.out_dir / "kg_health_metrics.csv",
        summary["metric_summary"],
        ["metric", "count", "delta_mean", "delta_min", "delta_max"],
    )
    _write_csv(
        args.out_dir / "kg_health_models.csv",
        summary["model_summary"],
        ["model_key", "group_count", "delta_mean", "delta_min", "delta_max"],
    )

    (args.out_dir / "kg_health_report.md").write_text(
        _build_markdown(summary, args.db),
        encoding="utf-8",
    )

    print(f"Wrote KG health analysis to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
