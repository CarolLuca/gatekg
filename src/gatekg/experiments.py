"""Generic experiment orchestration used by the CLI and downstream projects."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gatekg.contracts import JudgeResult, RunnerContext
from gatekg.datasets import DatasetSpec, load_entries
from gatekg.judges import build_judge
from gatekg.paths import runtime_paths
from gatekg.records import write_run_manifest
from gatekg.registry import build_runner, get_runner
from gatekg.utils.eval_runner import (
    EvaluationResult,
    ExperimentEntry,
    evaluate_entries,
)
from gatekg.utils.run_logging import experiment_logging
from gatekg.utils.trajectory import trace_path


@dataclass(frozen=True, slots=True)
class ExperimentOptions:
    dataset: DatasetSpec
    runner: str
    limit: int = 0
    judge: str = "google"
    artifact_dir: Path | None = None
    output_path: Path | None = None
    verbose: bool = False
    rebuild_kg: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    run_id: str
    runner: str
    dataset: str
    dataset_path: str
    count: int
    metrics: dict[str, float]
    judge_distribution: dict[int, int]
    predictions: dict[str, str]
    answers: dict[str, str]
    output_path: str
    log_path: str
    judgments: dict[str, JudgeResult] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "gatekg-run-v1",
            "run_id": self.run_id,
            "runner": self.runner,
            "dataset": self.dataset,
            "dataset_path": self.dataset_path,
            "count": self.count,
            "metrics": self.metrics,
            "judge_distribution": self.judge_distribution,
            "predictions": self.predictions,
            "answers": self.answers,
            "judgments": {key: asdict(result) for key, result in self.judgments.items()},
            "output_path": self.output_path,
            "log_path": self.log_path,
        }


async def _evaluate(
    options: ExperimentOptions,
    entries: list[ExperimentEntry],
    *,
    trajectory_path: Path | None = None,
) -> EvaluationResult:
    judge = build_judge(options.judge)
    runner = build_runner(
        options.runner,
        RunnerContext(
            dataset_name=options.dataset.name,
            artifact_dir=options.artifact_dir,
            verbose=options.verbose,
            rebuild_kg=options.rebuild_kg,
            trajectory_path=trajectory_path,
        ),
    )
    try:
        await runner.start()
        return await evaluate_entries(
            entries,
            predict_entry=runner.predict,
            judge_llm=judge,
        )
    finally:
        await runner.close()


def _default_output(run_id: str) -> Path:
    return runtime_paths().outputs / f"{run_id}.json"


def _run_id(dataset_name: str, runner: str) -> str:
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("_") or "dataset"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{safe_dataset}_{runner}_{timestamp}"


def run_experiment(options: ExperimentOptions) -> ExperimentReport:
    """Run one experiment synchronously and persist a machine-readable report."""

    get_runner(options.runner)
    if options.limit < 0:
        raise ValueError("limit cannot be negative")
    entries = load_entries(options.dataset, limit=options.limit)
    run_id = _run_id(options.dataset.name, options.runner)
    output_path = (options.output_path or _default_output(run_id)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with experiment_logging(
        task_name=options.dataset.name,
        runner_name=options.runner,
        judge=options.judge,
    ) as log_path:
        write_run_manifest(
            log_path,
            run_id=run_id,
            dataset=options.dataset,
            runner=options.runner,
            judge=options.judge,
            entries=entries,
        )
        result = asyncio.run(_evaluate(options, entries, trajectory_path=trace_path(log_path)))
        report = ExperimentReport(
            run_id=run_id,
            runner=options.runner,
            dataset=options.dataset.name,
            dataset_path=str(options.dataset.path),
            count=result.metrics.count,
            metrics=result.metrics.averages(),
            judge_distribution=result.metrics.judge_distribution(),
            predictions=result.predictions,
            answers=result.answers,
            judgments=result.judgments,
            output_path=str(output_path),
            log_path=str(log_path),
        )
        output_path.write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Result: {output_path}")
    return report
