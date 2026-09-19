"""Command-line interface for reproducible GateKG experiments."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gatekg",
        description="Run gated knowledge-graph memory experiments.",
    )
    parser.add_argument("--config", type=Path, help="YAML runtime configuration.")
    parser.add_argument("--home", type=Path, help="Project root for runtime directories.")
    parser.add_argument("--version", action="version", version="gatekg 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List registered datasets and runners.")
    list_parser.add_argument(
        "kind", choices=("datasets", "runners", "all"), nargs="?", default="all"
    )

    run_parser = commands.add_parser("run", help="Run one experiment.")
    _add_run_arguments(run_parser)

    batch_parser = commands.add_parser("batch", help="Run experiments from a YAML manifest.")
    batch_parser.add_argument("manifest", type=Path)
    batch_parser.add_argument("--dry-run", action="store_true")

    doctor_parser = commands.add_parser("doctor", help="Validate configuration and runtime inputs.")
    doctor_parser.add_argument("--runner", choices=_runner_names())
    doctor_parser.add_argument("--dataset", help="Registered name or dataset path to validate.")

    return parser


def _runner_names() -> tuple[str, ...]:
    from gatekg.registry import runner_names

    return runner_names()


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", help="Registered dataset name or JSON/JSONL path.")
    parser.add_argument("runner", choices=_runner_names())
    parser.add_argument("--format", choices=("auto", "qa_jsonl", "hotpotqa"), default="auto")
    parser.add_argument("--limit", type=int, default=0, help="0 means all entries.")
    parser.add_argument(
        "--segment-context",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--chunk-words", type=int, default=None)
    parser.add_argument("--overlap-ratio", type=float, default=None)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--context-field", default="context")
    parser.add_argument("--kg-context-field")
    from gatekg.judges import judge_names

    parser.add_argument("--judge", choices=judge_names(), default="google")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rebuild-kg", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and load data only.")


def _configure_process(args: argparse.Namespace) -> None:
    if args.home:
        os.environ["GATEKG_HOME"] = str(args.home.expanduser().resolve())
    if args.config:
        os.environ["GATEKG_CONFIG"] = str(args.config.expanduser().resolve())


@contextlib.contextmanager
def _configured_process(args: argparse.Namespace):
    keys = ("GATEKG_HOME", "GATEKG_CONFIG")
    previous = {key: os.environ.get(key) for key in keys}
    _configure_process(args)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _dataset_from_values(values: dict[str, Any]):
    from gatekg.datasets import FieldMapping, resolve_dataset

    fields = FieldMapping(
        identifier=str(values.get("id_field", "id")),
        question=str(values.get("question_field", "question")),
        answer=str(values.get("answer_field", "answer")),
        context=str(values.get("context_field", "context")),
        kg_context=values.get("kg_context_field"),
    )
    return resolve_dataset(
        values["dataset"],
        format=values.get("format", "auto"),
        segment_context=values.get("segment_context"),
        chunk_words=values.get("chunk_words"),
        overlap_ratio=values.get("overlap_ratio"),
        fields=fields,
    )


def _validate_runner_inputs(runner_name: str, artifact_dir: Path | None) -> None:
    from gatekg.paths import policy_artifact_dir
    from gatekg.registry import get_runner
    from gatekg.utils.config import load_config

    runner = get_runner(runner_name)
    config = load_config()
    if runner.requires_kg:
        missing = [
            name
            for name, value in (
                ("NEO4J_URI", config.neo4j_uri),
                ("NEO4J_USERNAME", config.neo4j_username),
                ("NEO4J_PASSWORD", config.neo4j_password),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Runner `{runner_name}` requires: {', '.join(missing)}")
    if runner.requires_artifacts:
        resolved = policy_artifact_dir(artifact_dir)
        required = ("feature_config.json", "label_maps.json", "selector.pt", "tool_gate.pt")
        missing = [name for name in required if not (resolved / name).is_file()]
        if missing:
            raise ValueError(
                f"Adaptive artifacts are incomplete in {resolved}; missing: {', '.join(missing)}"
            )


def _run(values: dict[str, Any], *, dry_run: bool = False):
    from gatekg.datasets import load_entries
    from gatekg.experiments import ExperimentOptions, run_experiment

    dataset = _dataset_from_values(values)
    artifact_value = values.get("artifact_dir")
    output_value = values.get("output")
    artifact_dir = Path(artifact_value).expanduser().resolve() if artifact_value else None
    output_path = Path(output_value).expanduser().resolve() if output_value else None
    from gatekg.judges import judge_names
    from gatekg.registry import get_runner

    get_runner(values["runner"])
    if values.get("judge", "google") not in judge_names():
        raise ValueError(f"Unknown judge: {values['judge']}")
    if dry_run or values.get("dry_run"):
        entries = load_entries(dataset, limit=int(values.get("limit", 0)))
        print(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "path": str(dataset.path),
                    "format": dataset.format.value,
                    "entries": len(entries),
                    "runner": values["runner"],
                },
                indent=2,
            )
        )
        return None

    _validate_runner_inputs(values["runner"], artifact_dir)
    options = ExperimentOptions(
        dataset=dataset,
        runner=values["runner"],
        limit=int(values.get("limit", 0)),
        judge=values.get("judge", "google"),
        artifact_dir=artifact_dir,
        output_path=output_path,
        verbose=bool(values.get("verbose", False)),
        rebuild_kg=bool(values.get("rebuild_kg", False)),
    )
    return run_experiment(options)


def _list(kind: str) -> None:
    from gatekg.datasets import builtin_datasets
    from gatekg.registry import RUNNERS

    if kind in {"runners", "all"}:
        print("Runners:")
        for runner in RUNNERS.values():
            requirements = []
            if runner.requires_kg:
                requirements.append("Neo4j")
            if runner.requires_artifacts:
                requirements.append("trained artifacts")
            suffix = f" [{', '.join(requirements)}]" if requirements else ""
            print(f"  {runner.name:24} {runner.description}{suffix}")
    if kind == "all":
        print()
    if kind in {"datasets", "all"}:
        print("Datasets:")
        for spec in builtin_datasets().values():
            state = "ready" if spec.path.is_file() else "missing"
            print(f"  {spec.name:24} {state:7} {spec.path}")


def _doctor(args: argparse.Namespace) -> int:
    from gatekg.datasets import resolve_dataset
    from gatekg.paths import runtime_paths
    from gatekg.utils.config import load_config, resolve_config_path

    config = load_config()
    paths = runtime_paths()
    checks: list[tuple[str, bool, str]] = [
        ("config", True, str(resolve_config_path())),
        ("project root", paths.root.is_dir(), str(paths.root)),
        ("data directory", paths.data.is_dir(), str(paths.data)),
        ("provider", True, f"{config.provider}:{config.llm_model_name}"),
    ]
    if config.provider == "google":
        checks.append(("Google credential", bool(config.google.api_key), "environment variable"))
    if args.dataset:
        spec = resolve_dataset(args.dataset)
        checks.append(("dataset", spec.path.is_file(), str(spec.path)))
    if args.runner:
        try:
            _validate_runner_inputs(args.runner, None)
        except ValueError as exc:
            checks.append((f"runner {args.runner}", False, str(exc)))
        else:
            checks.append((f"runner {args.runner}", True, "requirements present"))

    failed = False
    for name, ok, detail in checks:
        marker = "OK" if ok else "WARN"
        print(f"[{marker:4}] {name}: {detail}")
        failed = failed or (not ok and name not in {"data directory"})
    return 1 if failed else 0


def _batch(manifest_path: Path, *, dry_run: bool) -> None:
    from yaml import safe_load

    from gatekg.datasets import builtin_datasets

    resolved = manifest_path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as source:
        manifest = safe_load(source)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list):
        raise ValueError("Batch manifest must contain a `runs` list.")
    defaults = manifest.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("Batch manifest `defaults` must be a mapping.")
    for index, run in enumerate(manifest["runs"], start=1):
        if not isinstance(run, dict):
            raise ValueError(f"Batch run {index} must be a mapping.")
        values = {**defaults, **run}
        if "dataset" not in values or "runner" not in values:
            raise ValueError(f"Batch run {index} requires `dataset` and `runner`.")
        if str(values["dataset"]) not in builtin_datasets():
            path = Path(values["dataset"]).expanduser()
            values["dataset"] = str(path if path.is_absolute() else resolved.parent / path)
        for key in ("artifact_dir", "output"):
            if values.get(key):
                path = Path(values[key]).expanduser()
                values[key] = str(path if path.is_absolute() else resolved.parent / path)
        print(
            f"Batch run {index}/{len(manifest['runs'])}: {values['dataset']} / {values['runner']}"
        )
        _run(values, dry_run=dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    with _configured_process(args):
        try:
            if args.command == "list":
                _list(args.kind)
            elif args.command == "run":
                _run(vars(args))
            elif args.command == "batch":
                _batch(args.manifest, dry_run=args.dry_run)
            elif args.command == "doctor":
                return _doctor(args)
            else:  # pragma: no cover
                parser.error(f"unsupported command: {args.command}")
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            print(f"gatekg: error: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
