"""Evaluate graph-backed retrieval using the configured Neo4j database."""

from pathlib import Path

from gatekg.datasets import resolve_dataset
from gatekg.experiments import ExperimentOptions, run_experiment


def main() -> None:
    dataset = resolve_dataset(Path(__file__).with_name("qa.jsonl"))
    run_experiment(ExperimentOptions(dataset, runner="read_only_rlm", judge="exact"))


if __name__ == "__main__":
    main()
