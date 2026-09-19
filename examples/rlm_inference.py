"""Run recursive inference against a small QA dataset."""

from pathlib import Path

from gatekg.datasets import resolve_dataset
from gatekg.experiments import ExperimentOptions, run_experiment


def main() -> None:
    dataset = resolve_dataset(Path(__file__).with_name("qa.jsonl"))
    run_experiment(ExperimentOptions(dataset, runner="base_rlm", judge="exact"))


if __name__ == "__main__":
    main()
