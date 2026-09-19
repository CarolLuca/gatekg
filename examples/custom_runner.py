"""Register an offline runner and execute the standard evaluation workflow."""

from pathlib import Path

from gatekg.contracts import RunnerContext
from gatekg.datasets import resolve_dataset
from gatekg.experiments import ExperimentOptions, run_experiment
from gatekg.registry import RunnerDefinition, register_runner
from gatekg.utils.eval_runner import ExperimentEntry


class ContextRunner:
    """Return supplied evidence as a minimal extension example."""

    def __init__(self, context: RunnerContext) -> None:
        self.context = context

    async def start(self) -> None:
        pass

    async def predict(self, entry: ExperimentEntry) -> str:
        return entry.context_text

    async def close(self) -> None:
        pass


def main() -> None:
    register_runner(RunnerDefinition("context", "Return input evidence", factory=ContextRunner))
    dataset = resolve_dataset(Path(__file__).with_name("qa.jsonl"))
    run_experiment(ExperimentOptions(dataset, runner="context", judge="exact"))


if __name__ == "__main__":
    main()
