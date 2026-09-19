from __future__ import annotations

import asyncio

import pytest

from gatekg.contracts import JudgeResult, RunnerContext
from gatekg.datasets import DatasetSpec
from gatekg.experiments import ExperimentOptions
from gatekg.judges import JudgeRegistry
from gatekg.registry import RunnerDefinition, RunnerRegistry


class _StubRunner:
    async def start(self) -> None:
        return None

    async def predict(self, entry) -> str:
        return entry.question

    async def close(self) -> None:
        return None


class _StubJudge:
    async def ajudge(self, **kwargs) -> JudgeResult:
        return JudgeResult(score=1, explanation="accepted")


def test_runner_registry_builds_custom_runner():
    registry = RunnerRegistry()
    context = RunnerContext(dataset_name="custom")

    def build_runner(received: RunnerContext) -> _StubRunner:
        assert received == context
        return _StubRunner()

    registry.register(
        RunnerDefinition(
            name="custom",
            description="Test runner.",
            factory=build_runner,
        )
    )

    assert isinstance(registry.create("custom", context), _StubRunner)
    assert registry.names() == ("custom",)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(RunnerDefinition("custom", "Duplicate.", lambda _: _StubRunner()))


def test_judge_registry_builds_custom_judge():
    registry = JudgeRegistry()
    registry.register("custom", _StubJudge)

    assert isinstance(registry.create("custom"), _StubJudge)


def test_experiment_runner_is_closed_when_evaluation_fails(monkeypatch, tmp_path):
    from gatekg import experiments

    events: list[str] = []

    class ManagedStub(_StubRunner):
        async def start(self) -> None:
            events.append("start")

        async def close(self) -> None:
            events.append("close")

    async def fail_evaluation(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(experiments, "build_runner", lambda *args: ManagedStub())
    monkeypatch.setattr(experiments, "build_judge", lambda name: _StubJudge())
    monkeypatch.setattr(experiments, "evaluate_entries", fail_evaluation)

    options = ExperimentOptions(
        dataset=DatasetSpec(name="custom", path=tmp_path / "data.jsonl"),
        runner="custom",
        judge="custom",
    )
    with pytest.raises(RuntimeError, match="evaluation failed"):
        asyncio.run(experiments._evaluate(options, []))

    assert events == ["start", "close"]
