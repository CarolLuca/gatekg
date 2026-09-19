"""Public contracts for experiment runners and judges."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from gatekg.utils.eval_runner import ExperimentEntry


@dataclass(frozen=True, slots=True)
class JudgeResult:
    score: int
    explanation: str
    context_chars_omitted: int = 0


class Judge(Protocol):
    async def ajudge(
        self,
        *,
        question: str,
        gold: str,
        prediction: str,
        context: str | list[str] | None = None,
    ) -> JudgeResult: ...


@dataclass(frozen=True, slots=True)
class RunnerContext:
    dataset_name: str
    artifact_dir: Path | None = None
    verbose: bool = False
    rebuild_kg: bool = False
    trajectory_path: Path | None = None


class PredictionModel(Protocol):
    @property
    def predict(self) -> Callable[..., str]: ...


class ExperimentRunner(Protocol):
    async def start(self) -> None: ...

    async def predict(self, entry: ExperimentEntry) -> str: ...

    async def close(self) -> None: ...


RunnerFactory = Callable[[RunnerContext], ExperimentRunner]
JudgeFactory = Callable[[], Judge]
