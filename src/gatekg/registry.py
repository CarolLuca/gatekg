"""Discoverable registry for experiment runners."""

from __future__ import annotations

from collections.abc import Iterator, ValuesView
from dataclasses import dataclass

from gatekg.contracts import ExperimentRunner, RunnerContext, RunnerFactory


@dataclass(frozen=True, slots=True)
class RunnerDefinition:
    name: str
    description: str
    factory: RunnerFactory
    requires_kg: bool = False
    requires_artifacts: bool = False


class RunnerRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, RunnerDefinition] = {}

    def register(self, definition: RunnerDefinition, *, replace: bool = False) -> None:
        if not definition.name:
            raise ValueError("runner name cannot be empty")
        if definition.name in self._definitions and not replace:
            raise ValueError(f"Runner `{definition.name}` is already registered")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> RunnerDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            choices = ", ".join(self._definitions)
            raise ValueError(f"Unknown runner `{name}`. Choose one of: {choices}") from exc

    def create(self, name: str, context: RunnerContext) -> ExperimentRunner:
        return self.get(name).factory(context)

    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def values(self) -> ValuesView[RunnerDefinition]:
        return self._definitions.values()

    def __iter__(self) -> Iterator[str]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, name: object) -> bool:
        return name in self._definitions

    def __getitem__(self, name: str) -> RunnerDefinition:
        return self._definitions[name]


def _builtin_factory(name: str) -> RunnerFactory:
    def create(context: RunnerContext) -> ExperimentRunner:
        from gatekg.runners import BUILTIN_RUNNER_FACTORIES

        return BUILTIN_RUNNER_FACTORIES[name](context)

    return create


RUNNERS = RunnerRegistry()


def _register_builtins() -> None:
    definitions = (
        RunnerDefinition("llm", "Direct language-model baseline.", _builtin_factory("llm")),
        RunnerDefinition(
            "base_rlm",
            "Recursive model over raw context.",
            _builtin_factory("base_rlm"),
        ),
        RunnerDefinition(
            "custom_base_rlm",
            "Recursive model with explicit supporting-facts access.",
            _builtin_factory("custom_base_rlm"),
        ),
        RunnerDefinition(
            "read_only_rlm",
            "Recursive model with read-only KG tools.",
            _builtin_factory("read_only_rlm"),
            requires_kg=True,
        ),
        RunnerDefinition(
            "read_write_rlm",
            "Recursive model with read/write KG tools.",
            _builtin_factory("read_write_rlm"),
            requires_kg=True,
        ),
        RunnerDefinition(
            "aggressive_write_rlm",
            "Write-heavy recursive KG baseline.",
            _builtin_factory("aggressive_write_rlm"),
            requires_kg=True,
        ),
        RunnerDefinition(
            "gatekg",
            "Learned runner and tool-use gate.",
            _builtin_factory("gatekg"),
            requires_kg=True,
            requires_artifacts=True,
        ),
    )
    for definition in definitions:
        RUNNERS.register(definition)


_register_builtins()


def register_runner(definition: RunnerDefinition, *, replace: bool = False) -> None:
    """Register a runner for programmatic use and CLI discovery."""

    RUNNERS.register(definition, replace=replace)


def get_runner(name: str) -> RunnerDefinition:
    return RUNNERS.get(name)


def build_runner(name: str, context: RunnerContext) -> ExperimentRunner:
    return RUNNERS.create(name, context)


def runner_names() -> tuple[str, ...]:
    return RUNNERS.names()
