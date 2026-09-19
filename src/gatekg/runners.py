"""Built-in experiment runner adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gatekg.contracts import ExperimentRunner, RunnerContext
from gatekg.paths import policy_artifact_dir
from gatekg.utils.eval_runner import ExperimentEntry, predict_with_model
from gatekg.utils.trajectory import capture_entry


class ManagedRunner(ABC):
    """Base class for runners with optional asynchronous resources."""

    async def start(self) -> None:
        return None

    @abstractmethod
    async def predict(self, entry: ExperimentEntry) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class DirectLLMRunner(ManagedRunner):
    def __init__(self) -> None:
        from gatekg.api.llm import LLM

        self._model = LLM()

    async def predict(self, entry: ExperimentEntry) -> str:
        from gatekg.utils.prompts import LLM_SYSTEM_PROMPT_FUNC

        prompt = LLM_SYSTEM_PROMPT_FUNC(entry.question, entry.context_text)
        return await predict_with_model(self._model, prompt)


class RawContextRLMRunner(ManagedRunner):
    def __init__(self, *, verbose: bool) -> None:
        from gatekg.api.mcp_rlm import BaseRLM

        self._model = BaseRLM(verbose=verbose)

    async def predict(self, entry: ExperimentEntry) -> str:
        from gatekg.utils.mcp_prompts import BASE_RLM_SYSTEM_PROMPT_FUNC

        prompt = BASE_RLM_SYSTEM_PROMPT_FUNC(entry.question, entry.context_text)
        return await predict_with_model(self._model, prompt, entry.question)


class SupportingFactsRLMRunner(ManagedRunner):
    def __init__(self, *, verbose: bool) -> None:
        from gatekg.api.mcp_rlm import CustomBaseRLM

        self._model = CustomBaseRLM(verbose=verbose)

    async def predict(self, entry: ExperimentEntry) -> str:
        from gatekg.utils.mcp_prompts import CUSTOM_BASE_RLM_SYSTEM_PROMPT_FUNC

        self._model.set_context(entry.context_chunks)
        prompt = CUSTOM_BASE_RLM_SYSTEM_PROMPT_FUNC(entry.question)
        return await predict_with_model(self._model, prompt, entry.question)


class GraphRLMRunner(ManagedRunner):
    def __init__(
        self,
        *,
        model: Any,
        prompt: Callable[[str], str],
        rebuild_kg: bool,
        trajectory_path: Path | None = None,
    ) -> None:
        self._model = model
        self._prompt = prompt
        self._rebuild_kg = rebuild_kg
        self._trajectory_path = trajectory_path

    async def start(self) -> None:
        await self._model.init_kg()

    async def predict(self, entry: ExperimentEntry) -> str:
        await self._model.prepare_context(
            entry.context_chunks,
            kg_context_chunks=entry.kg_context_chunks,
            rebuild=self._rebuild_kg,
        )
        with capture_entry(self._model, self._trajectory_path, entry.entry_id):
            return await predict_with_model(
                self._model,
                self._prompt(entry.question),
                entry.question,
            )

    async def close(self) -> None:
        await self._model.close_kg()


class GateKGExperimentRunner(ManagedRunner):
    def __init__(self, context: RunnerContext) -> None:
        try:
            from gatekg.api.gatekg import GateKG
        except ImportError as exc:
            raise RuntimeError(
                "The gatekg runner requires the `adaptive` extra: pip install 'gatekg[adaptive]'"
            ) from exc

        artifact_dir = policy_artifact_dir(context.artifact_dir)
        self._model = GateKG(
            dataset_variant=context.dataset_name,
            artifact_dir=artifact_dir,
            verbose=context.verbose,
        )
        self._rebuild_kg = context.rebuild_kg
        self._trajectory_path = context.trajectory_path

    async def start(self) -> None:
        await self._model.init_kg()

    async def predict(self, entry: ExperimentEntry) -> str:
        await self._model.prepare_context(
            entry.context_chunks,
            question=entry.question,
            kg_context_chunks=entry.kg_context_chunks,
            rebuild=self._rebuild_kg,
        )
        with capture_entry(self._model, self._trajectory_path, entry.entry_id):
            return await predict_with_model(
                self._model,
                self._model.build_prompt(entry.question),
                entry.question,
            )

    async def close(self) -> None:
        await self._model.close_kg()


def build_direct_llm(context: RunnerContext) -> ExperimentRunner:
    del context
    return DirectLLMRunner()


def build_base_rlm(context: RunnerContext) -> ExperimentRunner:
    return RawContextRLMRunner(verbose=context.verbose)


def build_custom_base_rlm(context: RunnerContext) -> ExperimentRunner:
    return SupportingFactsRLMRunner(verbose=context.verbose)


def build_read_only_rlm(context: RunnerContext) -> ExperimentRunner:
    from gatekg.api.mcp_rlm import ReadOnlyRLM
    from gatekg.utils.mcp_prompts import READ_ONLY_RLM_SYSTEM_PROMPT_FUNC

    return GraphRLMRunner(
        model=ReadOnlyRLM(verbose=context.verbose),
        prompt=READ_ONLY_RLM_SYSTEM_PROMPT_FUNC,
        rebuild_kg=context.rebuild_kg,
        trajectory_path=context.trajectory_path,
    )


def build_read_write_rlm(context: RunnerContext) -> ExperimentRunner:
    from gatekg.api.mcp_rlm import ReadWriteRLM
    from gatekg.utils.mcp_prompts import READ_WRITE_RLM_SYSTEM_PROMPT_FUNC

    return GraphRLMRunner(
        model=ReadWriteRLM(verbose=context.verbose),
        prompt=READ_WRITE_RLM_SYSTEM_PROMPT_FUNC,
        rebuild_kg=context.rebuild_kg,
        trajectory_path=context.trajectory_path,
    )


def build_aggressive_write_rlm(context: RunnerContext) -> ExperimentRunner:
    from gatekg.api.mcp_rlm import AggressiveWriteRLM
    from gatekg.utils.mcp_prompts import AGGRESSIVE_WRITE_RLM_PROMPT_FUNC

    return GraphRLMRunner(
        model=AggressiveWriteRLM(verbose=context.verbose),
        prompt=AGGRESSIVE_WRITE_RLM_PROMPT_FUNC,
        rebuild_kg=context.rebuild_kg,
        trajectory_path=context.trajectory_path,
    )


def build_gatekg(context: RunnerContext) -> ExperimentRunner:
    return GateKGExperimentRunner(context)


BUILTIN_RUNNER_FACTORIES = {
    "llm": build_direct_llm,
    "base_rlm": build_base_rlm,
    "custom_base_rlm": build_custom_base_rlm,
    "read_only_rlm": build_read_only_rlm,
    "read_write_rlm": build_read_write_rlm,
    "aggressive_write_rlm": build_aggressive_write_rlm,
    "gatekg": build_gatekg,
}
