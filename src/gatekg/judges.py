"""Judge implementations and extension registry."""

from __future__ import annotations

from gatekg.contracts import Judge, JudgeFactory, JudgeResult
from gatekg.evaluation.hotpot import exact_match_score


class ExactMatchJudge:
    async def ajudge(
        self,
        *,
        question: str,
        gold: str,
        prediction: str,
        context: str | list[str] | None = None,
    ) -> JudgeResult:
        del question, context
        score = int(exact_match_score(prediction, gold))
        explanation = "normalized exact match" if score else "normalized answers differ"
        return JudgeResult(score=score, explanation=explanation)


class JudgeRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, JudgeFactory] = {}

    def register(self, name: str, factory: JudgeFactory, *, replace: bool = False) -> None:
        if not name:
            raise ValueError("judge name cannot be empty")
        if name in self._factories and not replace:
            raise ValueError(f"Judge `{name}` is already registered")
        self._factories[name] = factory

    def create(self, name: str) -> Judge:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            choices = ", ".join(self.names())
            raise ValueError(f"Unknown judge `{name}`. Choose one of: {choices}") from exc
        return factory()

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)


def _google_judge() -> Judge:
    from gatekg.api.judge_llm import JudgeLLM

    return JudgeLLM()


JUDGES = JudgeRegistry()
JUDGES.register("google", _google_judge)
JUDGES.register("exact", ExactMatchJudge)


def register_judge(name: str, factory: JudgeFactory, *, replace: bool = False) -> None:
    """Register a judge factory for programmatic use and CLI discovery."""

    JUDGES.register(name, factory, replace=replace)


def build_judge(name: str) -> Judge:
    return JUDGES.create(name)


def judge_names() -> tuple[str, ...]:
    return JUDGES.names()
