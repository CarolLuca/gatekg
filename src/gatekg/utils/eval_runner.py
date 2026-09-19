from __future__ import annotations

import asyncio
import inspect
import json
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from gatekg.contracts import Judge, JudgeResult, PredictionModel
from gatekg.evaluation.hotpot import exact_match_score, f1_score

LOCAL_GENERATION_CONCURRENCY = 1
GEMINI_JUDGE_CONCURRENCY = 8
NEO4J_BUILD_CONCURRENCY = 1
JUDGE_SCORE_VALUES = (0, 1)


@dataclass(frozen=True)
class ExperimentEntry:
    entry_id: str
    question: str
    gold: str
    context_chunks: list[str]
    kg_context_chunks: list[str] | None = None

    @property
    def context_text(self) -> str:
        return "\n\n".join(self.context_chunks)

    @property
    def kg_context_text(self) -> str:
        return "\n\n".join(self.kg_context_chunks or self.context_chunks)


@dataclass
class AggregateMetrics:
    count: int = 0
    judge_total: float = 0.0
    em_total: float = 0.0
    f1_total: float = 0.0
    precision_total: float = 0.0
    recall_total: float = 0.0
    judge_histogram: Counter[int] = field(default_factory=Counter)

    def update(self, prediction: str, gold: str, judge_result: JudgeResult) -> None:
        em = float(exact_match_score(prediction, gold))
        f1, precision, recall = f1_score(prediction, gold)
        self.count += 1
        self.judge_total += judge_result.score
        self.em_total += em
        self.f1_total += f1
        self.precision_total += precision
        self.recall_total += recall
        self.judge_histogram[judge_result.score] += 1

    def averages(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "judge": 0.0,
                "em": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
        return {
            "judge": self.judge_total / self.count,
            "em": self.em_total / self.count,
            "f1": self.f1_total / self.count,
            "precision": self.precision_total / self.count,
            "recall": self.recall_total / self.count,
        }

    def judge_distribution(self) -> dict[int, int]:
        return {score: self.judge_histogram.get(score, 0) for score in JUDGE_SCORE_VALUES}


@dataclass
class EvaluationResult:
    metrics: AggregateMetrics
    predictions: dict[str, str]
    answers: dict[str, str]
    judgments: dict[str, JudgeResult] = field(default_factory=dict)


@dataclass
class _PendingJudge:
    entry: ExperimentEntry
    prediction: str
    task: asyncio.Task[JudgeResult]


def load_hotpotqa_entries(dataset_path: str, entry_limit: int = 0) -> list[ExperimentEntry]:
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = dataset[:entry_limit] if entry_limit > 0 else dataset

    return [
        ExperimentEntry(
            entry_id=entry["id"],
            question=entry["question"],
            gold=entry["answer"],
            context_chunks=[
                entry["context"]["title"][index] + ":\n" + "\n".join(sentences)
                for index, sentences in enumerate(entry["context"]["sentences"])
            ],
        )
        for entry in dataset
    ]


async def predict_with_model(
    model: PredictionModel,
    prompt: str,
    question: str | None = None,
) -> str:
    if getattr(model, "run_predict_in_worker_thread", True):
        if question is None:
            prediction = await asyncio.to_thread(model.predict, prompt)
        else:
            prediction = await asyncio.to_thread(model.predict, prompt, question)
    else:
        if question is None:
            prediction = model.predict(prompt)
        else:
            prediction = model.predict(prompt, question)

    await _run_after_predict_hook(
        model,
        prompt=prompt,
        question=question,
        prediction=prediction,
    )
    return prediction


async def _run_after_predict_hook(
    model: object,
    *,
    prompt: str,
    question: str | None,
    prediction: str,
) -> None:
    hook = getattr(model, "after_predict", None)
    if not callable(hook):
        return

    hook_result = hook(
        prompt=prompt,
        question=question,
        prediction=prediction,
    )
    if inspect.isawaitable(hook_result):
        await hook_result


def _print_running_metrics(metrics: AggregateMetrics) -> None:
    averages = metrics.averages()
    print(f"curr judge: {averages['judge']}")
    print(f"curr em: {averages['em']}")
    print(f"curr f1: {averages['f1']}")
    print("-----------------")


def _print_entry_result(
    entry: ExperimentEntry,
    prediction: str,
    judge_result: JudgeResult,
    metrics: AggregateMetrics,
) -> None:
    print("-----------------")
    print(entry.question)
    print(entry.gold)
    print("-----------------")
    print(prediction)
    print("-----------------")
    print(f"judge score: {judge_result.score}")
    print(f"judge explanation: {judge_result.explanation}")
    print("-----------------")
    _print_running_metrics(metrics)


async def _judge_prediction(
    judge_llm: Judge,
    semaphore: asyncio.Semaphore,
    entry: ExperimentEntry,
    prediction: str,
) -> JudgeResult:
    async with semaphore:
        return await judge_llm.ajudge(
            question=entry.question,
            gold=entry.gold,
            prediction=prediction,
            context=entry.kg_context_chunks or entry.context_chunks,
        )


async def _drain_pending(
    pending: deque[_PendingJudge],
    metrics: AggregateMetrics,
    *,
    force: bool,
    judgments: dict[str, JudgeResult],
) -> None:
    while pending:
        head = pending[0]
        if not force and not head.task.done():
            break

        pending.popleft()
        judge_result = await head.task
        judgments[head.entry.entry_id] = judge_result
        metrics.update(head.prediction, head.entry.gold, judge_result)
        _print_entry_result(head.entry, head.prediction, judge_result, metrics)


async def evaluate_entries(
    entries: Sequence[ExperimentEntry],
    *,
    predict_entry: Callable[[ExperimentEntry], Awaitable[str]],
    judge_llm: Judge,
) -> EvaluationResult:
    metrics = AggregateMetrics()
    predictions: dict[str, str] = {}
    answers: dict[str, str] = {}
    judge_semaphore = asyncio.Semaphore(GEMINI_JUDGE_CONCURRENCY)
    pending: deque[_PendingJudge] = deque()
    judgments: dict[str, JudgeResult] = {}
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise ValueError("Evaluation requires unique question IDs")

    try:
        for index, entry in enumerate(entries):
            print(f"Testing entry {index + 1}/{len(entries)}")
            prediction = await predict_entry(entry)
            predictions[entry.entry_id] = prediction
            answers[entry.entry_id] = entry.gold
            pending.append(
                _PendingJudge(
                    entry=entry,
                    prediction=prediction,
                    task=asyncio.create_task(
                        _judge_prediction(
                            judge_llm,
                            judge_semaphore,
                            entry,
                            prediction,
                        )
                    ),
                )
            )
            await _drain_pending(pending, metrics, force=False, judgments=judgments)

        await _drain_pending(pending, metrics, force=True, judgments=judgments)
    finally:
        tasks = [item.task for item in pending]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    averages = metrics.averages()

    print(f"Judge: {averages['judge']}")
    print(f"EM: {averages['em']}")
    print(f"F1: {averages['f1']}")
    print(f"PREC: {averages['precision']}")
    print(f"RECALL: {averages['recall']}")
    print(f"JUDGE_DISTRIBUTION: {metrics.judge_distribution()}")

    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
        answers=answers,
        judgments=judgments,
    )
