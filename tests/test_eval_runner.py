import asyncio
import threading

import gatekg.utils.eval_runner as eval_runner
from gatekg.api.judge_llm import JudgeResult


class _DirectModel:
    run_predict_in_worker_thread = False

    def predict(self, prompt: str, question: str) -> int:
        return threading.get_ident()


class _WorkerModel:
    def predict(self, prompt: str) -> int:
        return threading.get_ident()


class _ThreadBoundModel:
    def __init__(self, owner_thread_id: int):
        self.owner_thread_id = owner_thread_id

    def predict(self, prompt: str) -> str:
        if threading.get_ident() != self.owner_thread_id:
            raise RuntimeError("thread-bound resource used from a different thread")
        return "ok"


class _ThreadBoundDirectModel(_ThreadBoundModel):
    run_predict_in_worker_thread = False


class _HookModel:
    run_predict_in_worker_thread = False

    def __init__(self):
        self.hook_calls: list[dict[str, str | None]] = []

    def predict(self, prompt: str, question: str) -> str:
        return "answer"

    async def after_predict(self, *, prompt: str, question: str | None, prediction: str) -> None:
        self.hook_calls.append(
            {
                "prompt": prompt,
                "question": question,
                "prediction": prediction,
            }
        )


async def _manual_to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    result_future = loop.create_future()

    def runner() -> None:
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            loop.call_soon_threadsafe(result_future.set_exception, exc)
        else:
            loop.call_soon_threadsafe(result_future.set_result, result)

    thread = threading.Thread(target=runner)
    thread.start()
    try:
        return await result_future
    finally:
        thread.join()


def test_predict_with_model_runs_opted_out_models_on_event_loop_thread():
    async def run():
        main_thread_id = threading.get_ident()
        result = await eval_runner.predict_with_model(_DirectModel(), "prompt", "question")
        assert result == main_thread_id

    asyncio.run(run())


def test_predict_with_model_uses_worker_thread_by_default():
    async def run():
        original_to_thread = eval_runner.asyncio.to_thread
        main_thread_id = threading.get_ident()
        eval_runner.asyncio.to_thread = _manual_to_thread
        try:
            result = await eval_runner.predict_with_model(_WorkerModel(), "prompt")
        finally:
            eval_runner.asyncio.to_thread = original_to_thread
        assert result != main_thread_id

    asyncio.run(run())


def test_predict_with_model_worker_thread_breaks_thread_bound_resources():
    async def run():
        original_to_thread = eval_runner.asyncio.to_thread
        owner_thread_id = threading.get_ident()
        eval_runner.asyncio.to_thread = _manual_to_thread
        try:
            try:
                await eval_runner.predict_with_model(_ThreadBoundModel(owner_thread_id), "prompt")
            except RuntimeError as exc:
                assert "different thread" in str(exc)
                return
            raise AssertionError("Expected cross-thread failure from worker-thread execution")
        finally:
            eval_runner.asyncio.to_thread = original_to_thread

    asyncio.run(run())


def test_predict_with_model_direct_path_preserves_thread_bound_resources():
    async def run():
        owner_thread_id = threading.get_ident()
        result = await eval_runner.predict_with_model(
            _ThreadBoundDirectModel(owner_thread_id),
            "prompt",
        )
        assert result == "ok"

    asyncio.run(run())


def test_predict_with_model_runs_post_predict_hook():
    async def run():
        model = _HookModel()
        result = await eval_runner.predict_with_model(model, "prompt", "question")
        assert result == "answer"
        assert model.hook_calls == [
            {
                "prompt": "prompt",
                "question": "question",
                "prediction": "answer",
            }
        ]

    asyncio.run(run())


def test_judge_prediction_prefers_small_kg_context_when_available():
    captured = {}

    class _Judge:
        async def ajudge(self, *, question, gold, prediction, context):
            captured["question"] = question
            captured["gold"] = gold
            captured["prediction"] = prediction
            captured["context"] = context
            return "ok"

    async def run():
        result = await eval_runner._judge_prediction(
            _Judge(),
            asyncio.Semaphore(1),
            eval_runner.ExperimentEntry(
                entry_id="entry-1",
                question="question",
                gold="gold",
                context_chunks=["extended-a", "extended-b"],
                kg_context_chunks=["small-a"],
            ),
            "prediction",
        )
        assert result == "ok"

    asyncio.run(run())

    assert captured == {
        "question": "question",
        "gold": "gold",
        "prediction": "prediction",
        "context": ["small-a"],
    }


def test_judge_prediction_falls_back_to_runtime_context_without_kg_context():
    captured = {}

    class _Judge:
        async def ajudge(self, *, question, gold, prediction, context):
            captured["context"] = context
            return "ok"

    async def run():
        result = await eval_runner._judge_prediction(
            _Judge(),
            asyncio.Semaphore(1),
            eval_runner.ExperimentEntry(
                entry_id="entry-1",
                question="question",
                gold="gold",
                context_chunks=["extended-a", "extended-b"],
            ),
            "prediction",
        )
        assert result == "ok"

    asyncio.run(run())

    assert captured["context"] == ["extended-a", "extended-b"]


def test_judge_distribution_is_binary():
    metrics = eval_runner.AggregateMetrics()
    metrics.update("prediction", "gold", JudgeResult(score=1, explanation="ok"))
    metrics.update("prediction", "gold", JudgeResult(score=0, explanation="bad"))

    assert metrics.judge_distribution() == {0: 1, 1: 1}
