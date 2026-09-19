from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors

from gatekg.training.label import (
    LLMTrajectoryStateJudge,
    TrajectoryLabel,
    label_rows,
)
from gatekg.utils.config import load_config


class _FakeJudge:
    model_name = "fake-judge"

    def __init__(self):
        self.calls = 0

    def judge_tool_call(self, *, row, tool_call, call_index):
        self.calls += 1
        states = [0, 1, 0]
        return TrajectoryLabel(
            state=states[call_index],
            explanation=f"state {states[call_index]}",
            model_name=self.model_name,
        )


def _row() -> dict:
    return {
        "run_log_path": "run.log",
        "dataset_variant": "base",
        "model_key": "ollama:qwen3.6:27b",
        "entry_id": "entry-a",
        "question": "Question?",
        "gold": "gold",
        "prediction": "wrong",
        "judge_binary": 0,
        "tool_calls": [
            {"tool": "search_nodes", "args_summary": "('a',)", "kwargs_summary": "{}"},
            {
                "tool": "search_memory_facts",
                "args_summary": "('b',)",
                "kwargs_summary": "{}",
            },
            {"tool": "add_memory", "args_summary": "('c',)", "kwargs_summary": "{}"},
        ],
    }


def test_label_rows_uses_mocked_judge_and_right_fill_cache(tmp_path):
    cache_path = tmp_path / "trajectory_cache.json"
    judge = _FakeJudge()

    labeled_rows = label_rows([_row()], judge=judge, cache_path=cache_path)

    assert judge.calls == 3
    calls = labeled_rows[0]["tool_calls"]
    assert [call["trajectory_state_raw"] for call in calls] == [0, 1, 0]
    assert [call["trajectory_state"] for call in calls] == [1, 1, -1]
    assert all(call["trajectory_label_model"] == "fake-judge" for call in calls)
    assert cache_path.is_file()

    cache_payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    assert len(cache_payload) == 3

    second_judge = _FakeJudge()
    second_labeled_rows = label_rows([_row()], judge=second_judge, cache_path=cache_path)
    assert second_judge.calls == 0
    assert [call["trajectory_state"] for call in second_labeled_rows[0]["tool_calls"]] == [
        1,
        1,
        -1,
    ]


def test_llm_trajectory_judge_retries_429(monkeypatch):
    sleeps = []

    class _FakeModels:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=20)

        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise errors.ClientError(
                    429,
                    {
                        "error": {
                            "code": 429,
                            "status": "RESOURCE_EXHAUSTED",
                            "message": "quota",
                        }
                    },
                )
            return SimpleNamespace(text='{"state": 1, "explanation": "ok"}')

    fake_models = _FakeModels()
    judge = object.__new__(LLMTrajectoryStateJudge)
    judge.config = load_config()
    judge.model_name = "fake-model"
    judge.max_retries = 2
    judge.retry_delay_seconds = 0.5
    judge.client = SimpleNamespace(models=fake_models)

    monkeypatch.setattr("gatekg.training.label.time.sleep", sleeps.append)

    label = judge.judge_tool_call(
        row=_row(),
        tool_call=_row()["tool_calls"][0],
        call_index=0,
    )

    assert label.state == 1
    assert fake_models.calls == 2
    assert sleeps == [0.5]


def test_llm_trajectory_judge_retries_empty_response(monkeypatch):
    sleeps = []

    class _FakeModels:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=20)

        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(text=None)
            return SimpleNamespace(text='{"state": -1, "explanation": "wrong path"}')

    fake_models = _FakeModels()
    judge = object.__new__(LLMTrajectoryStateJudge)
    judge.config = load_config()
    judge.model_name = "fake-model"
    judge.max_retries = 2
    judge.retry_delay_seconds = 0.25
    judge.client = SimpleNamespace(models=fake_models)

    monkeypatch.setattr("gatekg.training.label.time.sleep", sleeps.append)

    label = judge.judge_tool_call(
        row=_row(),
        tool_call=_row()["tool_calls"][0],
        call_index=0,
    )

    assert label.state == -1
    assert label.fallback is False
    assert fake_models.calls == 2
    assert sleeps == [0.25]


def test_llm_trajectory_judge_raises_after_retries(monkeypatch):
    sleeps = []

    class _FakeModels:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=20)

        def generate_content(self, **kwargs):
            return SimpleNamespace(text="")

    judge = object.__new__(LLMTrajectoryStateJudge)
    judge.config = load_config()
    judge.model_name = "fake-model"
    judge.max_retries = 2
    judge.retry_delay_seconds = 0.25
    judge.client = SimpleNamespace(models=_FakeModels())

    monkeypatch.setattr("gatekg.training.label.time.sleep", sleeps.append)

    with pytest.raises(ValueError, match="empty response"):
        judge.judge_tool_call(row=_row(), tool_call=_row()["tool_calls"][0], call_index=0)
    assert sleeps == [0.25]


def test_label_rows_cannot_cache_a_judge_failure(tmp_path):
    from unittest.mock import Mock

    judge = Mock(model_name="offline")
    judge.judge_tool_call.side_effect = ConnectionError("outage")
    cache = tmp_path / "cache.json"
    with pytest.raises(ConnectionError, match="outage"):
        label_rows([_row()], judge=judge, cache_path=cache)
    assert not cache.exists()
