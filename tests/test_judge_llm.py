import pytest

from gatekg.api.judge_llm import JudgeLLM


def test_parse_response_maps_binary_score_and_preserves_explanation():
    judge = JudgeLLM.__new__(JudgeLLM)
    result = judge._parse_response('{"score": 1, "explanation": "Correct enough."}')
    assert result.score == 1
    assert result.explanation == "Correct enough."


def test_parse_response_fills_empty_explanation():
    judge = JudgeLLM.__new__(JudgeLLM)
    result = judge._parse_response('{"score": 0, "explanation": ""}')
    assert result.score == 0
    assert result.explanation == "No explanation provided."


@pytest.mark.parametrize("score", [-1, 0.5, 2, 9])
def test_parse_response_rejects_nonbinary_scores(score):
    import json

    judge = JudgeLLM.__new__(JudgeLLM)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        judge._parse_response(json.dumps({"score": score, "explanation": "Invalid score."}))


def test_exhausted_judge_propagates_failure(monkeypatch):
    from unittest.mock import Mock

    import gatekg.api.judge_llm as module

    judge = JudgeLLM.__new__(JudgeLLM)
    operation = Mock(side_effect=ConnectionError("offline"))
    monkeypatch.setattr(judge, "unsafe_judge", operation)
    monkeypatch.setattr(module, "RETRY_DELAY_SECONDS", 0)
    with pytest.raises(ConnectionError, match="offline"):
        judge.judge("q", "g", "p")
    assert operation.call_count == module.MAX_RETRIES


def test_long_judge_input_preserves_required_fields(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import gatekg.api.judge_llm as module

    fields = dict(question="Question marker", gold="Gold marker", prediction="Prediction marker")
    fixed = module._judge_prompt(**fields, context=None)
    # Fake tokenizer counts UTF-8 bytes, so a character estimate would overflow.
    limit = len(fixed.encode()) + 60
    monkeypatch.setattr(module, "config", SimpleNamespace(input_token_cap_for=lambda _: limit))
    models = Mock()
    models.count_tokens.side_effect = lambda **kw: SimpleNamespace(
        total_tokens=len(kw["contents"].encode())
    )
    models.generate_content.return_value.text = '{"score":1,"explanation":"ok"}'
    judge = JudgeLLM.__new__(JudgeLLM)
    judge.model_name, judge.client = "offline", SimpleNamespace(models=models)
    result = judge.unsafe_judge(**fields, context="猫" * 500)
    prompt = models.generate_content.call_args.kwargs["contents"]
    assert len(prompt.encode()) <= limit
    assert all(field in prompt for field in [*fields.values(), "Scoring rubric:", "fully correct"])
    assert 0 < prompt.count("猫") == 500 - result.context_chars_omitted
    with pytest.raises(ValueError, match="Required judge fields"):
        judge.unsafe_judge("猫" * 1000, "g", "p")
    assert models.generate_content.call_count == 1
