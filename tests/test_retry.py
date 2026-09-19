import pytest

from gatekg.utils.retry import call_with_retries


def test_retry_returns_success_without_extra_calls(monkeypatch):
    sleeps = []
    calls = []
    monkeypatch.setattr("gatekg.utils.retry.time.sleep", sleeps.append)

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return "answer"

    assert call_with_retries(operation, attempts=3, delay_seconds=0.5) == "answer"
    assert len(calls) == 2
    assert sleeps == [0.5]


def test_retry_exhaustion_propagates_failure(monkeypatch):
    sleeps = []
    monkeypatch.setattr("gatekg.utils.retry.time.sleep", sleeps.append)

    def operation():
        raise RuntimeError("unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        call_with_retries(operation, attempts=2, delay_seconds=1)
    assert sleeps == [1]
