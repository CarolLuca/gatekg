from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

import pytest
from yaml import safe_load

from gatekg.cli import main
from gatekg.datasets import DatasetSpec
from gatekg.experiments import ExperimentOptions, run_experiment
from gatekg.paths import policy_artifact_dir
from gatekg.records import manifest_path, read_run_manifest
from gatekg.training.dataset import find_training_logs, load_entries_for_log, parse_run_log
from gatekg.utils.config import PACKAGED_CONFIG_PATH, load_config
from gatekg.utils.eval_runner import ExperimentEntry, evaluate_entries
from gatekg.utils.run_logging import experiment_logging


def _dataset(path: Path) -> Path:
    path.write_text(
        json.dumps({"id": "item", "question": "Where?", "answer": "Here", "context": "Here"})
        + "\n",
        encoding="utf-8",
    )
    return path


def test_manifested_run_is_portable(monkeypatch, tmp_path):
    from gatekg import experiments

    monkeypatch.setenv("GATEKG_HOME", str(tmp_path))
    closed = []

    class Runner:
        async def start(self):
            pass

        async def predict(self, entry):
            logging.getLogger("gatekg.test").info(
                "Preparing context: runtime_chunks=1 kg_chunks=1 "
                "cache_model_key=local:test base_group_id=mcp-cache_test writable=False"
            )
            return entry.context_text

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(experiments, "build_runner", lambda *args: Runner())
    dataset = _dataset(tmp_path / "questions.jsonl")
    report = run_experiment(
        ExperimentOptions(DatasetSpec("custom-domain", dataset), "read_only_rlm", judge="exact")
    )
    assert closed == [True]
    assert report.count == 1
    assert json.loads(Path(report.output_path).read_text())["predictions"] == {"item": "Here"}
    assert find_training_logs(tmp_path / "logs") == [Path(report.log_path)]

    moved_log = tmp_path / "portable.log"
    shutil.copyfile(report.log_path, moved_log)
    shutil.copyfile(manifest_path(report.log_path), manifest_path(moved_log))
    dataset.unlink()
    spec, entries = load_entries_for_log(moved_log)
    assert spec.task_name == "custom-domain"
    assert entries[0].question == "Where?"
    assert read_run_manifest(moved_log)["models"]["judge"] == "exact"

    class EmptyHealthStore:
        def get_baseline_snapshot(self, *args):
            return None

        def list_snapshots(self, **kwargs):
            return []

    rows = parse_run_log(moved_log, store=EmptyHealthStore())
    assert rows[0]["dataset_variant"] == "custom-domain"
    assert rows[0]["judge_binary"] == 1
    assert rows[0]["judge_model_key"] == "exact"


def test_invalid_manifest_reports_schema_error(tmp_path):
    log_path = tmp_path / "run.log"
    manifest_path(log_path).write_text('{"schema_version": "unknown"}')
    with pytest.raises(ValueError, match="Unsupported experiment manifest"):
        read_run_manifest(log_path)


def test_missing_manifests_are_not_training_inputs(tmp_path):
    (tmp_path / "run_unrelated.log").write_text("Unrelated log")
    assert find_training_logs(tmp_path) == []


def test_logging_restores_callers_state_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("GATEKG_LOG_DIR", str(tmp_path))
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    library_level = logging.getLogger("httpx").level
    with (
        pytest.raises(RuntimeError, match="expected"),
        experiment_logging(task_name="domain", runner_name="llm") as log_path,
    ):
        logging.getLogger("gatekg.test").warning("Captured message")
        raise RuntimeError("expected")
    assert root.handlers == handlers
    assert root.level == level
    assert logging.getLogger("httpx").level == library_level
    assert "Captured message" in log_path.read_text()


def test_batch_paths_are_relative_to_manifest(monkeypatch, tmp_path, capsys):
    directory = tmp_path / "batch"
    directory.mkdir()
    _dataset(directory / "qa.jsonl")
    manifest = directory / "runs.yaml"
    manifest.write_text(
        "defaults:\n  dataset: qa.jsonl\n  judge: exact\nruns:\n  - runner: gatekg\n"
    )
    monkeypatch.chdir(tmp_path)
    for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    assert main(["batch", str(manifest), "--dry-run"]) == 0
    assert str(directory / "qa.jsonl") in capsys.readouterr().out


def test_batch_dry_run_rejects_unknown_runner(tmp_path):
    manifest = tmp_path / "runs.yaml"
    manifest.write_text("runs:\n  - dataset: qa.jsonl\n    runner: missing\n")
    assert main(["batch", str(manifest), "--dry-run"]) == 2


def test_policy_directory_resolves_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv("GATEKG_HOME", str(tmp_path))
    monkeypatch.setenv("GATEKG_POLICY_ARTIFACT_DIR", "policies/release")
    assert policy_artifact_dir() == tmp_path / "policies/release"
    assert policy_artifact_dir("custom") == tmp_path / "custom"


def test_default_config_matches_packaged_copy():
    root = Path(__file__).resolve().parents[1]
    assert safe_load((root / "configs/default.yaml").read_text()) == safe_load(
        PACKAGED_CONFIG_PATH.read_text()
    )


def test_kg_override_does_not_change_inference_model(monkeypatch):
    monkeypatch.setenv("GATEKG_PROVIDER", "ollama")
    monkeypatch.setenv("GATEKG_LLM_MODEL", "inference")
    monkeypatch.setenv("GATEKG_KG_PROVIDER", "ollama")
    monkeypatch.setenv("GATEKG_KG_MODEL", "extraction")
    config = load_config(PACKAGED_CONFIG_PATH)
    assert config.llm_model_name == "inference"
    assert config.kg_llm_model_name == "extraction"


def test_evaluation_cancels_pending_judges_on_failure():
    async def run():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingJudge:
            async def ajudge(self, **kwargs):
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

        async def predict(entry):
            if entry.entry_id == "second":
                await started.wait()
                raise RuntimeError("prediction failed")
            return "answer"

        entries = [
            ExperimentEntry("first", "q", "a", ["context"]),
            ExperimentEntry("second", "q", "a", ["context"]),
        ]
        with pytest.raises(RuntimeError, match="prediction failed"):
            await evaluate_entries(entries, predict_entry=predict, judge_llm=BlockingJudge())
        assert cancelled.is_set()

    asyncio.run(run())


def test_tool_graph_deletion_requires_scope():
    from gatekg.api.mcp_kg import MCP_KG

    graph = MCP_KG.__new__(MCP_KG)
    graph.active_group_id = None
    with pytest.raises(ValueError, match="Explicit graph groups"):
        asyncio.run(graph.clear_graph())


def test_vllm_command_is_local_and_has_no_forced_quantization(capsys):
    from gatekg.serving import main as serve

    assert serve(["--config", str(PACKAGED_CONFIG_PATH), "--print-command"]) == 0
    command = capsys.readouterr().out
    assert "vllm serve Qwen/Qwen3.5-9B" in command
    assert "--host 127.0.0.1" in command
    assert "--quantization" not in command
