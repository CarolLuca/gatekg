from __future__ import annotations

import json

from gatekg.cli import main
from gatekg.paths import RuntimePaths


def test_runtime_paths_honor_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEKG_HOME", str(tmp_path))
    monkeypatch.setenv("GATEKG_DATA_DIR", "inputs")

    paths = RuntimePaths.from_env()

    assert paths.root == tmp_path
    assert paths.data == tmp_path / "inputs"
    assert paths.outputs == tmp_path / "outputs"


def test_cli_dry_run_loads_custom_dataset(tmp_path, capsys):
    dataset_path = tmp_path / "qa.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "a", "question": "q", "answer": "a", "context": "context"}) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--home",
            str(tmp_path),
            "run",
            str(dataset_path),
            "llm",
            "--judge",
            "exact",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"] == 1
    assert payload["runner"] == "llm"
