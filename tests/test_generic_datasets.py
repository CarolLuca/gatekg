from __future__ import annotations

import json

import pytest

from gatekg.datasets import DatasetFormat, DatasetSpec, FieldMapping, load_entries


def test_generic_jsonl_supports_field_mapping_and_context_lists(tmp_path):
    dataset_path = tmp_path / "custom.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "key": "row-a",
                "query": "Which token is present?",
                "target": ["alpha", "alternative"],
                "evidence": ["alpha beta", "gamma delta"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = DatasetSpec(
        name="custom",
        path=dataset_path,
        format=DatasetFormat.QA_JSONL,
        fields=FieldMapping(
            identifier="key",
            question="query",
            answer="target",
            context="evidence",
        ),
        segment_context=False,
    )

    entries = load_entries(spec)

    assert entries[0].entry_id == "row-a"
    assert entries[0].gold == "alpha"
    assert entries[0].context_chunks == ["alpha beta", "gamma delta"]


def test_generic_jsonl_reports_the_bad_line(tmp_path):
    dataset_path = tmp_path / "broken.jsonl"
    dataset_path.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")
    spec = DatasetSpec("broken", dataset_path, DatasetFormat.QA_JSONL)

    with pytest.raises(ValueError, match=r"broken\.jsonl:1|Missing `question`"):
        load_entries(spec)


def test_segmented_dataset_loading_is_deterministic(tmp_path):
    dataset_path = tmp_path / "qa.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "a",
                "question": "q",
                "answer": "a",
                "context": "one two three four five six",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = DatasetSpec(
        "qa",
        dataset_path,
        DatasetFormat.QA_JSONL,
        chunk_words=3,
        overlap_ratio=0,
    )

    first = load_entries(spec)
    second = load_entries(spec)

    assert first == second
    assert first[0].context_chunks == ["one two three", "four five six"]
