from __future__ import annotations

import json
from pathlib import Path

from scripts import build_loogle_extended_datasets as builder


def _make_context(prefix: str, size: int) -> str:
    return " ".join(f"{prefix}_{index}" for index in range(size))


def _make_rows() -> list[dict[str, object]]:
    group_specs = [
        ("doc-a", 2, 3),
        ("doc-b", 1, 4),
        ("doc-c", 2, 5),
        ("doc-d", 1, 6),
    ]

    rows: list[dict[str, object]] = []
    for doc_id, row_count, context_size in group_specs:
        context = _make_context(doc_id, context_size)
        for index in range(row_count):
            rows.append(
                {
                    "id": f"{doc_id}-{index}",
                    "doc_id": doc_id,
                    "task": "longdep_qa",
                    "context": context,
                    "question": f"question-{doc_id}-{index}",
                    "answer": f"answer-{doc_id}-{index}",
                    "evidence": [f"evidence-{doc_id}-{index}"],
                    "title": f"title-{doc_id}",
                }
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


def test_group_rows_by_doc_id_preserves_order_and_contiguity():
    groups = builder.group_rows_by_doc_id(_make_rows())

    assert [group.doc_id for group in groups] == ["doc-a", "doc-b", "doc-c", "doc-d"]
    assert [group.source_index for group in groups] == [0, 1, 2, 3]
    assert [len(group.rows) for group in groups] == [2, 1, 2, 1]
    assert [row["id"] for row in groups[2].rows] == ["doc-c-0", "doc-c-1"]


def test_select_output_groups_keeps_last_group_intact_when_limit_is_crossed():
    groups = builder.group_rows_by_doc_id(_make_rows())

    selected_groups = builder.select_output_groups(groups, entry_limit=4)

    assert [group.doc_id for group in selected_groups] == ["doc-a", "doc-b", "doc-c"]
    assert sum(len(group.rows) for group in selected_groups) == 5


def test_select_output_groups_for_target_skips_oversized_sources_and_keeps_order():
    rows: list[dict[str, object]] = []
    for doc_id, row_count, context_size in [
        ("doc-a", 2, 3),
        ("doc-b", 1, 8),
        ("doc-c", 2, 4),
        ("doc-d", 1, 5),
    ]:
        context = _make_context(doc_id, context_size)
        for index in range(row_count):
            rows.append(
                {
                    "id": f"{doc_id}-{index}",
                    "doc_id": doc_id,
                    "task": "longdep_qa",
                    "context": context,
                    "question": f"question-{doc_id}-{index}",
                    "answer": f"answer-{doc_id}-{index}",
                    "evidence": [f"evidence-{doc_id}-{index}"],
                    "title": f"title-{doc_id}",
                }
            )

    groups = builder.group_rows_by_doc_id(rows)

    selected_groups = builder.select_output_groups_for_target(
        groups,
        entry_limit=4,
        target_avg_tokens=5,
        tokens_per_word=1.0,
    )

    assert [group.doc_id for group in selected_groups] == ["doc-a", "doc-c"]
    assert sum(len(group.rows) for group in selected_groups) == 4


def test_build_extended_context_is_deterministic_and_cycles_without_repeats():
    groups = builder.group_rows_by_doc_id(_make_rows())

    first = builder.build_extended_context(
        groups,
        source_group_index=2,
        target_avg_tokens=14,
        tokens_per_word=1.0,
    )
    second = builder.build_extended_context(
        groups,
        source_group_index=2,
        target_avg_tokens=14,
        tokens_per_word=1.0,
    )

    assert first == second
    assert first.extended_doc_ids == ["doc-c", "doc-d", "doc-a"]
    assert len(first.extended_doc_ids) == len(set(first.extended_doc_ids))
    assert first.approx_initial_tokens == 5
    assert first.approx_extended_tokens == 14
    assert first.extended_context == "\n\n".join(
        [
            groups[2].context,
            groups[3].context,
            groups[0].context,
        ]
    )


def test_build_extended_context_picks_best_fit_under_token_cap():
    groups = builder.group_rows_by_doc_id(_make_rows())

    extended = builder.build_extended_context(
        groups,
        source_group_index=0,
        target_avg_tokens=11,
        tokens_per_word=1.0,
    )

    assert extended.extended_doc_ids == ["doc-a", "doc-d"]
    assert extended.approx_initial_tokens == 3
    assert extended.approx_extended_tokens == 9
    assert extended.extended_context == "\n\n".join(
        [
            groups[0].context,
            groups[3].context,
        ]
    )


def test_build_output_row_preserves_original_fields_and_adds_metadata():
    groups = builder.group_rows_by_doc_id(_make_rows())
    extended_context = builder.build_extended_context(
        groups,
        source_group_index=0,
        target_avg_tokens=7,
        tokens_per_word=1.0,
    )

    source_row = groups[0].rows[0]
    output_row = builder.build_output_row(
        source_row,
        extended_context=extended_context,
        target_avg_tokens=7,
    )

    assert output_row["id"] == source_row["id"]
    assert output_row["question"] == source_row["question"]
    assert output_row["context"] == source_row["context"]
    assert output_row["initial_context"] == source_row["context"]
    assert output_row["extended_context"] != source_row["context"]
    assert output_row["target_avg_tokens"] == 7
    assert output_row["approx_initial_tokens"] == 3
    assert output_row["approx_extended_tokens"] == 7
    assert output_row["extended_doc_ids"] == ["doc-a", "doc-b"]


def test_generate_datasets_writes_jsonl_and_prints_summary(tmp_path, capsys):
    rows = _make_rows()
    source_path = tmp_path / "source.jsonl"
    output_dir = tmp_path / "generated"
    _write_jsonl(source_path, rows)

    avg_row_words = sum(builder.word_count(row["context"]) for row in rows) / len(rows)
    summaries = builder.generate_datasets(
        source_dataset_path=source_path,
        output_dir=output_dir,
        target_avg_tokens=[10],
        entry_limit=4,
        baseline_avg_tokens=avg_row_words,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.rows_written == 5
    assert summary.source_doc_groups == 3
    assert summary.avg_approx_extended_tokens == 9.2
    assert summary.output_path == output_dir / "longdep_qa_extended_10.jsonl"

    written_rows = builder.load_jsonl_rows(summary.output_path)
    assert len(written_rows) == 5
    assert [row["doc_id"] for row in written_rows] == [
        "doc-a",
        "doc-a",
        "doc-b",
        "doc-c",
        "doc-c",
    ]
    assert all(row["context"] == row["initial_context"] for row in written_rows)
    assert [row["approx_extended_tokens"] for row in written_rows] == [9, 9, 10, 9, 9]
    assert all(row["approx_extended_tokens"] <= 10 for row in written_rows)
    assert all(
        builder.estimate_tokens(row["extended_context"], 1.0) == row["approx_extended_tokens"]
        for row in written_rows
    )

    captured = capsys.readouterr()
    assert "rows_written=5" in captured.out
    assert str(summary.output_path) in captured.out
