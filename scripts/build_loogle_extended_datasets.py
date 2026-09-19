from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TARGET_AVG_TOKENS = [
    50_000,
    100_000,
    150_000,
    200_000,
    225_000,
    250_000,
    500_000,
    750_000,
    855_000,
    875_000,
    1_000_000,
]
ENTRY_LIMIT = 100
BASELINE_AVG_TOKENS = 25_000

from gatekg.paths import runtime_paths

DATA_ROOT = runtime_paths().data
SOURCE_DATASET_PATH = DATA_ROOT / "loogle" / "longdep_qa.jsonl"
OUTPUT_DIR = DATA_ROOT / "loogle"


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class DocumentGroup:
    doc_id: str
    rows: list[JsonDict]
    context: str
    context_word_count: int
    source_index: int


@dataclass(frozen=True)
class ExtendedContext:
    initial_context: str
    extended_context: str
    extended_doc_ids: list[str]
    approx_initial_tokens: int
    approx_extended_tokens: int


@dataclass(frozen=True)
class DatasetSummary:
    target_avg_tokens: int
    rows_written: int
    source_doc_groups: int
    avg_approx_extended_tokens: float
    output_path: Path


def load_jsonl_rows(dataset_path: Path) -> list[JsonDict]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def estimate_tokens_from_word_count(word_total: int, tokens_per_word: float) -> int:
    return round(word_total * tokens_per_word)


def estimate_tokens(text: str, tokens_per_word: float) -> int:
    return estimate_tokens_from_word_count(word_count(text), tokens_per_word)


def compute_tokens_per_word(
    rows: Sequence[JsonDict],
    *,
    baseline_avg_tokens: float = BASELINE_AVG_TOKENS,
) -> float:
    if not rows:
        raise ValueError("Cannot compute token approximation from an empty dataset.")

    avg_source_row_words = sum(word_count(row["context"]) for row in rows) / len(rows)
    if avg_source_row_words <= 0:
        raise ValueError("Average source row word count must be positive.")

    return baseline_avg_tokens / avg_source_row_words


def group_rows_by_doc_id(rows: Sequence[JsonDict]) -> list[DocumentGroup]:
    groups: list[DocumentGroup] = []
    current_doc_id: str | None = None
    current_context: str | None = None
    current_rows: list[JsonDict] = []

    def finalize_group() -> None:
        if current_doc_id is None or current_context is None or not current_rows:
            return
        groups.append(
            DocumentGroup(
                doc_id=current_doc_id,
                rows=list(current_rows),
                context=current_context,
                context_word_count=word_count(current_context),
                source_index=len(groups),
            )
        )

    for row in rows:
        doc_id = row["doc_id"]
        context = row["context"]

        if current_doc_id is None:
            current_doc_id = doc_id
            current_context = context
            current_rows = [row]
            continue

        if doc_id != current_doc_id:
            finalize_group()
            current_doc_id = doc_id
            current_context = context
            current_rows = [row]
            continue

        if context != current_context:
            raise ValueError(
                f"Rows for doc_id {doc_id!r} have inconsistent contexts within the same group."
            )
        current_rows.append(row)

    finalize_group()
    return groups


def select_output_groups(
    groups: Sequence[DocumentGroup],
    *,
    entry_limit: int,
) -> list[DocumentGroup]:
    if entry_limit <= 0:
        return list(groups)

    selected_groups: list[DocumentGroup] = []
    emitted_rows = 0
    for group in groups:
        selected_groups.append(group)
        emitted_rows += len(group.rows)
        if emitted_rows >= entry_limit:
            break
    return selected_groups


def source_group_fits_token_cap(
    group: DocumentGroup,
    *,
    target_avg_tokens: int,
    tokens_per_word: float,
) -> bool:
    max_total_words = max_word_count_for_token_cap(target_avg_tokens, tokens_per_word)
    return group.context_word_count <= max_total_words


def select_output_groups_for_target(
    groups: Sequence[DocumentGroup],
    *,
    entry_limit: int,
    target_avg_tokens: int,
    tokens_per_word: float,
) -> list[DocumentGroup]:
    if entry_limit <= 0:
        return [
            group
            for group in groups
            if source_group_fits_token_cap(
                group,
                target_avg_tokens=target_avg_tokens,
                tokens_per_word=tokens_per_word,
            )
        ]

    selected_groups: list[DocumentGroup] = []
    emitted_rows = 0
    for group in groups:
        if not source_group_fits_token_cap(
            group,
            target_avg_tokens=target_avg_tokens,
            tokens_per_word=tokens_per_word,
        ):
            continue

        selected_groups.append(group)
        emitted_rows += len(group.rows)
        if emitted_rows >= entry_limit:
            break
    return selected_groups


def max_word_count_for_token_cap(target_tokens: int, tokens_per_word: float) -> int:
    if target_tokens < 0:
        raise ValueError("Target token cap must be non-negative.")
    if tokens_per_word <= 0:
        raise ValueError("tokens_per_word must be positive.")

    word_cap = int(target_tokens / tokens_per_word) + 2
    while (
        word_cap > 0 and estimate_tokens_from_word_count(word_cap, tokens_per_word) > target_tokens
    ):
        word_cap -= 1
    return word_cap


def select_donor_groups_close_to_token_cap(
    groups: Sequence[DocumentGroup],
    *,
    source_group_index: int,
    max_total_words: int,
) -> list[DocumentGroup]:
    source_group = groups[source_group_index]
    remaining_word_budget = max_total_words - source_group.context_word_count
    if remaining_word_budget <= 0:
        return []

    donor_groups = [
        groups[(source_group_index + offset) % len(groups)] for offset in range(1, len(groups))
    ]

    # Bitset DP keeps subset selection exact while remaining deterministic.
    reachable_word_totals = 1
    reachable_states = [reachable_word_totals]
    mask = (1 << (remaining_word_budget + 1)) - 1
    for donor_group in donor_groups:
        reachable_word_totals |= reachable_word_totals << donor_group.context_word_count
        reachable_word_totals &= mask
        reachable_states.append(reachable_word_totals)

    best_extra_words = reachable_states[-1].bit_length() - 1

    selected_reversed: list[DocumentGroup] = []
    current_extra_words = best_extra_words
    for donor_group, previous_state in zip(
        reversed(donor_groups),
        reversed(reachable_states[:-1]),
        strict=True,
    ):
        if (previous_state >> current_extra_words) & 1:
            continue

        donor_words = donor_group.context_word_count
        if current_extra_words >= donor_words and (
            (previous_state >> (current_extra_words - donor_words)) & 1
        ):
            selected_reversed.append(donor_group)
            current_extra_words -= donor_words
            continue

        raise RuntimeError("Failed to reconstruct donor selection for the token cap.")

    selected_reversed.reverse()
    return selected_reversed


def build_extended_context(
    groups: Sequence[DocumentGroup],
    *,
    source_group_index: int,
    target_avg_tokens: int,
    tokens_per_word: float,
) -> ExtendedContext:
    if not groups:
        raise ValueError("Cannot build extended context without document groups.")

    source_group = groups[source_group_index]
    approx_initial_tokens = estimate_tokens_from_word_count(
        source_group.context_word_count,
        tokens_per_word,
    )
    max_total_words = max_word_count_for_token_cap(target_avg_tokens, tokens_per_word)
    if source_group.context_word_count > max_total_words:
        raise ValueError(
            f"Source doc_id {source_group.doc_id!r} already exceeds token cap {target_avg_tokens}."
        )

    donor_groups = select_donor_groups_close_to_token_cap(
        groups,
        source_group_index=source_group_index,
        max_total_words=max_total_words,
    )
    extended_doc_ids = [source_group.doc_id, *[group.doc_id for group in donor_groups]]
    context_parts = [source_group.context, *[group.context for group in donor_groups]]
    total_words = source_group.context_word_count + sum(
        group.context_word_count for group in donor_groups
    )
    approx_extended_tokens = estimate_tokens_from_word_count(total_words, tokens_per_word)

    return ExtendedContext(
        initial_context=source_group.context,
        extended_context="\n\n".join(context_parts),
        extended_doc_ids=extended_doc_ids,
        approx_initial_tokens=approx_initial_tokens,
        approx_extended_tokens=approx_extended_tokens,
    )


def build_output_row(
    row: JsonDict,
    *,
    extended_context: ExtendedContext,
    target_avg_tokens: int,
) -> JsonDict:
    output_row = dict(row)
    output_row["initial_context"] = extended_context.initial_context
    output_row["extended_context"] = extended_context.extended_context
    output_row["target_avg_tokens"] = target_avg_tokens
    output_row["approx_initial_tokens"] = extended_context.approx_initial_tokens
    output_row["approx_extended_tokens"] = extended_context.approx_extended_tokens
    output_row["extended_doc_ids"] = list(extended_context.extended_doc_ids)
    return output_row


def format_target_suffix(target_avg_tokens: int) -> str:
    if target_avg_tokens % 1_000_000 == 0:
        return f"{target_avg_tokens // 1_000_000}m"
    if target_avg_tokens % 1_000 == 0:
        return f"{target_avg_tokens // 1_000}k"
    return str(target_avg_tokens)


def build_output_path(output_dir: Path, target_avg_tokens: int) -> Path:
    suffix = format_target_suffix(target_avg_tokens)
    return output_dir / f"longdep_qa_extended_{suffix}.jsonl"


def write_dataset_for_target(
    groups: Sequence[DocumentGroup],
    *,
    selected_groups: Sequence[DocumentGroup],
    target_avg_tokens: int,
    tokens_per_word: float,
    output_path: Path,
) -> DatasetSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    total_approx_extended_tokens = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for group in selected_groups:
            extended_context = build_extended_context(
                groups,
                source_group_index=group.source_index,
                target_avg_tokens=target_avg_tokens,
                tokens_per_word=tokens_per_word,
            )
            for row in group.rows:
                output_row = build_output_row(
                    row,
                    extended_context=extended_context,
                    target_avg_tokens=target_avg_tokens,
                )
                json.dump(output_row, handle)
                handle.write("\n")
                rows_written += 1
                total_approx_extended_tokens += extended_context.approx_extended_tokens

    avg_approx_extended_tokens = (
        total_approx_extended_tokens / rows_written if rows_written else 0.0
    )
    return DatasetSummary(
        target_avg_tokens=target_avg_tokens,
        rows_written=rows_written,
        source_doc_groups=len(selected_groups),
        avg_approx_extended_tokens=avg_approx_extended_tokens,
        output_path=output_path,
    )


def print_summary(summary: DatasetSummary) -> None:
    print(
        f"target_avg_tokens={summary.target_avg_tokens} "
        f"rows_written={summary.rows_written} "
        f"source_doc_groups={summary.source_doc_groups} "
        f"avg_approx_extended_tokens={summary.avg_approx_extended_tokens:.2f} "
        f"output_path={summary.output_path}"
    )


def generate_datasets(
    *,
    source_dataset_path: Path = SOURCE_DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
    target_avg_tokens: Sequence[int] = TARGET_AVG_TOKENS,
    entry_limit: int = ENTRY_LIMIT,
    baseline_avg_tokens: float = BASELINE_AVG_TOKENS,
) -> list[DatasetSummary]:
    rows = load_jsonl_rows(source_dataset_path)
    groups = group_rows_by_doc_id(rows)
    tokens_per_word = compute_tokens_per_word(
        rows,
        baseline_avg_tokens=baseline_avg_tokens,
    )

    summaries: list[DatasetSummary] = []
    for target in target_avg_tokens:
        selected_groups = select_output_groups_for_target(
            groups,
            entry_limit=entry_limit,
            target_avg_tokens=target,
            tokens_per_word=tokens_per_word,
        )
        output_path = build_output_path(output_dir, target)
        summary = write_dataset_for_target(
            groups,
            selected_groups=selected_groups,
            target_avg_tokens=target,
            tokens_per_word=tokens_per_word,
            output_path=output_path,
        )
        print_summary(summary)
        summaries.append(summary)
    return summaries


def main() -> None:
    generate_datasets()


if __name__ == "__main__":
    main()
