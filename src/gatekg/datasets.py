"""Dataset registry and adapters for question-answering experiments."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from gatekg.paths import RuntimePaths, runtime_paths
from gatekg.utils.eval_runner import ExperimentEntry, load_hotpotqa_entries
from gatekg.utils.text import segment_long_text


class DatasetFormat(StrEnum):
    AUTO = "auto"
    QA_JSONL = "qa_jsonl"
    HOTPOTQA = "hotpotqa"


@dataclass(frozen=True, slots=True)
class FieldMapping:
    identifier: str = "id"
    question: str = "question"
    answer: str = "answer"
    context: str = "context"
    kg_context: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    path: Path
    format: DatasetFormat = DatasetFormat.AUTO
    fields: FieldMapping = FieldMapping()
    segment_context: bool = True
    chunk_words: int = 1_000
    overlap_ratio: float = 0.25

    def with_path(self, path: str | Path) -> DatasetSpec:
        return replace(self, path=Path(path).expanduser().resolve())


def builtin_datasets(paths: RuntimePaths | None = None) -> dict[str, DatasetSpec]:
    resolved = paths or runtime_paths()
    return {
        "hotpotqa": DatasetSpec("hotpotqa", resolved.data / "hotpotqa" / "qa.jsonl"),
        "hotpotqa_raw": DatasetSpec(
            "hotpotqa_raw",
            resolved.data / "hotpotqa" / "clean" / "validation.json",
            DatasetFormat.HOTPOTQA,
        ),
        "loogle": DatasetSpec("loogle", resolved.data / "loogle" / "longdep_qa.jsonl"),
        "loogle_v2": DatasetSpec("loogle_v2", resolved.data / "loogle_v2" / "qa.jsonl"),
        "two_wiki_multihopqa": DatasetSpec(
            "two_wiki_multihopqa", resolved.data / "two_wiki_multihopqa" / "qa.jsonl"
        ),
        "squad": DatasetSpec("squad", resolved.data / "squad" / "qa.jsonl"),
        "novelhopqa": DatasetSpec("novelhopqa", resolved.data / "novelhopqa" / "qa.jsonl"),
        "mquake": DatasetSpec("mquake", resolved.data / "mquake" / "qa.jsonl"),
    }


def resolve_dataset(
    value: str | Path,
    *,
    format: DatasetFormat | str = DatasetFormat.AUTO,
    segment_context: bool | None = None,
    chunk_words: int | None = None,
    overlap_ratio: float | None = None,
    fields: FieldMapping | None = None,
) -> DatasetSpec:
    """Resolve a registered dataset name or an arbitrary local file."""

    text = str(value)
    registry = builtin_datasets()
    if text in registry:
        spec = registry[text]
    else:
        path = Path(text).expanduser().resolve()
        spec = DatasetSpec(name=path.stem, path=path)

    resolved_format = DatasetFormat(format)
    return replace(
        spec,
        format=resolved_format if resolved_format is not DatasetFormat.AUTO else spec.format,
        segment_context=(spec.segment_context if segment_context is None else segment_context),
        chunk_words=spec.chunk_words if chunk_words is None else chunk_words,
        overlap_ratio=spec.overlap_ratio if overlap_ratio is None else overlap_ratio,
        fields=fields or spec.fields,
    )


def detect_format(path: Path) -> DatasetFormat:
    if path.suffix.lower() == ".jsonl":
        return DatasetFormat.QA_JSONL
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot detect dataset format for {path}: {exc}") from exc
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, Mapping) and isinstance(first.get("context"), Mapping):
            return DatasetFormat.HOTPOTQA
    raise ValueError(f"Cannot detect dataset format for {path}. Use --format qa_jsonl or hotpotqa.")


def load_entries(spec: DatasetSpec, *, limit: int = 0) -> list[ExperimentEntry]:
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if spec.chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not spec.path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {spec.path}. Supply a file path or populate GATEKG_DATA_DIR."
        )

    format = detect_format(spec.path) if spec.format is DatasetFormat.AUTO else spec.format
    if format is DatasetFormat.HOTPOTQA:
        return load_hotpotqa_entries(str(spec.path), limit)
    if format is not DatasetFormat.QA_JSONL:
        raise ValueError(f"Unsupported dataset format: {format}")
    return _load_qa_jsonl(spec, limit=limit)


def _read_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield line_number, row


def _required(row: Mapping[str, Any], key: str, *, path: Path, line: int) -> Any:
    if key not in row:
        raise ValueError(f"Missing `{key}` at {path}:{line}")
    return row[key]


def _answer_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0])
    return str(value)


def _context_chunks(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    raise TypeError(f"Context must be a string or list of strings, got {type(value).__name__}")


def _prepare_context(value: Any, spec: DatasetSpec) -> list[str]:
    chunks = _context_chunks(value)
    if not spec.segment_context:
        return chunks
    return [
        segment
        for chunk in chunks
        for segment in segment_long_text(
            chunk,
            target_words=spec.chunk_words,
            overlap_ratio=spec.overlap_ratio,
        )
    ]


def _load_qa_jsonl(spec: DatasetSpec, *, limit: int) -> list[ExperimentEntry]:
    entries: list[ExperimentEntry] = []
    fields = spec.fields
    for line_number, row in _read_jsonl(spec.path):
        question = str(_required(row, fields.question, path=spec.path, line=line_number))
        answer = _answer_text(_required(row, fields.answer, path=spec.path, line=line_number))
        raw_context = _required(row, fields.context, path=spec.path, line=line_number)
        context_chunks = _prepare_context(raw_context, spec)
        raw_kg_context = row.get(fields.kg_context) if fields.kg_context else None
        kg_context_chunks = (
            _prepare_context(raw_kg_context, spec) if raw_kg_context is not None else None
        )
        entry_id = str(row.get(fields.identifier, line_number - 1))
        entries.append(
            ExperimentEntry(
                entry_id=entry_id,
                question=question,
                gold=answer,
                context_chunks=context_chunks,
                kg_context_chunks=kg_context_chunks,
            )
        )
        if limit and len(entries) >= limit:
            break
    if not entries:
        raise ValueError(f"Dataset contains no entries: {spec.path}")
    return entries
