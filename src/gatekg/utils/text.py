"""Deterministic text normalization and chunking utilities."""

from __future__ import annotations

import re

TARGET_WORDS = 1_000
MAX_WORDS = 1_400
OVERLAP_RATIO = 0.25


def segment_long_text(
    text: str,
    *,
    target_words: int = TARGET_WORDS,
    max_words: int | None = None,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[str]:
    """Split text into stable, paragraph-aware, overlapping chunks.

    The original implementation constructed an LLM client but deliberately
    disabled the LLM call. This deterministic implementation is faster,
    reproducible, and safe to use while loading datasets offline.
    """

    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")
    resolved_max_words = max_words or max(target_words, int(target_words * 1.4))
    if resolved_max_words < target_words:
        raise ValueError("max_words must be at least target_words")

    normalized = re.sub(r"\n{3,}", "\n\n", str(text).strip())
    if not normalized:
        return []
    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]

    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > resolved_max_words:
            if current:
                chunks.append(current)
                current, current_size = [], 0
            for offset in range(0, len(words), target_words):
                chunks.append(words[offset : offset + target_words])
            continue

        if current and current_size + len(words) > resolved_max_words:
            chunks.append(current)
            current, current_size = [], 0
        current.extend(words)
        current_size += len(words)
        if current_size >= target_words:
            chunks.append(current)
            current, current_size = [], 0
    if current:
        chunks.append(current)

    overlap_words = int(target_words * overlap_ratio)
    result: list[str] = []
    for index, words in enumerate(chunks):
        if index and overlap_words:
            words = chunks[index - 1][-overlap_words:] + words
        result.append(" ".join(words))
    return result
