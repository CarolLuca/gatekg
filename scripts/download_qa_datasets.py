"""Download and normalize QA datasets for the experiment runners.

Each dataset is written once at its original context length as:
    data/<dataset>/qa.jsonl

Usage:
    python scripts/download_qa_datasets.py
    python scripts/download_qa_datasets.py --datasets squad,hotpotqa --limit 100
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

from gatekg.paths import runtime_paths

DATA_ROOT = runtime_paths().data
DEFAULT_LIMIT = 100
MQUAKE_URL = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd/datasets/MQuAKE-CF-3k-v2.json"
TWO_WIKI_DEV = (
    "hf://datasets/xanhho/2WikiMultihopQA@612bc5039a457880d9e7d84c3b0a4cf154b70e4f/dev.parquet"
)


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' library is required. Install with: pip install datasets"
        ) from exc
    return load_dataset


def _write_jsonl(dataset_name: str, rows: Iterable[dict[str, Any]]) -> Path:
    output_dir = DATA_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qa.jsonl"
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")
            count += 1
    print(f"Wrote {count:>3} rows -> {output_path}")
    return output_path


def _hotpot_context(row: dict[str, Any]) -> str:
    context = row["context"]
    return "\n\n".join(
        f"{title}:\n" + "\n".join(sentences)
        for title, sentences in zip(context["title"], context["sentences"], strict=False)
    )


def _two_wiki_context(row: dict[str, Any]) -> str:
    raw_context = row["context"]
    context = json.loads(raw_context) if isinstance(raw_context, str) else raw_context
    return "\n\n".join(f"{title}:\n" + "\n".join(sentences) for title, sentences in context)


def build_two_wiki(limit: int) -> list[dict[str, Any]]:
    load_dataset = _require_datasets()
    dataset = load_dataset("parquet", data_files={"validation": TWO_WIKI_DEV})["validation"]
    rows: list[dict[str, Any]] = []
    for row in dataset.select(range(min(limit, len(dataset)))):
        rows.append(
            {
                "id": row["_id"],
                "dataset": "two_wiki_multihopqa",
                "source": "xanhho/2WikiMultihopQA",
                "split": "dev",
                "question": row["question"],
                "answer": row["answer"],
                "context": _two_wiki_context(row),
                "metadata": {"type": row.get("type")},
            }
        )
    return rows


def build_hotpotqa(limit: int) -> list[dict[str, Any]]:
    load_dataset = _require_datasets()
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        revision="1908d6afbbead072334abe2965f91bd2709910ab",
    )
    rows: list[dict[str, Any]] = []
    for row in dataset.select(range(min(limit, len(dataset)))):
        rows.append(
            {
                "id": row["id"],
                "dataset": "hotpotqa",
                "source": "hotpotqa/hotpot_qa:distractor",
                "split": "validation",
                "question": row["question"],
                "answer": row["answer"],
                "context": _hotpot_context(row),
                "metadata": {"type": row.get("type"), "level": row.get("level")},
            }
        )
    return rows


def build_squad(limit: int) -> list[dict[str, Any]]:
    load_dataset = _require_datasets()
    dataset = load_dataset("squad", split="validation")
    rows: list[dict[str, Any]] = []
    for row in dataset.select(range(min(limit, len(dataset)))):
        answers = row["answers"]["text"]
        if not answers:
            continue
        rows.append(
            {
                "id": row["id"],
                "dataset": "squad",
                "source": "squad",
                "split": "validation",
                "question": row["question"],
                "answer": answers[0],
                "context": row["context"],
                "metadata": {"title": row.get("title"), "answers": answers},
            }
        )
    return rows


def build_novelhopqa(limit: int) -> list[dict[str, Any]]:
    load_dataset = _require_datasets()
    datasets = load_dataset(
        "abhaygupta1266/novelhopqa", revision="c642bf8ab04edcc23441b179799a0984a8a49eee"
    )
    split_names = sorted(datasets.keys())
    rows: list[dict[str, Any]] = []
    per_split = max(1, limit // len(split_names))
    remainder = limit % len(split_names)
    for split_index, split_name in enumerate(split_names):
        dataset = datasets[split_name]
        split_limit = per_split + (1 if split_index < remainder else 0)
        for index, row in enumerate(dataset.select(range(min(split_limit, len(dataset))))):
            rows.append(
                {
                    "id": f"{split_name}_{index}",
                    "dataset": "novelhopqa",
                    "source": "abhaygupta1266/novelhopqa",
                    "split": split_name,
                    "question": row["question"],
                    "answer": row["answer"],
                    "context": row["context"],
                    "metadata": {
                        "book": row.get("book"),
                        "number_of_hops": row.get("number_of_hops"),
                        "token_count": row.get("token_count"),
                        "keywords": row.get("keywords"),
                    },
                }
            )
    return rows[:limit]


def _mquake_context(row: dict[str, Any]) -> str:
    facts = []
    single_hops = row.get("new_single_hops") or row.get("single_hops") or []
    for hop in single_hops:
        cloze = str(hop.get("cloze") or "").strip()
        answer = str(hop.get("answer") or "").strip()
        question = str(hop.get("question") or "").strip()
        if cloze and answer:
            facts.append(f"{cloze} {answer}.")
        elif question and answer:
            facts.append(f"Q: {question}\nA: {answer}")

    if not facts:
        for rewrite in row.get("requested_rewrite", []):
            prompt = str(rewrite.get("prompt") or "").replace(
                "{}", str(rewrite.get("subject") or "")
            )
            target = rewrite.get("target_new") or rewrite.get("target_true") or {}
            if prompt and target.get("str"):
                facts.append(f"{prompt} {target['str']}.")

    return "\n".join(facts)


def build_mquake(limit: int) -> list[dict[str, Any]]:
    response = requests.get(MQUAKE_URL, timeout=60)
    response.raise_for_status()
    dataset = response.json()
    rows: list[dict[str, Any]] = []
    for row in dataset[:limit]:
        questions = row.get("questions") or []
        answer = row.get("new_answer") or row.get("answer")
        if not questions or not answer:
            continue
        rows.append(
            {
                "id": str(row["case_id"]),
                "dataset": "mquake",
                "source": "princeton-nlp/MQuAKE:MQuAKE-CF-3k-v2",
                "split": "main",
                "question": questions[0],
                "answer": answer,
                "context": _mquake_context(row),
                "metadata": {
                    "answer_alias": row.get("new_answer_alias") or row.get("answer_alias"),
                    "question_variants": questions,
                },
            }
        )
    return rows


BUILDERS = {
    "two_wiki_multihopqa": build_two_wiki,
    "hotpotqa": build_hotpotqa,
    "squad": build_squad,
    "novelhopqa": build_novelhopqa,
    "mquake": build_mquake,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default=",".join(BUILDERS),
        help="Comma-separated dataset keys, or 'all'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum rows to keep per dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    requested = list(BUILDERS) if args.datasets == "all" else args.datasets.split(",")
    for dataset_name in requested:
        dataset_name = dataset_name.strip()
        if dataset_name not in BUILDERS:
            raise SystemExit(
                f"Unknown dataset '{dataset_name}'. Choose from: {', '.join(BUILDERS)}"
            )
        print(f"Preparing {dataset_name} (limit={args.limit})")
        _write_jsonl(dataset_name, BUILDERS[dataset_name](args.limit))


if __name__ == "__main__":
    main()
