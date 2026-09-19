"""Download LooGLE v2 dataset and prepare QA-only JSONL for experiments.

Fetches the HuggingFace dataset, keeps only rows where type == "QA",
and writes a JSONL file compatible with the existing eval_runner loaders.

Usage:
    python scripts/download_loogle_v2.py
"""

from __future__ import annotations

import json

from gatekg.paths import runtime_paths

DATA_ROOT = runtime_paths().data
OUTPUT_DIR = DATA_ROOT / "loogle_v2"


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The 'datasets' library is required. Install with: pip install datasets"
        ) from exc

    print("Downloading MuLabPKU/LooGLE-v2 from HuggingFace…")
    ds = load_dataset(
        "MuLabPKU/LooGLE-v2", split="test", revision="1b83bd4b08a3faef45168d7cc1ea8aca785f3d9c"
    )

    # Filter to QA type only
    qa_rows = [row for row in ds if row["type"] == "QA"]
    print(f"Total rows in test split: {len(ds)}")
    print(f"QA-type rows kept:        {len(qa_rows)}")

    if not qa_rows:
        raise SystemExit("No QA-type rows found – aborting.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "qa.jsonl"

    # Group rows by a synthetic doc_id derived from source+task+context hash
    # to allow the existing extended-dataset builder to reuse context grouping.
    import hashlib

    with output_path.open("w", encoding="utf-8") as fh:
        for row in qa_rows:
            ctx_hash = hashlib.md5(row["context"].encode("utf-8")).hexdigest()[:10]
            record = {
                "id": row["id"],
                "doc_id": f"{row['source']}_{ctx_hash}",
                "task": row["task"],
                "type": row["type"],
                "source": row["source"],
                "instruction": row["instruction"],
                "question": row["question"],
                "options": row["options"],
                "answer": row["answer"],
                "context": row["context"],
                "evidence": row["evidence"],
            }
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")

    print(f"Wrote {len(qa_rows)} QA rows → {output_path}")

    # Print domain/task distribution summary
    from collections import Counter

    source_counts = Counter(r["source"] for r in qa_rows)
    task_counts = Counter(r["task"] for r in qa_rows)
    print("\n--- Domain distribution ---")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count}")
    print("\n--- Task distribution ---")
    for task, count in task_counts.most_common():
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
