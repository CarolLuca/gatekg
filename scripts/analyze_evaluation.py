"""Question-level bootstrap intervals and human/judge agreement on local evaluations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gatekg.evaluation.hotpot import f1_score


def bootstrap(report: dict, resamples: int, seed: int) -> dict:
    judgments = report["judgments"]
    questions = sorted(judgments)
    if (
        not questions
        or set(questions) != set(report["predictions"])
        or set(questions) != set(report["answers"])
    ):
        raise ValueError("A bootstrap sample requires one judgment and answer per prediction")
    if resamples < 1:
        raise ValueError("Resamples must be positive")
    scores = np.array(
        [
            [judgments[q]["score"], f1_score(report["predictions"][q], report["answers"][q])[0]]
            for q in questions
        ]
    )
    rng = np.random.default_rng(seed)
    means = np.array(
        [scores[rng.integers(len(scores), size=len(scores))].mean(axis=0) for _ in range(resamples)]
    )
    intervals = np.quantile(means, [0.025, 0.975], axis=0)
    return {
        "questions": len(scores),
        "resamples": resamples,
        "seed": seed,
        **{
            name: {"mean": scores[:, i].mean(), "ci95": intervals[:, i].tolist()}
            for i, name in enumerate(("judge", "f1"))
        },
    }


def audit(rows: list[dict]) -> dict:
    pairs = [(int(row["judge"]), int(row["human"])) for row in rows]
    if not pairs or any(a not in (0, 1) or b not in (0, 1) for a, b in pairs):
        raise ValueError("Audit CSV needs paired binary judge and human ratings")
    n = len(pairs)
    agreements = sum(a == b for a, b in pairs)
    expected = (
        sum(
            sum(a == label for a, b in pairs) * sum(b == label for a, b in pairs)
            for label in (0, 1)
        )
        / n**2
    )
    return {
        "pairs": n,
        "agreements": agreements,
        "agreement": agreements / n,
        "cohen_kappa": (agreements / n - expected) / (1 - expected) if expected < 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    boot = commands.add_parser("bootstrap")
    boot.add_argument("report", type=Path)
    boot.add_argument("--resamples", type=int, default=10000)
    boot.add_argument("--seed", type=int, default=42)
    commands.add_parser("audit").add_argument("ratings", type=Path)
    args = parser.parse_args()
    if args.command == "bootstrap":
        result = bootstrap(json.loads(args.report.read_text()), args.resamples, args.seed)
    else:
        with args.ratings.open(newline="", encoding="utf-8") as handle:
            result = audit(list(csv.DictReader(handle)))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
