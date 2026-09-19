"""Smoke-test a built wheel outside the source checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="gatekg-wheel-") as temporary:
        workdir = Path(temporary)
        site = workdir / "site"
        with ZipFile(args.wheel.resolve()) as wheel:
            for name in wheel.namelist():
                target = (site / name).resolve()
                if not target.is_relative_to(site):
                    raise ValueError(f"Unsafe wheel member: {name}")
            wheel.extractall(site)
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("GATEKG_")
        }
        environment.update(PYTHONPATH=str(site), GATEKG_HOME=str(workdir))
        commands = [
            ["-m", "gatekg", "--help"],
            ["-m", "gatekg", "run", str(root / "examples/qa.jsonl"), "gatekg", "--dry-run"],
            [str(root / "examples/custom_runner.py")],
        ]
        commands.extend(
            ["-m", f"gatekg.training.{module}", "--help"]
            for module in ("build", "train", "label", "rounds", "interpret")
        )
        for command in commands:
            subprocess.run(
                [sys.executable, *command],
                cwd=workdir,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        print("Wheel CLI, training commands, and offline experiment passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
