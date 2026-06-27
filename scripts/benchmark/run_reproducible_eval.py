"""Run the repeatable benchmark loop for selected embedding models."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from models import available_slugs


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indexes, evaluate retrieval, evaluate abstention, and write the benchmark report.")
    parser.add_argument("--models", nargs="+", default=["openai_small", "mpnet"], choices=available_slugs())
    parser.add_argument("--modes", nargs="+", default=["dense", "pipeline"], choices=["dense", "pipeline"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/benchmark/build_indexes.py", "--models", *args.models, "--batch-size", str(args.batch_size)])

    for model in args.models:
        for mode in args.modes:
            run([py, "scripts/benchmark/evaluate.py", model, "--mode", mode])
            run([py, "scripts/benchmark/evaluate_abstention.py", model, "--mode", mode])

    run([py, "scripts/benchmark/report.py"])


if __name__ == "__main__":
    main()
