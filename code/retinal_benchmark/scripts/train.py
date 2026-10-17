#!/usr/bin/env python3
"""Direct training entrypoint without benchmark wrapper."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinal_benchmark.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = PROJECT_ROOT / "experiments" / "logs" / f"manual__{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_training(args.data, args.model, args.train, args.eval, str(run_dir))
    print(json.dumps({"run_dir": str(run_dir), "test_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
