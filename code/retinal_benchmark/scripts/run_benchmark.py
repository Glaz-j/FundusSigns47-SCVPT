#!/usr/bin/env python3
"""Run a benchmark configuration with automatic logging layout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinal_benchmark.config import load_yaml
from retinal_benchmark.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Path to benchmark yaml")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark).resolve()
    project_root = PROJECT_ROOT
    cfg = load_yaml(benchmark_path)

    run_name = f"{cfg['dataset_version']}__{cfg['model_name']}__{cfg['seed']}__{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = project_root / "experiments" / "logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "benchmark": str(benchmark_path),
        "dataset_version": cfg["dataset_version"],
        "model_name": cfg["model_name"],
        "seed": cfg["seed"],
        "data_config": str((project_root / cfg["data_config"]).resolve()),
        "model_config": str((project_root / cfg["model_config"]).resolve()),
        "train_config": str((project_root / cfg["train_config"]).resolve()),
        "eval_config": str((project_root / cfg["eval_config"]).resolve()),
    }
    (run_dir / "benchmark_resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    metrics = run_training(
        resolved["data_config"],
        resolved["model_config"],
        resolved["train_config"],
        resolved["eval_config"],
        str(run_dir),
    )
    print(json.dumps({"run_dir": str(run_dir), "test_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
