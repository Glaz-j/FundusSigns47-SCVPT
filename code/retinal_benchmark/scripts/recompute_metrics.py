"""Recompute evaluation metrics from a saved benchmark checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retinal_benchmark.config import load_yaml
from retinal_benchmark.models import create_model
from retinal_benchmark.training import build_dataloaders, evaluate


def recompute(run_dir: Path, split: str, checkpoint_name: str) -> dict:
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    data_cfg = load_yaml(run_meta["data_config"])
    model_cfg = load_yaml(run_meta["model_config"])
    train_cfg = load_yaml(run_meta["train_config"])

    _, loaders = build_dataloaders(data_cfg, train_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_cfg["name"],
        model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
    )
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    metrics = evaluate(model, loaders[split], device)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    metrics = recompute(args.run_dir, args.split, args.checkpoint)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.write:
        out_name = f"{args.split}_metrics.json"
        (args.run_dir / out_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
