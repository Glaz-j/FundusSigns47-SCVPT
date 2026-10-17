#!/usr/bin/env python3
"""Run dual-model fusion / cascade experiments from saved benchmark checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retinal_benchmark.config import load_yaml
from retinal_benchmark.models import create_model
from retinal_benchmark.training import build_dataloaders, collect_logits_targets


def compute_macro_auc(targets: np.ndarray, probs: np.ndarray) -> tuple[float, int]:
    per_class_auc = []
    for class_idx in range(targets.shape[1]):
        y_true = targets[:, class_idx]
        y_score = probs[:, class_idx]
        if len(np.unique(y_true)) < 2:
            continue
        try:
            per_class_auc.append(float(roc_auc_score(y_true, y_score)))
        except Exception:
            continue
    if not per_class_auc:
        return 0.0, 0
    return float(np.mean(per_class_auc)), len(per_class_auc)


def evaluate_from_probs(targets: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    metrics = {}
    try:
        metrics["mAP"] = float(average_precision_score(targets, probs, average="macro"))
    except Exception:
        metrics["mAP"] = 0.0
    try:
        metrics["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
        metrics["micro_f1"] = float(f1_score(targets, preds, average="micro", zero_division=0))
    except Exception:
        metrics["macro_f1"] = 0.0
        metrics["micro_f1"] = 0.0
    try:
        metrics["micro_precision"] = float(precision_score(targets, preds, average="micro", zero_division=0))
        metrics["micro_recall"] = float(recall_score(targets, preds, average="micro", zero_division=0))
        metrics["sample_f1"] = float(f1_score(targets, preds, average="samples", zero_division=0))
    except Exception:
        metrics["micro_precision"] = 0.0
        metrics["micro_recall"] = 0.0
        metrics["sample_f1"] = 0.0
    metrics["macro_auc"], metrics["macro_auc_valid_classes"] = compute_macro_auc(targets, probs)
    return metrics


def find_per_class_alphas(
    targets: np.ndarray,
    cand_probs: np.ndarray,
    prec_probs: np.ndarray,
    cand_thresholds: np.ndarray,
    alpha_grid: list[float],
) -> np.ndarray:
    alphas = np.zeros(targets.shape[1], dtype=np.float32)
    cand_mask = (cand_probs >= cand_thresholds.reshape(1, -1)).astype(np.float32)
    for class_idx in range(targets.shape[1]):
        y_true = targets[:, class_idx]
        best_alpha = 0.5
        best_f1 = -1.0
        cand_col = cand_probs[:, class_idx]
        prec_col = prec_probs[:, class_idx]
        mask_col = cand_mask[:, class_idx]
        for alpha in alpha_grid:
            fused = (alpha * cand_col + (1.0 - alpha) * prec_col) * np.maximum(mask_col, (prec_col >= 0.5).astype(np.float32))
            preds = (fused >= 0.5).astype(np.int32)
            score = f1_score(y_true, preds, average="binary", zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_alpha = float(alpha)
        alphas[class_idx] = best_alpha
    return alphas


def load_run_probs(run_dir: Path, split: str, checkpoint_name: str):
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    data_cfg = load_yaml(run_meta["data_config"])
    model_cfg = load_yaml(run_meta["model_config"])
    train_cfg = load_yaml(run_meta["train_config"])
    train_cfg["num_workers"] = 0
    _, loaders = build_dataloaders(data_cfg, train_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_cfg, pretrained=model_cfg.get("pretrained", True))
    state = torch.load(run_dir / "checkpoints" / checkpoint_name, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    logits, targets = collect_logits_targets(model, loaders[split], device)
    probs = 1.0 / (1.0 + np.exp(-logits))
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    threshold_path = run_dir / "thresholds.json"
    if threshold_path.exists():
        thresholds = np.asarray(json.loads(threshold_path.read_text(encoding="utf-8")), dtype=np.float32)
    return probs, targets, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--precision-run", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["hard", "soft", "perclass_soft", "candidate_soft", "uncertainty_soft"],
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--candidate-threshold-source", choices=["run", "default"], default="run")
    args = parser.parse_args()

    cand_probs, targets, cand_thresholds = load_run_probs(args.candidate_run, args.split, args.checkpoint)
    prec_probs, targets2, prec_thresholds = load_run_probs(args.precision_run, args.split, args.checkpoint)
    if not np.array_equal(targets, targets2):
        raise RuntimeError("Candidate and precision runs produced mismatched targets")

    if args.candidate_threshold_source == "default":
        cand_thresholds = np.full_like(cand_thresholds, 0.5)

    cand_mask = (cand_probs >= cand_thresholds.reshape(1, -1)).astype(np.float32)
    prec_mask = (prec_probs >= prec_thresholds.reshape(1, -1)).astype(np.float32)

    if args.mode == "hard":
        final_probs = prec_probs * cand_mask
        final_thresholds = prec_thresholds
    elif args.mode == "soft":
        # Soft candidate-gated geometric mean: keeps both models involved,
        # but still allows candidate thresholds to suppress obvious negatives.
        final_probs = np.sqrt(np.clip(cand_probs, 1e-8, 1.0) * np.clip(prec_probs, 1e-8, 1.0)) * np.maximum(
            cand_mask, prec_mask
        )
        final_thresholds = np.full_like(prec_thresholds, 0.5)
    elif args.mode == "candidate_soft":
        # Only intervene on candidate classes; outside the candidate mask we trust the recall model.
        fused = np.sqrt(np.clip(cand_probs, 1e-8, 1.0) * np.clip(prec_probs, 1e-8, 1.0))
        final_probs = cand_probs * (1.0 - cand_mask) + fused * cand_mask
        final_thresholds = cand_thresholds
    elif args.mode == "uncertainty_soft":
        # Rely more on the precision model when the candidate model is near its decision boundary.
        uncertainty = 1.0 - np.clip(np.abs(cand_probs - cand_thresholds.reshape(1, -1)) / 0.5, 0.0, 1.0)
        fused = (1.0 - uncertainty) * cand_probs + uncertainty * prec_probs
        final_probs = fused * np.maximum(cand_mask, prec_mask)
        final_thresholds = np.full_like(prec_thresholds, 0.5)
    else:
        val_cand_probs, val_targets, val_cand_thresholds = load_run_probs(args.candidate_run, "val", args.checkpoint)
        val_prec_probs, val_targets2, _ = load_run_probs(args.precision_run, "val", args.checkpoint)
        if not np.array_equal(val_targets, val_targets2):
            raise RuntimeError("Candidate and precision validation targets mismatched")
        if args.candidate_threshold_source == "default":
            val_cand_thresholds = np.full_like(val_cand_thresholds, 0.5)
        alphas = find_per_class_alphas(
            val_targets,
            val_cand_probs,
            val_prec_probs,
            val_cand_thresholds,
            alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        final_probs = (
            alphas.reshape(1, -1) * cand_probs + (1.0 - alphas.reshape(1, -1)) * prec_probs
        ) * np.maximum(cand_mask, prec_mask)
        final_thresholds = np.full_like(prec_thresholds, 0.5)

    metrics = evaluate_from_probs(targets, final_probs, final_thresholds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    meta = {
        "candidate_run": str(args.candidate_run),
        "precision_run": str(args.precision_run),
        "mode": args.mode,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "candidate_threshold_source": args.candidate_threshold_source,
    }
    if args.mode == "perclass_soft":
        meta["alpha_grid"] = [0.0, 0.25, 0.5, 0.75, 1.0]
        meta["per_class_alphas"] = alphas.tolist()
    (args.out_dir / "fusion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "test_metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
