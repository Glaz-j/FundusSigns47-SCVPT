#!/usr/bin/env python3
"""Reviewer-facing fusion controls for SCVPT.

This script answers a narrow question: is SCVPT simply a late ensemble of a
prompt-tuned VLM and an independent CNN? It uses the saved SCVPT decomposition
and the trained InceptionV3 baseline checkpoint to evaluate simple fusion
controls without retraining SCVPT.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path("/path/to/project")
RB_ROOT = ROOT / "retinal_benchmark"
RB_SRC = RB_ROOT / "src"
if str(RB_SRC) not in sys.path:
    sys.path.insert(0, str(RB_SRC))

from retinal_benchmark.config import load_yaml  # noqa: E402
from retinal_benchmark.models import create_model  # noqa: E402
from retinal_benchmark.training import build_dataloaders  # noqa: E402


SCVPT_NPZ = ROOT / "reports/mechanism/scvpt/scvpt_decomposition_test.npz"
INCEPTION_RUN = (
    ROOT
    / "retinal_benchmark/experiments/logs/vision3__inception_v3__42__20260412_055020"
)
OUT_DIR = ROOT / "reports/mechanism/scvpt/reviewer_fusion_controls"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_from_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def compute_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    targets = targets.astype(np.int32)
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(np.int32)
    aucs = []
    for class_idx in range(targets.shape[1]):
        if len(np.unique(targets[:, class_idx])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(targets[:, class_idx], probs[:, class_idx])))
        except ValueError:
            pass
    return {
        "mAP": float(average_precision_score(targets, probs, average="macro")) * 100.0,
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)) * 100.0,
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "macro_auc": float(np.mean(aucs)) * 100.0 if aucs else 0.0,
        "micro_precision": float(precision_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "micro_recall": float(recall_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)) * 100.0,
    }


def collect_inception_logits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = OUT_DIR / "inception_v3_test_logits.npz"
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return payload["sample_ids"], payload["logits"], payload["targets"]

    run_meta = json.loads((INCEPTION_RUN / "run_meta.json").read_text(encoding="utf-8"))
    data_cfg = load_yaml(run_meta["data_config"])
    model_cfg = load_yaml(run_meta["model_config"])
    train_cfg = load_yaml(run_meta["train_config"])
    train_cfg = dict(train_cfg)
    train_cfg["num_workers"] = min(int(train_cfg.get("num_workers", 8)), 4)
    _, loaders = build_dataloaders(data_cfg, train_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the same torchvision Inception construction path as the original run:
    # pretrained=True sets transform_input=True before the trained weights are loaded.
    model = create_model(model_cfg, pretrained=bool(model_cfg.get("pretrained", True)))
    state = torch.load(INCEPTION_RUN / "checkpoints/best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logits_list = []
    targets_list = []
    sample_ids = []
    with torch.no_grad():
        for images, targets, meta in loaders["test"]:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())
            ids = meta["sample_id"] if isinstance(meta, dict) else [m["sample_id"] for m in meta]
            sample_ids.extend([str(x) for x in ids])

    logits = torch.cat(logits_list).numpy()
    targets = torch.cat(targets_list).numpy()
    sample_ids_arr = np.asarray(sample_ids)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, sample_ids=sample_ids_arr, logits=logits, targets=targets)
    return sample_ids_arr, logits, targets


def align_by_sample_id(
    reference_ids: np.ndarray,
    other_ids: np.ndarray,
    other_logits: np.ndarray,
    other_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    index = {str(sample_id): i for i, sample_id in enumerate(other_ids)}
    order = [index[str(sample_id)] for sample_id in reference_ids]
    return other_logits[order], other_targets[order]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scvpt = np.load(SCVPT_NPZ, allow_pickle=True)
    sample_ids = scvpt["sample_ids"]
    targets = scvpt["targets"]
    base_logits = scvpt["base_logits"]
    final_logits = scvpt["final_logits"]
    all_weight_logits = scvpt["all_weight_logits"]
    hard_topk_logits = scvpt["hard_topk_logits"]
    soft_only_logits = scvpt["soft_only_logits"]

    inc_ids, inc_logits, inc_targets = collect_inception_logits()
    inc_logits, inc_targets = align_by_sample_id(sample_ids, inc_ids, inc_logits, inc_targets)
    if not np.array_equal(targets.astype(np.int32), inc_targets.astype(np.int32)):
        raise RuntimeError("Aligned Inception targets do not match SCVPT targets")

    base_prob = sigmoid(base_logits)
    inc_prob = sigmoid(inc_logits)
    simple_prob_ensemble_logits = logit_from_prob(0.5 * base_prob + 0.5 * inc_prob)
    simple_logit_ensemble_logits = 0.5 * base_logits + 0.5 * inc_logits

    variants = {
        "Sig2-g-dVPT-Tuned": base_logits,
        "InceptionV3": inc_logits,
        "Independent prob ensemble": simple_prob_ensemble_logits,
        "Independent logit ensemble": simple_logit_ensemble_logits,
        "All-label residual verifier": all_weight_logits,
        "Top-k residual verifier": hard_topk_logits,
        "Soft candidate residual verifier": soft_only_logits,
        "SCVPT": final_logits,
    }
    rows = []
    for name, logits in variants.items():
        metrics = compute_metrics(logits, targets)
        row = {"variant": name}
        row.update(metrics)
        rows.append(row)

    csv_path = OUT_DIR / "fusion_control_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "fusion_control_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path = OUT_DIR / "fusion_control_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# SCVPT Reviewer Fusion Controls\n\n")
        f.write("| Variant | mAP | Macro-F1 | Micro-F1 | Macro-AUC | Precision | Recall | Sample-F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['variant']} | {row['mAP']:.2f} | {row['macro_f1']:.2f} | "
                f"{row['micro_f1']:.2f} | {row['macro_auc']:.2f} | "
                f"{row['micro_precision']:.2f} | {row['micro_recall']:.2f} | "
                f"{row['sample_f1']:.2f} |\n"
            )
        f.write(
            "\nThe independent ensembles use the separately trained InceptionV3 baseline. "
            "The residual-verifier controls use the SCVPT-trained verifier branch and are "
            "therefore diagnostic controls rather than independent submitted models.\n"
        )
    print(md_path)
    for row in rows:
        print(
            row["variant"],
            f"mAP={row['mAP']:.2f}",
            f"MaF1={row['macro_f1']:.2f}",
            f"MiF1={row['micro_f1']:.2f}",
            f"AUC={row['macro_auc']:.2f}",
            f"P={row['micro_precision']:.2f}",
            f"R={row['micro_recall']:.2f}",
            f"SF1={row['sample_f1']:.2f}",
        )


if __name__ == "__main__":
    main()
