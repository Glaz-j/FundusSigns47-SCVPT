#!/usr/bin/env python3
"""Bootstrap confidence intervals for SCVPT reviewer-facing controls."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DECOMP_PATH = ROOT / "reports/mechanism/scvpt/scvpt_decomposition_test.npz"
INCEPTION_PATH = ROOT / "reports/mechanism/scvpt/reviewer_fusion_controls/inception_v3_test_logits.npz"
OUT_DIR = ROOT / "reports/statistics/scvpt_bootstrap_ci"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(logits: np.ndarray, targets: np.ndarray, include_auc: bool = True) -> dict[str, float]:
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_ap = float(average_precision_score(targets, probs, average="macro"))

    metrics = {
        "mAP": m_ap * 100.0,
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)) * 100.0,
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "micro_precision": float(precision_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "micro_recall": float(recall_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)) * 100.0,
    }
    if include_auc:
        aucs: list[float] = []
        for class_idx in range(targets.shape[1]):
            y_true = targets[:, class_idx]
            if np.unique(y_true).size < 2:
                continue
            aucs.append(float(roc_auc_score(y_true, probs[:, class_idx])))
        metrics["macro_auc"] = float(np.mean(aucs)) * 100.0 if aucs else 0.0
    return metrics


def summarize_bootstrap(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(values, [2.5, 97.5]).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    decomp = np.load(DECOMP_PATH, allow_pickle=True)
    inception = np.load(INCEPTION_PATH, allow_pickle=True)

    targets = decomp["targets"].astype(np.int32)
    if not np.array_equal(targets, inception["targets"].astype(np.int32)):
        raise RuntimeError("SCVPT and Inception target arrays do not match.")
    if not np.array_equal(decomp["sample_ids"], inception["sample_ids"]):
        raise RuntimeError("SCVPT and Inception sample order does not match.")

    base_logits = decomp["base_logits"].astype(np.float32)
    inception_logits = inception["logits"].astype(np.float32)
    variants = {
        "Base dVPT-Tuned": base_logits,
        "InceptionV3": inception_logits,
        "Independent logit ensemble": (base_logits + inception_logits) / 2.0,
        "All-label residual verifier": decomp["all_weight_logits"].astype(np.float32),
        "SCVPT": decomp["final_logits"].astype(np.float32),
    }

    rng = np.random.default_rng(args.seed)
    n = targets.shape[0]
    metric_names = ["mAP", "macro_f1", "micro_f1", "micro_recall", "sample_f1"]
    full_metrics = {name: compute_metrics(logits, targets, include_auc=True) for name, logits in variants.items()}
    boot_values = {
        name: {metric: np.empty(args.n_bootstrap, dtype=np.float32) for metric in metric_names}
        for name in variants
    }

    for boot_idx in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_targets = targets[idx]
        for name, logits in variants.items():
            metrics = compute_metrics(logits[idx], boot_targets, include_auc=False)
            for metric in metric_names:
                boot_values[name][metric][boot_idx] = metrics[metric]

    rows = []
    for name in variants:
        row = {"variant": name}
        for metric in metric_names:
            low, high = summarize_bootstrap(boot_values[name][metric])
            row[metric] = full_metrics[name][metric]
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "bootstrap_ci_summary.csv", index=False)
    (args.out_dir / "bootstrap_ci_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected = metric_names
    lines = [
        "# SCVPT bootstrap confidence intervals",
        "",
        f"- Bootstrap samples: {args.n_bootstrap}",
        f"- Seed: {args.seed}",
        "- Unit: test image, sampled with replacement.",
        "- Values are percentages; intervals are percentile 95% CIs.",
        "",
        "| Variant | " + " | ".join(selected) + " |",
        "|---" + "|---:" * len(selected) + "|",
    ]
    for row in rows:
        cells = [row["variant"]]
        for metric in selected:
            cells.append(f"{row[metric]:.2f} [{row[f'{metric}_ci_low']:.2f}, {row[f'{metric}_ci_high']:.2f}]")
        lines.append("| " + " | ".join(cells) + " |")
    (args.out_dir / "bootstrap_ci_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
