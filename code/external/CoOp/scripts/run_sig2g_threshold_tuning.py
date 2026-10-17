#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from train import setup_cfg
from dassl.engine import build_trainer


def compute_metrics(logits, targets, thresholds):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    aucs = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(targets[:, i], probs[:, i])))
        except Exception:
            pass
    metrics = {
        "mAP": float(average_precision_score(targets, probs, average="macro")) * 100.0,
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)) * 100.0,
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "macro_auc": (float(np.mean(aucs)) * 100.0) if aucs else 0.0,
        "micro_precision": float(precision_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "micro_recall": float(recall_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)) * 100.0,
    }
    return metrics, probs, preds


def collect_logits(trainer, loader):
    outputs, targets = [], []
    trainer.set_model_mode("eval")
    with torch.no_grad():
        for batch in loader:
            inp, lab = trainer.parse_batch_test(batch)
            out = trainer.model_inference(inp)
            outputs.append(out.detach().cpu())
            targets.append(lab.detach().cpu())
    logits = torch.cat(outputs).numpy()
    labels = torch.cat(targets).numpy()
    return logits, labels


def tune_thresholds(val_logits, val_targets):
    probs = 1.0 / (1.0 + np.exp(-val_logits))
    thresholds = np.full(val_targets.shape[1], 0.5, dtype=np.float32)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    for i in range(val_targets.shape[1]):
        y_true = val_targets[:, i]
        if y_true.sum() == 0:
            thresholds[i] = 0.5
            continue
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            y_pred = (probs[:, i] >= t).astype(np.int32)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)
        thresholds[i] = best_t
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset-config-file", required=True)
    parser.add_argument("--trainer", default="OpenCLIPVPTDeepOnlyMultiLabel")
    parser.add_argument("--root", default="/path/to/project")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg_args = SimpleNamespace(
        root=args.root,
        output_dir=args.output_dir,
        resume="",
        seed=-1,
        source_domains=None,
        target_domains=None,
        transforms=None,
        config_file=args.config_file,
        dataset_config_file=args.dataset_config_file,
        trainer=args.trainer,
        backbone="",
        head="",
        eval_only=False,
        model_dir="",
        load_epoch=None,
        no_train=False,
        opts=[],
    )
    cfg = setup_cfg(cfg_args)
    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)

    val_logits, val_targets = collect_logits(trainer, trainer.val_loader)
    thresholds = tune_thresholds(val_logits, val_targets)
    val_metrics, _, _ = compute_metrics(val_logits, val_targets, thresholds)

    test_logits, test_targets = collect_logits(trainer, trainer.test_loader)
    test_metrics, _, _ = compute_metrics(test_logits, test_targets, thresholds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "thresholds": thresholds.tolist(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (out_dir / "threshold_tuning_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=> threshold tuning result")
    print(json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
