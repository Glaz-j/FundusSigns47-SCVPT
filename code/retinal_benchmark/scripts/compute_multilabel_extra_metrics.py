#!/usr/bin/env python3
"""Compute threshold-dependent multi-label metrics for saved runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RB_ROOT = ROOT / "retinal_benchmark"
RB_SRC = RB_ROOT / "src"
if str(RB_SRC) not in sys.path:
    sys.path.insert(0, str(RB_SRC))

from retinal_benchmark.config import load_yaml  # noqa: E402
from retinal_benchmark.models import create_model  # noqa: E402
from retinal_benchmark.training import build_dataloaders, collect_logits_targets  # noqa: E402


BASELINE_RUNS = [
    ("ResNet50", "vision3__resnet50__42__20260412_081237"),
    ("DenseNet121", "vision3__densenet121__42__20260412_081236"),
    ("EfficientNet-B4", "vision3__efficientnet_b4__42__20260412_090351"),
    ("ConvNeXt-Tiny", "vision3__convnext_tiny__42__20260412_055036"),
    ("InceptionV3", "vision3__inception_v3__42__20260412_055020"),
    ("DeiT-Small", "vision3__deit_small__42__20260412_090351"),
    ("ViT-B/16", "vision3__vit_b16__42__20260412_090351"),
    ("Swin-Tiny", "vision3__swin_tiny__42__20260412_055036"),
]

SUMMARY_COLUMNS = [
    "method",
    "mAP",
    "macro_f1",
    "micro_f1",
    "macro_auc",
    "micro_precision",
    "micro_recall",
    "sample_f1",
    "hamming_loss",
    "exact_match",
    "macro_sensitivity",
    "macro_specificity",
]

COOP_BEST = {
    "name": "Sig2-g-dVPT-Tuned+CV+SPCTR",
    "config": ROOT
    / "external/CoOp/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/"
    / "vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidate_selectiveperclassthr_20ep_vision3.yaml",
    "dataset_config": ROOT / "external/CoOp/configs/datasets/retinal_vision3.yaml",
    "model_dir": ROOT
    / "external/CoOp/output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel/"
    / "sig2g_fullinception_candidate_selectiveperclassthr_20ep_cuda3",
    "epoch": 20,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_metrics_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    targets = targets.astype(np.int32)
    probs = sigmoid(logits)
    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)

    per_class = []
    aucs = []
    sensitivities = []
    specificities = []
    tp_total = tn_total = fp_total = fn_total = 0
    for class_idx in range(targets.shape[1]):
        y_true = targets[:, class_idx]
        y_pred = preds[:, class_idx]
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp_total += tp
        tn_total += tn
        fp_total += fp
        fn_total += fn

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        if not np.isnan(sensitivity):
            sensitivities.append(float(sensitivity))
        if not np.isnan(specificity):
            specificities.append(float(specificity))
        if len(np.unique(y_true)) >= 2:
            try:
                aucs.append(float(roc_auc_score(y_true, probs[:, class_idx])))
            except Exception:
                pass

        per_class.append(
            {
                "class_index": class_idx,
                "positive_count": int(y_true.sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "sensitivity": None if np.isnan(sensitivity) else float(sensitivity) * 100.0,
                "specificity": None if np.isnan(specificity) else float(specificity) * 100.0,
            }
        )

    summary = {
        "mAP": float(average_precision_score(targets, probs, average="macro")) * 100.0,
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)) * 100.0,
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "macro_auc": _safe_mean(aucs) * 100.0,
        "micro_precision": float(precision_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "micro_recall": float(recall_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)) * 100.0,
        "hamming_loss": float(np.mean(preds != targets)) * 100.0,
        "exact_match": float(np.mean(np.all(preds == targets, axis=1))) * 100.0,
        "macro_sensitivity": _safe_mean(sensitivities) * 100.0,
        "macro_specificity": _safe_mean(specificities) * 100.0,
        "micro_specificity": (tn_total / (tn_total + fp_total) * 100.0) if (tn_total + fp_total) > 0 else 0.0,
    }
    return summary, per_class


def load_benchmark_logits(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    data_cfg = load_yaml(run_meta["data_config"])
    model_cfg = load_yaml(run_meta["model_config"])
    train_cfg = load_yaml(run_meta["train_config"])
    _, loaders = build_dataloaders(data_cfg, train_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The full trained weights are loaded immediately below, so local architecture
    # initialization is enough and avoids accidental downloads from model hubs.
    model = create_model(model_cfg, pretrained=False)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    logits, targets = collect_logits_targets(model, loaders[split], device)
    thresholds_path = run_dir / "thresholds.json"
    thresholds = None
    if thresholds_path.exists():
        thresholds = np.array(json.loads(thresholds_path.read_text(encoding="utf-8")), dtype=np.float32)
    return logits, targets, thresholds


def load_coop_visual_logits(split: str) -> tuple[np.ndarray, np.ndarray, None]:
    coop_root = ROOT / "external/CoOp"
    dassl_root = ROOT / "external/Dassl.pytorch"
    sys.path.insert(0, str(dassl_root))
    sys.path.insert(0, str(coop_root))

    from dassl.engine import build_trainer  # noqa: WPS433
    from train import setup_cfg  # noqa: WPS433

    output_dir = ROOT / "tmp_extra_metrics_coop_eval"
    args = SimpleNamespace(
        root=str(ROOT),
        output_dir=str(output_dir),
        resume="",
        seed=-1,
        source_domains=None,
        target_domains=None,
        transforms=None,
        config_file=str(COOP_BEST["config"]),
        dataset_config_file=str(COOP_BEST["dataset_config"]),
        trainer="OpenCLIPVPTDeepVisualResidualMultiLabel",
        backbone="",
        head="",
        eval_only=False,
        model_dir="",
        load_epoch=None,
        no_train=False,
        opts=[],
    )
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.load_model(str(COOP_BEST["model_dir"]), epoch=int(COOP_BEST["epoch"]))
    loader = trainer.val_loader if split == "val" else trainer.test_loader
    trainer.set_model_mode("eval")
    outputs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            inp, lab = trainer.parse_batch_test(batch)
            out = trainer.model_inference(inp)
            outputs.append(out.detach().cpu())
            labels.append(lab.detach().cpu())
    return torch.cat(outputs).numpy(), torch.cat(labels).numpy(), None


def write_outputs(out_dir: Path, rows: list[dict], per_class_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vision3_extra_metrics_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        with (out_dir / "vision3_extra_metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    if per_class_rows:
        with (out_dir / "vision3_extra_metrics_per_class.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_class_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", default=str(RB_ROOT / "reports"))
    parser.add_argument("--skip-coop-best", action="store_true")
    args = parser.parse_args()

    rows = []
    per_class_rows = []
    log_root = RB_ROOT / "experiments" / "logs"

    jobs: list[tuple[str, str | Path]] = [(name, log_root / run_name) for name, run_name in BASELINE_RUNS]
    if not args.skip_coop_best:
        jobs.append((str(COOP_BEST["name"]), "coop_best"))

    for name, source in jobs:
        print(f"[extra-metrics] {name}", flush=True)
        if source == "coop_best":
            logits, targets, thresholds = load_coop_visual_logits(args.split)
        else:
            logits, targets, thresholds = load_benchmark_logits(Path(source), args.split)
        summary, per_class = compute_metrics_from_logits(logits, targets, thresholds=thresholds)
        rounded_summary = {k: round(v, 4) for k, v in summary.items()}
        row = {"method": name, **{k: rounded_summary[k] for k in SUMMARY_COLUMNS if k != "method"}}
        rows.append(row)
        for item in per_class:
            per_class_rows.append({"method": name, **item})

    write_outputs(Path(args.output_dir), rows, per_class_rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
