#!/usr/bin/env python
"""Evaluate SigLIP2 ViT-g zero-shot on the vision3 multi-label test set."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/path/to/project")
COOP = ROOT / "external/CoOp"
MINI_DASSL = ROOT / "external/Dassl.pytorch"
sys.path.insert(0, str(COOP))
sys.path.insert(0, str(MINI_DASSL))

import open_clip  # noqa: E402
from datasets.retinal_vision2 import _parse_multilabel, _resolve_impath  # noqa: E402
from trainers.openclip_coop_multilabel_cached import parse_openclip_backbone  # noqa: E402


BACKBONE = "openclip:ViT-gopt-16-SigLIP2-256:webli"
MANIFEST_PATH = ROOT / "retinal_benchmark/data_versions/vision3/manifest.csv"
LABEL_MAP_PATH = ROOT / "retinal_benchmark/data_versions/vision3/lesion_label_map.json"
IMAGE_CACHE_DIR = COOP / "cache/retinal_image_cache_288_v3"
OUTPUT_DIR = COOP / "output/retinal_vision3/ZeroShot/sig2g_zeroshot_vision3"


class RetinalZeroShotDataset(Dataset):
    def __init__(self, rows, transform, num_classes: int):
        self.rows = rows
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample_id = str(row.get("sample_id", "")).strip()
        split = str(row.get("split", "")).strip()
        cached = IMAGE_CACHE_DIR / split / f"{sample_id}.jpg"
        impath = str(cached) if cached.exists() else _resolve_impath(row.get("linked_image_path"), row.get("source_image_path"))
        with Image.open(impath) as img:
            image = self.transform(img.convert("RGB"))
        label = _parse_multilabel(row.get("lesion_ids"), self.num_classes)
        return image, label, sample_id


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tune_thresholds(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    probs = sigmoid(logits)
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for idx in range(targets.shape[1]):
        y_true = targets[:, idx].astype(np.int32)
        if y_true.sum() == 0:
            continue
        # Zero-shot SigLIP similarities are often badly calibrated after sigmoid,
        # so use data-adaptive probability cutoffs instead of a fixed 0.05-0.95 grid.
        grid = np.unique(np.quantile(probs[:, idx], np.linspace(0.0, 1.0, 201)))
        grid = np.unique(np.concatenate([grid, np.array([0.5], dtype=np.float32)]))
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            y_pred = (probs[:, idx] >= threshold).astype(np.int32)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[idx] = best_threshold
    return thresholds


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    classnames: list[str],
    thresholds: np.ndarray | None = None,
) -> tuple[dict, list[dict]]:
    probs = sigmoid(logits)
    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    targets = targets.astype(np.int32)

    aucs = []
    per_class = []
    for idx, class_name in enumerate(classnames):
        y_true = targets[:, idx]
        y_score = probs[:, idx]
        ap_val = None
        auc_val = None
        try:
            ap_val = float(average_precision_score(y_true, y_score)) * 100.0
        except Exception:
            pass
        if len(np.unique(y_true)) >= 2:
            try:
                auc_metric = float(roc_auc_score(y_true, y_score))
                aucs.append(auc_metric)
                auc_val = auc_metric * 100.0
            except Exception:
                pass
        per_class.append(
            {
                "class_index": idx,
                "class_name": class_name,
                "positive_count": int(y_true.sum()),
                "ap": None if ap_val is None else round(ap_val, 6),
                "auc": None if auc_val is None else round(auc_val, 6),
            }
        )

    summary = {
        "mAP": float(average_precision_score(targets, probs, average="macro")) * 100.0,
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)) * 100.0,
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "macro_auc": (float(np.mean(aucs)) * 100.0) if aucs else 0.0,
        "micro_precision": float(precision_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "micro_recall": float(recall_score(targets, preds, average="micro", zero_division=0)) * 100.0,
        "sample_f1": float(f1_score(targets, preds, average="samples", zero_division=0)) * 100.0,
    }
    return {k: round(v, 6) for k, v in summary.items()}, per_class


@torch.no_grad()
def collect_split(
    split: str,
    df: pd.DataFrame,
    preprocess_val,
    model,
    text_features: torch.Tensor,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = df[df["split"] == split].reset_index(drop=True).to_dict("records")
    dataset = RetinalZeroShotDataset(rows, preprocess_val, num_classes)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    logits_list = []
    targets_list = []
    for batch_idx, (images, targets, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * image_features.float() @ text_features.float().t()
        logits_list.append(logits.cpu())
        targets_list.append(targets.cpu())
        if batch_idx % 10 == 0:
            print(f"[{split}] Processed {batch_idx * loader.batch_size} samples", flush=True)

    logits = torch.cat(logits_list).numpy()
    targets = torch.cat(targets_list).numpy()
    return logits, targets


@torch.no_grad()
def main() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / ".model_cache" / "hf_home"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".model_cache" / "torch_home"))

    with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    classnames = [label_map[str(i)] for i in range(1, len(label_map) + 1)]
    prompts = [f"a fundus photograph of {name.replace('_', ' ')}." for name in classnames]

    model_name, pretrained_name = parse_openclip_backbone(BACKBONE)
    print(f"Loading OpenCLIP zero-shot backbone: {model_name} / {pretrained_name}", flush=True)
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_name,
        device="cpu",
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenized = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    df = pd.read_csv(MANIFEST_PATH)
    val_logits, val_targets = collect_split("val", df, preprocess_val, model, text_features, device, len(classnames))
    test_logits, test_targets = collect_split("test", df, preprocess_val, model, text_features, device, len(classnames))
    thresholds = tune_thresholds(val_logits, val_targets)

    fixed_summary, fixed_per_class = compute_metrics(test_logits, test_targets, classnames)
    val_summary, _ = compute_metrics(val_logits, val_targets, classnames, thresholds)
    calibrated_summary, calibrated_per_class = compute_metrics(test_logits, test_targets, classnames, thresholds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details = {
        "split": "test",
        "num_samples": int(test_targets.shape[0]),
        "num_classes": int(test_targets.shape[1]),
        "backbone": BACKBONE,
        "prompt_template": "a fundus photograph of {}.",
        "fixed_threshold": 0.5,
        "fixed_summary": fixed_summary,
        "val_calibrated_thresholds": thresholds.tolist(),
        "val_calibrated_val_summary": val_summary,
        "val_calibrated_test_summary": calibrated_summary,
        "per_class": calibrated_per_class,
    }
    (OUTPUT_DIR / "test_metrics_detailed.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "test_per_class_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_index", "class_name", "positive_count", "ap", "auc"])
        writer.writeheader()
        writer.writerows(calibrated_per_class)
    print(json.dumps({"fixed@0.5": fixed_summary, "val_calibrated": calibrated_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
