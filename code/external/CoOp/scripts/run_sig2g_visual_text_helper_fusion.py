#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path("/path/to/project")
COOP = ROOT / "external/CoOp"
sys.path.insert(0, str(COOP))
sys.path.insert(0, str(ROOT / "external/Dassl.pytorch"))

from dassl.engine import build_trainer  # noqa: E402
from train import setup_cfg  # noqa: E402


VISUAL_CONFIG = COOP / "configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidate_selectiveperclassthr_20ep_vision3.yaml"
DATASET_CONFIG = COOP / "configs/datasets/retinal_vision3.yaml"
VISUAL_MODEL_DIR = COOP / "output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel/sig2g_fullinception_candidate_selectiveperclassthr_20ep_cuda3"

HELPER_SPECS = {
    "candidate": {
        "config": COOP / "configs/trainers/OpenCLIPVPTDeepResidualMultiLabelCached/vit_gopt16_siglip2_256_vptdeep_stage2_coop_taskres_candidate_100ep_vision3.yaml",
        "model_dir": COOP / "output/retinal_vision3/OpenCLIPVPTDeepResidualMultiLabelCached/sig2g_cooptaskres_candidate_cuda3",
        "epoch": 100,
    },
    "hardclass": {
        "config": COOP / "configs/trainers/OpenCLIPVPTDeepResidualMultiLabelCached/vit_gopt16_siglip2_256_vptdeep_stage2_coop_taskres_hardclass_100ep_vision3.yaml",
        "model_dir": COOP / "output/retinal_vision3/OpenCLIPVPTDeepResidualMultiLabelCached/sig2g_cooptaskres_hardclass_cuda1",
        "epoch": 100,
    },
    "recall": {
        "config": COOP / "configs/trainers/OpenCLIPVPTDeepResidualMultiLabelCached/vit_gopt16_siglip2_256_vptdeep_stage2_coop_taskres_recall_100ep_vision3.yaml",
        "model_dir": COOP / "output/retinal_vision3/OpenCLIPVPTDeepResidualMultiLabelCached/sig2g_cooptaskres_recall_cuda2",
        "epoch": 100,
    },
}


def _args(config_file, output_dir, trainer, model_dir="", load_epoch=None, eval_only=False):
    return SimpleNamespace(
        root=str(ROOT),
        output_dir=str(output_dir),
        resume="",
        seed=-1,
        source_domains=None,
        target_domains=None,
        transforms=None,
        config_file=str(config_file),
        dataset_config_file=str(DATASET_CONFIG),
        trainer=trainer,
        backbone="",
        head="",
        eval_only=eval_only,
        model_dir=str(model_dir) if model_dir else "",
        load_epoch=load_epoch,
        no_train=True,
        opts=[],
    )


def build_loaded_trainer(config_file, trainer_name, model_dir, epoch):
    args = _args(config_file, output_dir="/tmp/sig2g_fusion_scratch", trainer=trainer_name, model_dir=model_dir, load_epoch=epoch)
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.load_model(str(model_dir), epoch=epoch)
    return trainer


@torch.no_grad()
def collect_visual_logits(trainer, split):
    loader = trainer.val_loader if split == "val" else trainer.test_loader
    outputs = []
    labels = []
    masks = []
    for batch in loader:
        parsed_input, label = trainer.parse_batch_test(batch)
        if isinstance(parsed_input, (tuple, list)):
            image, inception_input = parsed_input
        else:
            image, inception_input = parsed_input, None
        logits, aux = trainer.model.forward_with_aux(image, inception_input=inception_input)
        if "candidate_weight" in aux:
            mask = aux["candidate_weight"]
        elif "candidate_mask" in aux:
            mask = aux["candidate_mask"]
        elif "uncertain_mask" in aux:
            mask = aux["uncertain_mask"]
        else:
            probs = torch.sigmoid(logits)
            mask = (probs >= trainer.model.candidate_threshold).float()
            mask = torch.maximum(mask, trainer.model._topk_mask(probs))
        outputs.append(logits.detach().cpu())
        labels.append(label.detach().cpu())
        masks.append(mask.detach().cpu())
    return torch.cat(outputs), torch.cat(labels), torch.cat(masks)


@torch.no_grad()
def collect_helper_logits(trainer, split):
    loader = trainer.val_loader if split == "val" else trainer.test_loader
    outputs = []
    base_outputs = []
    labels = []
    for batch in loader:
        features, label = trainer.parse_batch_test(batch)
        logits = trainer.model(features)
        base_logits = trainer.model._base_logits(features)
        outputs.append(logits.detach().cpu())
        base_outputs.append(base_logits.detach().cpu())
        labels.append(label.detach().cpu())
    return torch.cat(outputs), torch.cat(base_outputs), torch.cat(labels)


def tune_per_class_thresholds(logits, labels, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    probs = torch.sigmoid(logits).numpy()
    y = labels.numpy().astype(np.int32)
    thresholds = np.full((y.shape[1],), 0.5, dtype=np.float32)
    for c in range(y.shape[1]):
        yc = y[:, c]
        if yc.sum() == 0:
            thresholds[c] = 0.95
            continue
        best_thr = 0.5
        best_f1 = -1.0
        for thr in grid:
            pred = (probs[:, c] >= thr).astype(np.int32)
            f1 = f1_score(yc, pred, zero_division=0)
            if f1 > best_f1 + 1e-12:
                best_f1 = f1
                best_thr = float(thr)
        thresholds[c] = best_thr
    return thresholds


def evaluate_with_thresholds(logits, labels, thresholds):
    probs = torch.sigmoid(logits).numpy()
    y = labels.numpy().astype(np.int32)
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)

    aucs = []
    for i in range(y.shape[1]):
        if len(np.unique(y[:, i])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[:, i], probs[:, i]))
        except Exception:
            pass

    results = {
        "mAP": float(average_precision_score(y, probs, average="macro")) * 100.0,
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)) * 100.0,
        "micro_f1": float(f1_score(y, preds, average="micro", zero_division=0)) * 100.0,
        "macro_auc": (float(np.mean(aucs)) * 100.0) if aucs else 0.0,
        "micro_precision": float(precision_score(y, preds, average="micro", zero_division=0)) * 100.0,
        "micro_recall": float(recall_score(y, preds, average="micro", zero_division=0)) * 100.0,
        "sample_f1": float(f1_score(y, preds, average="samples", zero_division=0)) * 100.0,
    }
    return {k: round(v, 6) for k, v in results.items()}


def rank_score(metrics):
    return (metrics["micro_f1"], metrics["sample_f1"], metrics["micro_precision"], metrics["micro_recall"])


def run_one(helper_name, out_root, betas):
    helper = HELPER_SPECS[helper_name]

    visual_trainer = build_loaded_trainer(VISUAL_CONFIG, "OpenCLIPVPTDeepVisualResidualMultiLabel", VISUAL_MODEL_DIR, epoch=20)
    val_visual, val_labels, val_mask = collect_visual_logits(visual_trainer, "val")
    test_visual, test_labels, test_mask = collect_visual_logits(visual_trainer, "test")
    del visual_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    helper_trainer = build_loaded_trainer(helper["config"], "OpenCLIPVPTDeepResidualMultiLabelCached", helper["model_dir"], epoch=helper["epoch"])
    val_helper, val_helper_base, val_labels_2 = collect_helper_logits(helper_trainer, "val")
    test_helper, test_helper_base, test_labels_2 = collect_helper_logits(helper_trainer, "test")
    del helper_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not torch.equal(val_labels, val_labels_2):
        raise RuntimeError(f"{helper_name}: val labels are misaligned between visual and helper branches")
    if not torch.equal(test_labels, test_labels_2):
        raise RuntimeError(f"{helper_name}: test labels are misaligned between visual and helper branches")

    val_delta = torch.relu(val_helper - val_helper_base)
    test_delta = torch.relu(test_helper - test_helper_base)

    best = None
    best_thresholds = None
    best_val = None

    for beta in betas:
        fused_val = val_visual + beta * val_mask * val_delta
        thresholds = tune_per_class_thresholds(fused_val, val_labels)
        val_metrics = evaluate_with_thresholds(fused_val, val_labels, thresholds)
        candidate = {"beta": beta, "val_metrics": val_metrics}
        if best is None or rank_score(val_metrics) > rank_score(best_val):
            best = candidate
            best_thresholds = thresholds
            best_val = val_metrics

    fused_test = test_visual + best["beta"] * test_mask * test_delta
    test_metrics = evaluate_with_thresholds(fused_test, test_labels, best_thresholds)

    out_dir = Path(out_root) / f"sig2g_visual_texthelper_{helper_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "helper": helper_name,
        "beta_grid": list(betas),
        "selected_beta": best["beta"],
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "thresholds": [float(x) for x in best_thresholds.tolist()],
    }
    with (out_dir / "fusion_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"=== {helper_name} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpers", nargs="+", default=["candidate", "hardclass", "recall"])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument(
        "--output-root",
        default=str(COOP / "output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabelFusion"),
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    all_results = {}
    for helper_name in args.helpers:
        all_results[helper_name] = run_one(helper_name, args.output_root, args.betas)

    summary_path = Path(args.output_root) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
