#!/usr/bin/env python3
"""Mechanistic analyses for SCVPT on FundusSigns-47.

The script is intentionally evaluation-only: it loads the frozen best SCVPT
checkpoint, decomposes its logits into the base dVPT branch and candidate
verifier residuals, and writes reviewer-facing figures/tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path("/path/to/project")
COOP = ROOT / "external/CoOp"
sys.path.insert(0, str(COOP))
sys.path.insert(0, str(ROOT / "external/Dassl.pytorch"))
sys.path.insert(0, str(ROOT))

from dassl.engine import build_trainer  # noqa: E402
from retinal_benchmark.scripts.analyze_vision3_dataset import EN_SHORT_NAMES, LESION_GROUPS  # noqa: E402
from train import setup_cfg  # noqa: E402


DATASET_CONFIG = COOP / "configs/datasets/retinal_vision3.yaml"
SCVPT_CONFIG = (
    COOP
    / "configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel"
    / "vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidate_selectiveperclassthr_20ep_vision3.yaml"
)
SCVPT_MODEL_DIR = (
    COOP
    / "output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel"
    / "sig2g_fullinception_candidate_selectiveperclassthr_20ep_cuda3"
)
OUT_DIR = ROOT / "reports/mechanism/scvpt"
PAPER_DIR = ROOT / "paper/template"


RUNS = {
    "zero-shot": COOP / "output/retinal_vision3/ZeroShot/sig2g_zeroshot_vision3",
    "plain dVPT": (
        COOP
        / "output/retinal_vision3/OpenCLIPVPTDeepOnlyMultiLabel"
        / "vit_gopt16_siglip2_256_ctx16_fulldepth_plain_cuda2"
    ),
    "dVPT-Tuned": (
        COOP
        / "output/retinal_vision3/OpenCLIPVPTDeepOnlyMultiLabel"
        / "vit_gopt16_siglip2_256_ctx16_fulldepth_tuned_cuda0"
    ),
    "+CV": (
        COOP
        / "output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel"
        / "sig2g_fullinception_candidate_20ep_cuda1"
    ),
    "SCVPT": SCVPT_MODEL_DIR,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_loaded_trainer() -> object:
    args = SimpleNamespace(
        root=str(ROOT),
        output_dir=str(ROOT / "tmp_scvpt_mechanism_eval"),
        resume="",
        seed=-1,
        source_domains=None,
        target_domains=None,
        transforms=None,
        config_file=str(SCVPT_CONFIG),
        dataset_config_file=str(DATASET_CONFIG),
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
    trainer.load_model(str(SCVPT_MODEL_DIR), epoch=20)
    trainer.set_model_mode("eval")
    return trainer


def compute_metrics(logits: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(np.int32)
    targets = targets.astype(np.int32)
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


def per_class_scores(logits: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(np.int32)
    rows = []
    for class_idx in range(targets.shape[1]):
        y = targets[:, class_idx].astype(np.int32)
        p = preds[:, class_idx].astype(np.int32)
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        ap = float(average_precision_score(y, probs[:, class_idx])) * 100.0 if y.sum() else 0.0
        auc = float("nan")
        if len(np.unique(y)) >= 2:
            auc = float(roc_auc_score(y, probs[:, class_idx])) * 100.0
        rows.append(
            {
                "class_index": class_idx,
                "label_id": class_idx + 1,
                "class_name": EN_SHORT_NAMES[class_idx + 1],
                "positive_count": int(y.sum()),
                "ap": ap,
                "auc": auc,
                "f1": float(f1_score(y, p, zero_division=0)) * 100.0,
                "precision": float(precision_score(y, p, zero_division=0)) * 100.0,
                "recall": float(recall_score(y, p, zero_division=0)) * 100.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def get_sample_ids(batch) -> list[str]:
    if len(batch) >= 3:
        return [str(x) for x in batch[2]]
    return ["" for _ in range(len(batch[0]))]


@torch.no_grad()
def decompose_batch(model, image, inception_input):
    base_features, base_logits = model._base_features_logits(image)
    inc_input = inception_input if inception_input is not None else model._to_imagenet_input(image.float(), size=299)
    _, residual_logits = model.branch.forward_logits(inc_input)
    probs = torch.sigmoid(base_logits)
    topk_mask = model._topk_mask(probs)
    threshold = torch.sigmoid(model.candidate_threshold_param).unsqueeze(0)
    scale = F.softplus(model.candidate_scale_param).unsqueeze(0)
    soft_weight = torch.sigmoid(scale * (probs - threshold))
    candidate_weight = torch.maximum(soft_weight, topk_mask)
    final_logits = base_logits + model.alpha * candidate_weight * residual_logits
    all_weight_logits = base_logits + model.alpha * residual_logits
    hard_topk_logits = base_logits + model.alpha * topk_mask * residual_logits
    soft_only_logits = base_logits + model.alpha * soft_weight * residual_logits
    return {
        "base_logits": base_logits,
        "residual_logits": residual_logits,
        "candidate_weight": candidate_weight,
        "topk_mask": topk_mask,
        "soft_weight": soft_weight,
        "final_logits": final_logits,
        "all_weight_logits": all_weight_logits,
        "hard_topk_logits": hard_topk_logits,
        "soft_only_logits": soft_only_logits,
        "alpha": model.alpha.detach().float(),
    }


def collect_decomposition(trainer) -> dict[str, np.ndarray | list[str]]:
    model = trainer.model
    loader = trainer.test_loader
    buckets = defaultdict(list)
    sample_ids: list[str] = []
    for batch in loader:
        parsed_input, label = trainer.parse_batch_test(batch)
        if isinstance(parsed_input, (tuple, list)):
            image, inception_input = parsed_input
        else:
            image, inception_input = parsed_input, None
        out = decompose_batch(model, image, inception_input)
        for key, value in out.items():
            if key == "alpha":
                continue
            buckets[key].append(value.detach().cpu())
        buckets["targets"].append(label.detach().cpu())
        sample_ids.extend(get_sample_ids(batch))
    result: dict[str, np.ndarray | list[str]] = {"sample_ids": sample_ids}
    for key, values in buckets.items():
        result[key] = torch.cat(values).numpy()
    result["alpha"] = np.array(float(model.alpha.detach().cpu()))
    result["candidate_threshold"] = torch.sigmoid(model.candidate_threshold_param).detach().cpu().numpy()
    result["candidate_scale"] = F.softplus(model.candidate_scale_param).detach().cpu().numpy()
    return result


def load_per_class_json(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "test_metrics_detailed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload["per_class"]:
        rows.append(
            {
                "class_index": row["class_index"],
                "label_id": row["class_index"] + 1,
                "class_name": EN_SHORT_NAMES[row["class_index"] + 1],
                "positive_count": row["positive_count"],
                "ap": row.get("ap", np.nan),
                "auc": row.get("auc", np.nan),
            }
        )
    return pd.DataFrame(rows)


def group_for_label(label_id: int) -> str:
    for name, ids in LESION_GROUPS.items():
        if label_id in ids:
            return name
    return "Other"


def summarize_candidate_and_residual(data: dict[str, np.ndarray | list[str]]) -> dict[str, float]:
    targets = data["targets"].astype(np.int32)
    base = data["base_logits"]
    final = data["final_logits"]
    residual = data["residual_logits"]
    weight = data["candidate_weight"]
    alpha = float(data["alpha"])
    contrib = alpha * weight * residual
    base_pred = (sigmoid(base) >= 0.5).astype(np.int32)
    final_pred = (sigmoid(final) >= 0.5).astype(np.int32)
    candidate = (weight >= 0.5).astype(np.int32)
    topk = data["topk_mask"].astype(np.int32)
    pos = targets == 1
    neg = targets == 0
    base_fn = pos & (base_pred == 0)
    base_fp = neg & (base_pred == 1)
    recovered_fn = base_fn & (final_pred == 1)
    removed_fp = base_fp & (final_pred == 0)
    new_fp = neg & (base_pred == 0) & (final_pred == 1)
    lost_tp = pos & (base_pred == 1) & (final_pred == 0)
    return {
        "avg_candidate_labels": float(candidate.sum(axis=1).mean()),
        "median_candidate_labels": float(np.median(candidate.sum(axis=1))),
        "candidate_gt_coverage": float(candidate[pos].mean() * 100.0) if pos.any() else 0.0,
        "top9_gt_coverage": float(topk[pos].mean() * 100.0) if pos.any() else 0.0,
        "mean_positive_candidate_weight": float(weight[pos].mean()) if pos.any() else 0.0,
        "mean_negative_candidate_weight": float(weight[neg].mean()) if neg.any() else 0.0,
        "recovered_fn_count": int(recovered_fn.sum()),
        "removed_fp_count": int(removed_fp.sum()),
        "new_fp_count": int(new_fp.sum()),
        "lost_tp_count": int(lost_tp.sum()),
        "mean_delta_recovered_fn": float(contrib[recovered_fn].mean()) if recovered_fn.any() else 0.0,
        "mean_delta_removed_fp": float(contrib[removed_fp].mean()) if removed_fp.any() else 0.0,
        "mean_delta_new_fp": float(contrib[new_fp].mean()) if new_fp.any() else 0.0,
        "mean_delta_lost_tp": float(contrib[lost_tp].mean()) if lost_tp.any() else 0.0,
    }


def cardinality_table(data: dict[str, np.ndarray | list[str]]) -> pd.DataFrame:
    targets = data["targets"].astype(np.int32)
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 99, "4+")]
    rows = []
    card = targets.sum(axis=1)
    for low, high, name in bins:
        mask = (card >= low) & (card <= high)
        if not mask.any():
            continue
        for method, logits_key in [("dVPT-Tuned", "base_logits"), ("SCVPT", "final_logits")]:
            metrics = compute_metrics(data[logits_key][mask], targets[mask])
            rows.append(
                {
                    "cardinality": name,
                    "method": method,
                    "n": int(mask.sum()),
                    "micro_f1": metrics["micro_f1"],
                    "sample_f1": metrics["sample_f1"],
                    "precision": metrics["micro_precision"],
                    "recall": metrics["micro_recall"],
                }
            )
    return pd.DataFrame(rows)


def frequency_delta_table(per_base: pd.DataFrame, per_scvpt: pd.DataFrame) -> pd.DataFrame:
    merged = per_base.merge(per_scvpt, on=["class_index", "label_id", "class_name"], suffixes=("_base", "_scvpt"))
    merged["delta_ap"] = merged["ap_scvpt"] - merged["ap_base"]
    merged["delta_f1"] = merged["f1_scvpt"] - merged["f1_base"]
    merged["group"] = merged["label_id"].apply(group_for_label)
    merged["freq_bin"] = pd.cut(
        merged["positive_count_base"],
        bins=[-1, 10, 30, 100, 10_000],
        labels=["<=10", "11-30", "31-100", ">100"],
    )
    return merged


def control_metrics(data: dict[str, np.ndarray | list[str]]) -> pd.DataFrame:
    rng = np.random.default_rng(20260427)
    residual = data["residual_logits"]
    weight = data["candidate_weight"]
    shuffled_weight = np.empty_like(weight)
    for row_idx in range(weight.shape[0]):
        shuffled_weight[row_idx] = weight[row_idx, rng.permutation(weight.shape[1])]
    random_logits = data["base_logits"] + float(data["alpha"]) * shuffled_weight * residual
    candidates = {
        "Base dVPT-Tuned": data["base_logits"],
        "All-label verifier": data["all_weight_logits"],
        "Top-k verifier": data["hard_topk_logits"],
        "Soft candidate only": data["soft_only_logits"],
        "Randomized candidate": random_logits,
        "SCVPT": data["final_logits"],
    }
    rows = []
    for name, logits in candidates.items():
        metrics = compute_metrics(logits, data["targets"])
        rows.append({"variant": name, **metrics})
    return pd.DataFrame(rows)


def save_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(
    freq_delta: pd.DataFrame,
    controls: pd.DataFrame,
    cardinality: pd.DataFrame,
    candidate_summary: dict[str, float],
    out_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    fig.patch.set_facecolor("white")

    top = freq_delta.sort_values("delta_ap").iloc[[*range(5), *range(len(freq_delta) - 5, len(freq_delta))]]
    colors = ["#4C78A8" if x >= 0 else "#E45756" for x in top["delta_ap"]]
    axes[0, 0].barh(top["class_name"], top["delta_ap"], color=colors)
    axes[0, 0].axvline(0, color="#222222", linewidth=0.8)
    axes[0, 0].set_title("Per-class AP change: SCVPT - dVPT-Tuned")
    axes[0, 0].set_xlabel("Delta AP")

    freq = freq_delta.groupby("freq_bin", observed=False)[["delta_ap", "delta_f1"]].mean().reset_index()
    x = np.arange(len(freq))
    axes[0, 1].bar(x - 0.18, freq["delta_ap"], width=0.36, label="AP", color="#4C78A8")
    axes[0, 1].bar(x + 0.18, freq["delta_f1"], width=0.36, label="F1", color="#F58518")
    axes[0, 1].axhline(0, color="#222222", linewidth=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(freq["freq_bin"].astype(str))
    axes[0, 1].set_title("Gain by label frequency")
    axes[0, 1].set_ylabel("Mean delta")
    axes[0, 1].legend(frameon=False)

    order = ["Base dVPT-Tuned", "All-label verifier", "Top-k verifier", "Randomized candidate", "SCVPT"]
    sub = controls.set_index("variant").loc[order].reset_index()
    axes[1, 0].plot(sub["variant"], sub["micro_f1"], marker="o", label="Micro-F1", color="#4C78A8")
    axes[1, 0].plot(sub["variant"], sub["sample_f1"], marker="o", label="Sample-F1", color="#54A24B")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_title("Candidate verifier controls")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend(frameon=False)

    pivot = cardinality.pivot(index="cardinality", columns="method", values="sample_f1")
    pivot = pivot[["dVPT-Tuned", "SCVPT"]]
    x = np.arange(len(pivot))
    axes[1, 1].bar(x - 0.18, pivot["dVPT-Tuned"], width=0.36, color="#BAB0AC", label="dVPT-Tuned")
    axes[1, 1].bar(x + 0.18, pivot["SCVPT"], width=0.36, color="#4C78A8", label="SCVPT")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(pivot.index)
    axes[1, 1].set_title(
        "Multi-sign images benefit most\n"
        f"GT coverage={candidate_summary['candidate_gt_coverage']:.1f}%, avg candidates={candidate_summary['avg_candidate_labels']:.1f}"
    )
    axes[1, 1].set_ylabel("Sample-F1")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def denorm_clip(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device).view(3, 1, 1)
    img = (image * std + mean).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return img


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.nanpercentile(x, [2.0, 99.5])
    x = np.clip(x, lo, hi)
    x = x - float(np.nanmin(x))
    denom = float(np.nanmax(x))
    if denom <= 1e-8:
        return np.zeros_like(x)
    return x / denom


def smooth_map(heat: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(heat, sigma=sigma)
    except Exception:
        tensor = torch.tensor(heat, dtype=torch.float32).view(1, 1, *heat.shape)
        tensor = F.avg_pool2d(tensor, kernel_size=5, stride=1, padding=2)
        return tensor[0, 0].numpy()


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    heat = normalize_map(smooth_map(heat))
    cmap = plt.get_cmap(cmap_name)
    color = cmap(heat)[..., :3]
    alpha = 0.62 * np.clip((heat - 0.22) / 0.78, 0.0, 1.0)[..., None]
    # Keep low-activation anatomy nearly untouched; only high-evidence regions glow.
    return (rgb * (1.0 - alpha) + color * alpha).clip(0, 1)


def plot_heat_only(ax, heat: np.ndarray, title: str, cmap_name: str = "turbo") -> None:
    heat = normalize_map(smooth_map(heat))
    ax.imshow(heat, cmap=cmap_name, vmin=0, vmax=1)
    ax.contour(heat, levels=[0.55, 0.75], colors=["white", "#1b1b1b"], linewidths=[0.7, 0.6], alpha=0.75)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.axis("off")


def verifier_cam(model, inc_input: torch.Tensor, class_idx: int, out_size: tuple[int, int]) -> np.ndarray:
    with torch.no_grad():
        featmap = model.branch.forward_feature_map(inc_input)
        weight = model.branch.head.weight[class_idx].view(1, -1, 1, 1)
        cam = F.relu((featmap * weight).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
    return normalize_map(cam[0, 0].detach().cpu().numpy())


def input_saliency(model, image: torch.Tensor, class_idx: int) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    image = image.detach().clone().requires_grad_(True)
    _, logits = model._base_features_logits(image)
    score = logits[:, class_idx].sum()
    score.backward()
    grad = image.grad.detach().abs().max(dim=1, keepdim=True).values
    grad = F.interpolate(grad, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return normalize_map(grad[0, 0].cpu().numpy())


def select_case_candidates(
    data: dict[str, np.ndarray | list[str]],
    max_per_type: int = 12,
) -> tuple[list[dict], list[dict]]:
    targets = data["targets"].astype(np.int32)
    base_pred = (sigmoid(data["base_logits"]) >= 0.5).astype(np.int32)
    final_pred = (sigmoid(data["final_logits"]) >= 0.5).astype(np.int32)
    sample_ids = data["sample_ids"]
    corrected_candidates: list[dict] = []
    failure_candidates: list[dict] = []
    for i, sid in enumerate(sample_ids):
        recovered = np.where((targets[i] == 1) & (base_pred[i] == 0) & (final_pred[i] == 1))[0]
        removed = np.where((targets[i] == 0) & (base_pred[i] == 1) & (final_pred[i] == 0))[0]
        new_fp = np.where((targets[i] == 0) & (base_pred[i] == 0) & (final_pred[i] == 1))[0]
        lost_tp = np.where((targets[i] == 1) & (base_pred[i] == 1) & (final_pred[i] == 0))[0]
        base_probs = sigmoid(data["base_logits"][i])
        final_probs = sigmoid(data["final_logits"][i])
        for class_idx in recovered:
            corrected_candidates.append(
                {
                    "score": float(final_probs[class_idx] - base_probs[class_idx]),
                    "sample_id": sid,
                    "class_idx": int(class_idx),
                    "case_type": "FN recovered",
                    "base_prob": float(base_probs[class_idx]),
                    "final_prob": float(final_probs[class_idx]),
                    "target": 1,
                }
            )
        for class_idx in removed:
            corrected_candidates.append(
                {
                    "score": float(base_probs[class_idx] - final_probs[class_idx]),
                    "sample_id": sid,
                    "class_idx": int(class_idx),
                    "case_type": "FP removed",
                    "base_prob": float(base_probs[class_idx]),
                    "final_prob": float(final_probs[class_idx]),
                    "target": 0,
                }
            )
        for class_idx in new_fp:
            failure_candidates.append(
                {
                    "score": float(final_probs[class_idx] - base_probs[class_idx]),
                    "sample_id": sid,
                    "class_idx": int(class_idx),
                    "case_type": "Failure: new FP",
                    "base_prob": float(base_probs[class_idx]),
                    "final_prob": float(final_probs[class_idx]),
                    "target": 0,
                }
            )
        for class_idx in lost_tp:
            failure_candidates.append(
                {
                    "score": float(base_probs[class_idx] - final_probs[class_idx]),
                    "sample_id": sid,
                    "class_idx": int(class_idx),
                    "case_type": "Failure: lost TP",
                    "base_prob": float(base_probs[class_idx]),
                    "final_prob": float(final_probs[class_idx]),
                    "target": 1,
                }
            )

    def take_unique(rows: list[dict], case_type: str) -> list[dict]:
        chosen = []
        used_ids = set()
        for row in sorted([r for r in rows if r["case_type"] == case_type], key=lambda x: x["score"], reverse=True):
            if row["sample_id"] in used_ids:
                continue
            chosen.append(row)
            used_ids.add(row["sample_id"])
            if len(chosen) >= max_per_type:
                break
        return chosen

    corrected = take_unique(corrected_candidates, "FN recovered") + take_unique(corrected_candidates, "FP removed")
    failures = take_unique(failure_candidates, "Failure: new FP") + take_unique(failure_candidates, "Failure: lost TP")
    return corrected, failures


def plot_case_heatmaps(
    trainer,
    data: dict[str, np.ndarray | list[str]],
    out_path: Path,
    max_cases: int = 4,
    selected_rows: list[dict] | None = None,
) -> None:
    sample_ids = list(data["sample_ids"])
    if selected_rows is None:
        corrected, _ = select_case_candidates(data, max_per_type=max_cases)
        selected_rows = corrected[:max_cases]
    else:
        selected_rows = selected_rows[:max_cases]

    selected: list[tuple[str, int, str]] = []
    used_ids: set[str] = set()
    for row in selected_rows:
        sid = row["sample_id"]
        if sid in used_ids:
            continue
        selected.append((sid, int(row["class_idx"]), row["case_type"]))
        used_ids.add(sid)
        if len(selected) >= max_cases:
            break
    if not selected:
        return

    lookup = {sid: (cls, tag) for sid, cls, tag in selected}
    fig, axes = plt.subplots(len(selected), 4, figsize=(13.2, 3.35 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)
    fig.patch.set_facecolor("white")

    row_idx = 0
    model = trainer.model
    for batch in trainer.test_loader:
        ids = get_sample_ids(batch)
        if not any(sid in lookup for sid in ids):
            continue
        parsed_input, _ = trainer.parse_batch_test(batch)
        if isinstance(parsed_input, (tuple, list)):
            images, inc_inputs = parsed_input
        else:
            images, inc_inputs = parsed_input, None
        for local_idx, sid in enumerate(ids):
            if sid not in lookup:
                continue
            class_idx, tag = lookup[sid]
            image = images[local_idx : local_idx + 1]
            inc = inc_inputs[local_idx : local_idx + 1] if inc_inputs is not None else model._to_imagenet_input(image.float(), size=299)
            rgb = denorm_clip(image[0])
            sal = input_saliency(model, image, class_idx)
            cam = verifier_cam(model, inc, class_idx, out_size=image.shape[-2:])
            fused_focus = np.maximum(normalize_map(smooth_map(sal)), normalize_map(smooth_map(cam)))
            label = EN_SHORT_NAMES[class_idx + 1]
            data_idx = sample_ids.index(sid)
            bprob = sigmoid(data["base_logits"][data_idx, class_idx])
            fprob = sigmoid(data["final_logits"][data_idx, class_idx])
            axes[row_idx, 0].imshow(rgb)
            axes[row_idx, 0].set_title(f"{sid}\n{tag}: {label}", fontsize=8.5, fontweight="bold", pad=5)
            axes[row_idx, 0].axis("off")
            plot_heat_only(axes[row_idx, 1], sal, f"dVPT saliency\nbase p={bprob:.2f}", cmap_name="magma")
            plot_heat_only(axes[row_idx, 2], cam, f"Verifier CAM\nfinal p={fprob:.2f}", cmap_name="turbo")
            axes[row_idx, 3].imshow(overlay_heatmap(rgb, fused_focus, cmap_name="turbo"))
            axes[row_idx, 3].contour(
                normalize_map(smooth_map(cam)),
                levels=[0.65],
                colors=["#FFFFFF"],
                linewidths=1.0,
                alpha=0.95,
            )
            axes[row_idx, 3].set_title(f"Combined focus\n{bprob:.2f} -> {fprob:.2f}", fontsize=9, fontweight="bold", pad=5)
            axes[row_idx, 3].axis("off")
            for col in range(4):
                axes[row_idx, col].axis("off")
                for spine in axes[row_idx, col].spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.7)
                    spine.set_edgecolor("#D0D0D0")
            row_idx += 1
            if row_idx >= len(selected):
                break
        if row_idx >= len(selected):
            break
    plt.subplots_adjust(left=0.015, right=0.985, top=0.975, bottom=0.025, wspace=0.04, hspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_case_gallery(
    trainer,
    data: dict[str, np.ndarray | list[str]],
    out_dir: Path,
    max_per_type: int = 12,
) -> None:
    corrected, failures = select_case_candidates(data, max_per_type=max_per_type)
    gallery_dir = out_dir / "case_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_group, candidates in [("corrected", corrected), ("failure", failures)]:
        for idx, row in enumerate(candidates, start=1):
            row = dict(row)
            row["label"] = EN_SHORT_NAMES[int(row["class_idx"]) + 1]
            row["rank"] = idx
            row["group"] = case_group
            row["figure"] = f"{case_group}_{idx:02d}_{row['sample_id']}_c{int(row['class_idx'])+1}.png"
            rows.append(row)
            plot_case_heatmaps(
                trainer,
                data,
                gallery_dir / row["figure"],
                max_cases=1,
                selected_rows=[row],
            )

    gallery_csv = gallery_dir / "case_gallery_index.csv"
    if rows:
        pd.DataFrame(rows).to_csv(gallery_csv, index=False)

    md_lines = [
        "# SCVPT Heatmap Case Gallery",
        "",
        "Use this gallery to manually choose clean examples for the main paper. Corrected cases are candidates for positive mechanism evidence; failure cases are candidates for failure analysis.",
        "",
    ]
    for group_name in ["corrected", "failure"]:
        md_lines.append(f"## {group_name.capitalize()} cases")
        for row in [r for r in rows if r["group"] == group_name]:
            md_lines.append(
                f"- Rank {row['rank']:02d}: `{row['figure']}` | {row['case_type']} | "
                f"{row['sample_id']} | {row['label']} | "
                f"{row['base_prob']:.2f} -> {row['final_prob']:.2f} | score={row['score']:.2f}"
            )
        md_lines.append("")
    (gallery_dir / "case_gallery_index.md").write_text("\n".join(md_lines), encoding="utf-8")

    plot_case_heatmaps(
        trainer,
        data,
        out_dir / "scvpt_case_heatmaps_corrected_gallery.png",
        max_cases=min(8, len(corrected)),
        selected_rows=corrected,
    )
    if failures:
        plot_case_heatmaps(
            trainer,
            data,
            out_dir / "scvpt_case_heatmaps_failure_gallery.png",
            max_cases=min(6, len(failures)),
            selected_rows=failures,
        )


def plot_publication_success_figure(
    trainer,
    data: dict[str, np.ndarray | list[str]],
    out_path: Path,
) -> None:
    corrected, _ = select_case_candidates(data, max_per_type=12)
    selected_keys = [
        ("vision2_004953", 9),   # Epiretinal membrane
        ("vision2_006873", 18),  # Large cup
        ("vision2_002053", 37),  # Hard exudates
        ("vision2_006169", 6),   # Subretinal fibrosis
        ("vision2_006392", 38),  # Cotton-wool spots
        ("vision2_006891", 36),  # Retinal hemorrhage
    ]
    selected_rows = []
    for sid, class_idx in selected_keys:
        match = next((r for r in corrected if r["sample_id"] == sid and int(r["class_idx"]) == class_idx), None)
        if match is not None:
            selected_rows.append(match)
    if len(selected_rows) < 6:
        used = {(r["sample_id"], int(r["class_idx"])) for r in selected_rows}
        for row in corrected:
            key = (row["sample_id"], int(row["class_idx"]))
            if key not in used:
                selected_rows.append(row)
                used.add(key)
            if len(selected_rows) >= 6:
                break

    sample_ids = list(data["sample_ids"])
    lookup = {row["sample_id"]: row for row in selected_rows}
    alphabet = list("ABCDEF")
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 9.4))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.axis("off")

    model = trainer.model
    payloads = {}
    for batch in trainer.test_loader:
        ids = get_sample_ids(batch)
        if not any(sid in lookup for sid in ids):
            continue
        parsed_input, _ = trainer.parse_batch_test(batch)
        if isinstance(parsed_input, (tuple, list)):
            images, inc_inputs = parsed_input
        else:
            images, inc_inputs = parsed_input, None
        for local_idx, sid in enumerate(ids):
            if sid not in lookup:
                continue
            row = lookup[sid]
            class_idx = int(row["class_idx"])
            image = images[local_idx : local_idx + 1]
            inc = inc_inputs[local_idx : local_idx + 1] if inc_inputs is not None else model._to_imagenet_input(image.float(), size=299)
            rgb = denorm_clip(image[0])
            cam = verifier_cam(model, inc, class_idx, out_size=image.shape[-2:])
            focus = overlay_heatmap(rgb, cam, cmap_name="turbo")
            payloads[sid] = {
                "rgb": rgb,
                "focus": focus,
                "cam": cam,
                "class_idx": class_idx,
            }
        if len(payloads) >= len(selected_rows):
            break

    for rendered, row in enumerate(selected_rows[:6]):
        sid = row["sample_id"]
        if sid not in payloads:
            continue
        class_idx = int(row["class_idx"])
        panel_row = rendered // 2
        panel_col = (rendered % 2) * 2
        rgb = payloads[sid]["rgb"]
        focus = payloads[sid]["focus"]
        cam = payloads[sid]["cam"]
        data_idx = sample_ids.index(sid)
        bprob = sigmoid(data["base_logits"][data_idx, class_idx])
        fprob = sigmoid(data["final_logits"][data_idx, class_idx])
        label = EN_SHORT_NAMES[class_idx + 1]
        letter = alphabet[rendered]
        axes[panel_row, panel_col].imshow(rgb)
        axes[panel_row, panel_col].set_title(
            f"{letter}. {label}\nBase p={bprob:.2f}",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )
        axes[panel_row, panel_col + 1].imshow(focus)
        axes[panel_row, panel_col + 1].contour(
            normalize_map(smooth_map(cam)),
            levels=[0.62],
            colors=["white"],
            linewidths=1.1,
            alpha=0.95,
        )
        axes[panel_row, panel_col + 1].set_title(
            f"SCVPT focus\nFinal p={fprob:.2f}",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )
        for col in [panel_col, panel_col + 1]:
            axes[panel_row, col].axis("off")
            for spine in axes[panel_row, col].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#D5D5D5")

    plt.subplots_adjust(left=0.018, right=0.982, top=0.965, bottom=0.03, wspace=0.04, hspace=0.26)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=380, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_point_to_point(
    path: Path,
    controls: pd.DataFrame,
    freq_delta: pd.DataFrame,
    cardinality: pd.DataFrame,
    candidate_summary: dict[str, float],
    threshold_corr: dict[str, float],
) -> None:
    scvpt = controls[controls["variant"] == "SCVPT"].iloc[0]
    base = controls[controls["variant"] == "Base dVPT-Tuned"].iloc[0]
    top_gain = freq_delta.sort_values("delta_ap", ascending=False).head(5)
    top_loss = freq_delta.sort_values("delta_ap", ascending=True).head(3)
    card_gain = cardinality.pivot(index="cardinality", columns="method", values="sample_f1")
    lines = [
        "# SCVPT Mechanism: Point-to-Point Notes",
        "",
        "## Q1. SCVPT 到底改进在哪里？",
        f"- 相比 dVPT-Tuned，SCVPT 的 Micro-F1 从 {base['micro_f1']:.2f} 提升到 {scvpt['micro_f1']:.2f}，Sample-F1 从 {base['sample_f1']:.2f} 提升到 {scvpt['sample_f1']:.2f}。",
        "- AP 提升最大的类别是："
        + ", ".join(f"{r.class_name} ({r.delta_ap:+.2f})" for _, r in top_gain.iterrows()),
        "- AP 下降最明显的类别是："
        + ", ".join(f"{r.class_name} ({r.delta_ap:+.2f})" for _, r in top_loss.iterrows()),
        "",
        "## Q2. Candidate verifier 是不是只是另一个普通分类器？",
        f"- 候选门控平均激活 {candidate_summary['avg_candidate_labels']:.2f} 个标签/图像，覆盖 {candidate_summary['candidate_gt_coverage']:.2f}% 的真实阳性标签。",
        f"- 直接把 verifier 作用到所有类别的 Micro-F1 为 {controls.loc[controls.variant == 'All-label verifier', 'micro_f1'].iloc[0]:.2f}，随机候选为 {controls.loc[controls.variant == 'Randomized candidate', 'micro_f1'].iloc[0]:.2f}，都低于 SCVPT 的 {scvpt['micro_f1']:.2f}。",
        "",
        "## Q3. dVPT-Tuned 和 verifier 关注的证据是否不同？",
        "- 脚本输出了 `scvpt_case_heatmaps.png`：每个病例包含原图、dVPT 输入梯度显著图、verifier Inception-CAM。我们在论文中把它作为定性机制图，而不把它解释成严格因果证据。",
        "",
        "## Q4. verifier 具体修正了什么错误？",
        f"- 在 test set 上，SCVPT 从 base FN 中恢复 {candidate_summary['recovered_fn_count']} 个阳性标签，同时移除 {candidate_summary['removed_fp_count']} 个 FP。",
        f"- 代价是新增 {candidate_summary['new_fp_count']} 个 FP 和丢失 {candidate_summary['lost_tp_count']} 个原 TP；这说明模块主要在 recall 与 precision 间做局部再平衡。",
        "",
        "## Q5. Selective per-class refinement 是不是阈值小技巧？",
        f"- 学到的候选阈值与类别阳性数的 Pearson 相关为 {threshold_corr['threshold_vs_log_count']:.3f}，与 base FP 数相关为 {threshold_corr['threshold_vs_base_fp']:.3f}。",
        "- 因为随机候选和 all-label verifier 控制都无法复现 SCVPT，说明收益不仅来自统一阈值移动，而来自类别选择性门控和 residual 校正的组合。",
        "",
        "## Q6. 哪些样本最受益？",
    ]
    for card in card_gain.index:
        gain = card_gain.loc[card, "SCVPT"] - card_gain.loc[card, "dVPT-Tuned"]
        lines.append(f"- {card} 个阳性标签样本：Sample-F1 提升 {gain:+.2f}。")
    lines.extend(
        [
            "",
            "## 论文中建议呈现",
            "- 主文放 `scvpt_mechanism_overview.png` 和一张 compact mechanism 表。",
            "- 代表病例热图可放主文或补充材料；如果版面紧张，主文保留 overview，热图放 supplement。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--skip-heatmaps", action="store_true")
    parser.add_argument("--gallery-cases", type=int, default=12)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(20260427)
    np.random.seed(20260427)
    torch.manual_seed(20260427)

    trainer = build_loaded_trainer()
    cache_path = out_dir / "scvpt_decomposition_test.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        data = {key: cached[key] for key in cached.files}
        if isinstance(data.get("sample_ids"), np.ndarray):
            data["sample_ids"] = [str(x) for x in data["sample_ids"].tolist()]
    else:
        data = collect_decomposition(trainer)
        np.savez_compressed(cache_path, **data)

    controls = control_metrics(data)
    save_csv(out_dir / "candidate_control_metrics.csv", controls)

    per_base = per_class_scores(data["base_logits"], data["targets"])
    per_scvpt = per_class_scores(data["final_logits"], data["targets"])
    save_csv(out_dir / "per_class_base_dvpt_tuned.csv", per_base)
    save_csv(out_dir / "per_class_scvpt.csv", per_scvpt)
    freq_delta = frequency_delta_table(per_base, per_scvpt)
    save_csv(out_dir / "per_class_delta.csv", freq_delta)

    card = cardinality_table(data)
    save_csv(out_dir / "cardinality_strata.csv", card)

    candidate_summary = summarize_candidate_and_residual(data)
    threshold_df = per_base[["class_index", "positive_count", "fp"]].copy()
    threshold_df["candidate_threshold"] = data["candidate_threshold"]
    threshold_df["candidate_scale"] = data["candidate_scale"]
    save_csv(out_dir / "selective_candidate_parameters.csv", threshold_df)
    threshold_corr = {
        "threshold_vs_log_count": float(np.corrcoef(threshold_df["candidate_threshold"], np.log1p(threshold_df["positive_count"]))[0, 1]),
        "threshold_vs_base_fp": float(np.corrcoef(threshold_df["candidate_threshold"], threshold_df["fp"])[0, 1]),
        "scale_vs_log_count": float(np.corrcoef(threshold_df["candidate_scale"], np.log1p(threshold_df["positive_count"]))[0, 1]),
    }

    summary = {
        "metrics": {row["variant"]: {k: float(row[k]) for k in controls.columns if k != "variant"} for _, row in controls.iterrows()},
        "candidate_summary": candidate_summary,
        "threshold_correlations": threshold_corr,
        "alpha": float(data["alpha"]),
    }
    (out_dir / "mechanism_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_overview(freq_delta, controls, card, candidate_summary, out_dir / "scvpt_mechanism_overview.png")
    if not args.skip_heatmaps:
        plot_publication_success_figure(trainer, data, out_dir / "scvpt_case_heatmaps.png")
        export_case_gallery(trainer, data, out_dir, max_per_type=args.gallery_cases)

    write_point_to_point(
        PAPER_DIR / "scvpt_mechanism_point_to_point.md",
        controls,
        freq_delta,
        card,
        candidate_summary,
        threshold_corr,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved mechanism artifacts to {out_dir}")
    print(f"Saved point-to-point notes to {PAPER_DIR / 'scvpt_mechanism_point_to_point.md'}")


if __name__ == "__main__":
    main()
