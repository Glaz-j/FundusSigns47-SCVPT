"""Training utilities for benchmark runs."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import load_yaml
from .datasets import VisionManifestDataset, _parse_label_vector
from .models import create_model


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1.0 - probs
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
        loss *= one_sided_w
        return -loss.mean()


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = ce * alpha_t * ((1 - p_t) ** self.gamma)
        return loss.mean()


class SigmoidFocalLossElementwise(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return ce * alpha_t * ((1 - p_t) ** self.gamma)


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )


class ClippedWeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, max_pos_weight: float):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight.clamp(max=float(max_pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )


class MixedPosWeightBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, tail_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("tail_mask", tail_mask)
        self.register_buffer("head_mask", 1.0 - tail_mask)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pos = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        loss = bce * self.head_mask + pos * self.tail_mask
        return loss.mean()


class TailOnlyWeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, tail_mask: torch.Tensor, max_pos_weight: float | None = None):
        super().__init__()
        if max_pos_weight is not None:
            pos_weight = pos_weight.clamp(max=float(max_pos_weight))
        self.register_buffer("effective_pos_weight", 1.0 + (pos_weight - 1.0) * tail_mask.squeeze(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.effective_pos_weight,
        )


class ClassAdaptiveLoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        head_mask: torch.Tensor,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        self.focal = SigmoidFocalLossElementwise(gamma=focal_gamma, alpha=focal_alpha)
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("head_mask", head_mask)
        self.register_buffer("tail_mask", 1.0 - head_mask)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        pos_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        loss = focal_loss * self.head_mask + pos_loss * self.tail_mask
        return loss.mean()


class DistributionBalancedLoss(nn.Module):
    def __init__(
        self,
        class_freq: torch.Tensor,
        neg_class_freq: torch.Tensor,
        focal_gamma: float = 2.0,
        focal_scale: float = 2.0,
        map_alpha: float = 10.0,
        map_beta: float = 0.2,
        map_gamma: float = 0.1,
        neg_scale: float = 5.0,
        init_bias: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("class_freq", class_freq)
        self.register_buffer("neg_class_freq", neg_class_freq)
        self.train_num = float((class_freq[0] + neg_class_freq[0]).item())
        self.register_buffer("freq_inv", 1.0 / class_freq.clamp_min(1.0))
        init = -torch.log(self.train_num / class_freq.clamp_min(1.0) - 1.0) * init_bias / neg_scale
        self.register_buffer("init_bias", init)
        self.focal_gamma = focal_gamma
        self.focal_scale = focal_scale
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_gamma = map_gamma
        self.neg_scale = neg_scale

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        repeat_rate = torch.sum(targets * self.freq_inv.unsqueeze(0), dim=1, keepdim=True).clamp_min(1e-6)
        pos_weight = self.freq_inv.unsqueeze(0) / repeat_rate
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha

        logits = logits + self.init_bias.unsqueeze(0)
        logits = logits * (1.0 - targets) * self.neg_scale + logits * targets
        weight = weight / self.neg_scale * (1.0 - targets) + weight * targets

        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.focal_gamma)
        loss = self.focal_scale * focal * bce * weight
        return loss.mean()


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def default_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_dataloaders(data_cfg: dict, train_cfg: dict):
    transform = default_transforms(train_cfg["image_size"])
    manifest_path = data_cfg["manifest_path"]
    image_cache_dir = data_cfg.get("image_cache_dir")
    image_cache_ext = data_cfg.get("image_cache_ext", "jpg")
    num_workers = int(train_cfg.get("num_workers", 8))
    datasets = {
        split: VisionManifestDataset(
            manifest_path=manifest_path,
            split=split,
            num_classes=data_cfg["num_classes"],
            transform=transform,
            image_cache_dir=image_cache_dir,
            image_cache_ext=image_cache_ext,
        )
        for split in ["train", "val", "test"]
    }
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loaders = {
        split: DataLoader(
            ds,
            batch_size=train_cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            **loader_kwargs,
        )
        for split, ds in datasets.items()
    }
    return datasets, loaders


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


def collect_logits_targets(model, loader, device):
    model.eval()
    logits_list = []
    target_list = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            target_list.append(targets.cpu())
    if not logits_list:
        return None, None
    logits = torch.cat(logits_list).numpy()
    targets = torch.cat(target_list).numpy()
    return logits, targets


def find_per_class_thresholds(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    grid = np.linspace(0.05, 0.95, 19)
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    for class_idx in range(targets.shape[1]):
        y_true = targets[:, class_idx]
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = probs[:, class_idx]
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            y_pred = (y_prob >= t).astype(np.int32)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)
        thresholds[class_idx] = best_t
    return thresholds


def _compute_class_stats(dataset: VisionManifestDataset):
    num_classes = dataset.num_classes
    positives = torch.zeros(num_classes, dtype=torch.float32)
    for label_text in dataset.df[dataset.label_column].tolist():
        positives += _parse_label_vector(label_text, num_classes)
    total = float(len(dataset))
    negatives = total - positives
    return positives, negatives


def _build_criterion(train_cfg: dict, train_dataset: VisionManifestDataset, device: torch.device):
    loss_name = str(train_cfg.get("loss", "bce")).lower()
    positives, negatives = _compute_class_stats(train_dataset)
    pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
    support_threshold = float(train_cfg.get("class_adaptive_support_threshold", 50))
    head_mask = (positives >= support_threshold).float().to(device).unsqueeze(0)
    explicit_head_indices = train_cfg.get("class_adaptive_head_indices")
    if explicit_head_indices:
        head_mask = torch.zeros_like(head_mask)
        for raw_idx in explicit_head_indices:
            idx = int(raw_idx) - 1
            if 0 <= idx < positives.numel():
                head_mask[0, idx] = 1.0
    tail_mask = (positives < support_threshold).float().to(device).unsqueeze(0)
    if explicit_head_indices:
        tail_mask = 1.0 - head_mask
    if loss_name == "pos_weight_bce":
        return WeightedBCELoss(pos_weight=pos_weight)
    if loss_name == "clipped_pos_weight_bce":
        return ClippedWeightedBCELoss(
            pos_weight=pos_weight,
            max_pos_weight=float(train_cfg.get("pos_weight_cap", 8.0)),
        )
    if loss_name == "tail_pos_weight_bce":
        return TailOnlyWeightedBCELoss(
            pos_weight=pos_weight,
            tail_mask=tail_mask,
            max_pos_weight=train_cfg.get("pos_weight_cap"),
        )
    if loss_name == "mixed_posweight_bce":
        return MixedPosWeightBCELoss(pos_weight=pos_weight, tail_mask=tail_mask)
    if loss_name == "focal":
        return SigmoidFocalLoss(
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            alpha=float(train_cfg.get("focal_alpha", 0.25)),
        )
    if loss_name == "class_adaptive":
        return ClassAdaptiveLoss(
            pos_weight=pos_weight,
            head_mask=head_mask,
            focal_gamma=float(train_cfg.get("focal_gamma", 2.0)),
            focal_alpha=float(train_cfg.get("focal_alpha", 0.25)),
        )
    if loss_name == "distribution_balanced":
        return DistributionBalancedLoss(
            class_freq=positives.to(device),
            neg_class_freq=negatives.to(device),
            focal_gamma=float(train_cfg.get("db_focal_gamma", 2.0)),
            focal_scale=float(train_cfg.get("db_focal_scale", 2.0)),
            map_alpha=float(train_cfg.get("db_map_alpha", 10.0)),
            map_beta=float(train_cfg.get("db_map_beta", 0.2)),
            map_gamma=float(train_cfg.get("db_map_gamma", 0.1)),
            neg_scale=float(train_cfg.get("db_neg_scale", 5.0)),
            init_bias=float(train_cfg.get("db_init_bias", 0.1)),
        )
    if loss_name == "asl":
        return AsymmetricLoss(
            gamma_pos=float(train_cfg.get("asl_gamma_pos", 1.0)),
            gamma_neg=float(train_cfg.get("asl_gamma_neg", 4.0)),
            clip=float(train_cfg.get("asl_clip", 0.05)),
        )
    if loss_name == "asl_mild":
        return AsymmetricLoss(
            gamma_pos=float(train_cfg.get("asl_gamma_pos", 0.0)),
            gamma_neg=float(train_cfg.get("asl_gamma_neg", 2.0)),
            clip=float(train_cfg.get("asl_clip", 0.01)),
        )
    return nn.BCEWithLogitsLoss()


def evaluate(model, loader, device, thresholds: np.ndarray | None = None):
    logits, targets = collect_logits_targets(model, loader, device)
    if logits is None:
        return {
            "mAP": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "macro_auc": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "sample_f1": 0.0,
        }
    probs = 1.0 / (1.0 + np.exp(-logits))
    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
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


def run_training(data_cfg_path: str, model_cfg_path: str, train_cfg_path: str, eval_cfg_path: str, run_dir: str):
    data_cfg = load_yaml(data_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    train_cfg = load_yaml(train_cfg_path)
    eval_cfg = load_yaml(eval_cfg_path)

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loaders = build_dataloaders(data_cfg, train_cfg)
    model = create_model(model_cfg, pretrained=model_cfg.get("pretrained", True))
    model.to(device)

    criterion = _build_criterion(train_cfg, loaders["train"].dataset, device)
    if hasattr(model, "parameter_groups"):
        optimizer_param_groups = model.parameter_groups(train_cfg)
    else:
        optimizer_param_groups = [
            {
                "params": model.parameters(),
                "lr": float(train_cfg["learning_rate"]),
                "weight_decay": float(train_cfg["weight_decay"]),
            }
        ]
    optimizer = torch.optim.AdamW(optimizer_param_groups)
    if hasattr(model, "initialize_from_loader"):
        model.initialize_from_loader(loaders["train"], device)
    mixed_precision = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(mixed_precision)
    eval_every = int(train_cfg.get("eval_every", 1))

    best_val = -1.0
    best_metrics = None
    best_state = None

    history = []
    for epoch in range(int(train_cfg["epochs"])):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for images, targets, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with _autocast_context(mixed_precision):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            num_batches += 1
        train_loss = epoch_loss / max(1, num_batches)
        train_time = time.time() - epoch_start

        val_metrics = {}
        val_time = 0.0
        if (epoch + 1) % eval_every == 0 or epoch + 1 == int(train_cfg["epochs"]):
            val_start = time.time()
            val_metrics = evaluate(model, loaders["val"], device)
            val_time = time.time() - val_start
            if val_metrics["mAP"] > best_val:
                best_val = val_metrics["mAP"]
                best_metrics = val_metrics
                best_state = model.state_dict()

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_time_sec": train_time,
            "val_time_sec": val_time,
            **val_metrics,
        }
        history.append(history_entry)
        epoch_msg = f"[epoch {epoch + 1}/{int(train_cfg['epochs'])}] train_loss={train_loss:.4f} train_time={train_time:.2f}s"
        if val_metrics:
            epoch_msg += (
                f" val_time={val_time:.2f}s"
                f" mAP={val_metrics['mAP']:.4f}"
                f" macro_f1={val_metrics['macro_f1']:.4f}"
                f" micro_f1={val_metrics['micro_f1']:.4f}"
                f" macro_auc={val_metrics['macro_auc']:.4f}"
            )
        print(epoch_msg, flush=True)

    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "last.pt")
    if best_state is not None:
        torch.save(best_state, checkpoint_dir / "best.pt")
        model.load_state_dict(best_state)

    thresholds = None
    if bool(train_cfg.get("search_thresholds", False)):
        val_logits, val_targets = collect_logits_targets(model, loaders["val"], device)
        if val_logits is not None:
            thresholds = find_per_class_thresholds(val_logits, val_targets)

    test_metrics = evaluate(model, loaders["test"], device, thresholds=thresholds)

    (run_path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_path / "best_val_metrics.json").write_text(json.dumps(best_metrics or {}, indent=2), encoding="utf-8")
    (run_path / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    if thresholds is not None:
        (run_path / "thresholds.json").write_text(json.dumps(thresholds.tolist(), indent=2), encoding="utf-8")
    (run_path / "run_meta.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "data_config": data_cfg_path,
                "model_config": model_cfg_path,
                "train_config": train_cfg_path,
                "eval_config": eval_cfg_path,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return test_metrics
