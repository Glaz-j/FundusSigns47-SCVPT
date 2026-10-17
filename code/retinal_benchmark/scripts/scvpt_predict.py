#!/usr/bin/env python3
"""Run inference with a packaged SCVPT checkpoint.

This script is intentionally inference-only. It does not require the FundusSigns
training set and it does not call the trainer/evaluator stack. Users provide a
CSV manifest or an image directory, and the script writes per-image 47-class
probabilities and thresholded predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
COOP = ROOT / "external" / "CoOp"
DASSL = ROOT / "external" / "Dassl.pytorch"
sys.path.insert(0, str(COOP))
sys.path.insert(0, str(DASSL))
sys.path.insert(0, str(ROOT))

import open_clip  # noqa: E402
from train import setup_cfg  # noqa: E402
from trainers.openclip_coop_multilabel_cached import parse_openclip_backbone  # noqa: E402
from trainers.openclip_vpt_deep_only_multilabel import OpenCLIPVPTDeepOnlyModel  # noqa: E402
from trainers.openclip_vptdeep_visualresidual_multilabel import (  # noqa: E402
    FrozenDeepVPTVisualResidualModel,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_label_map(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    return [label_map[str(i)] for i in range(1, len(label_map) + 1)]


def build_cfg(config: Path, root: Path, device: str):
    args = SimpleNamespace(
        root=str(root),
        output_dir=str(root / "tmp_scvpt_predict"),
        resume="",
        seed=-1,
        source_domains=None,
        target_domains=None,
        transforms=None,
        config_file=str(config),
        dataset_config_file="",
        trainer="OpenCLIPVPTDeepVisualResidualMultiLabel",
        backbone="",
        head="",
        eval_only=True,
        model_dir="",
        load_epoch=None,
        no_train=True,
        opts=[
            "MODEL.BACKBONE.NAME",
            "openclip:ViT-gopt-16-SigLIP2-256:none",
            "TRAINER.VPT.FROZEN_MODEL_PATH",
            "",
            "TRAINER.STAGE2VIS.TEACHER_CKPT",
            "",
            "TRAINER.STAGE2VIS.INIT_FROM_TEACHER",
            "False",
            "TRAINER.STAGE2VIS.INCEPTION_PRETRAINED",
            "False",
            "TRAINER.STAGE2VIS.RELEASE_INFERENCE",
            "True",
            "USE_CUDA",
            "True" if device.startswith("cuda") else "False",
        ],
    )
    return setup_cfg(args)


def load_model(config: Path, checkpoint: Path, label_names: list[str], device: torch.device):
    cfg = build_cfg(config, ROOT, str(device))
    model_name, pretrained_name = parse_openclip_backbone(cfg.MODEL.BACKBONE.NAME)
    if str(pretrained_name).strip().lower() in {"", "none", "null", "false"}:
        pretrained_name = None
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_name,
        device="cpu",
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    base_model = OpenCLIPVPTDeepOnlyModel(cfg, label_names, clip_model, tokenizer)
    model = FrozenDeepVPTVisualResidualModel(cfg, base_model)

    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_obj["state_dict"] if isinstance(checkpoint_obj, dict) else checkpoint_obj
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading checkpoint: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[warn] Unexpected keys while loading checkpoint: {len(unexpected)}", file=sys.stderr)

    model.to(device)
    model.eval()
    return model, preprocess_val


def rows_from_manifest(path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(path)
    rows = []
    for idx, row in df.iterrows():
        image_path = ""
        for col in ("image_path", "linked_image_path", "source_image_path", "path", "filepath"):
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                image_path = str(row[col]).strip()
                break
        if not image_path:
            raise ValueError(
                "Manifest must contain one of: image_path, linked_image_path, source_image_path, path, filepath"
            )
        sample_id = str(row.get("sample_id", Path(image_path).stem))
        rows.append({"sample_id": sample_id, "image_path": image_path})
    return rows


def rows_from_image_dir(path: Path) -> list[dict[str, str]]:
    rows = []
    for image_path in sorted(path.rglob("*")):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            rows.append({"sample_id": image_path.stem, "image_path": str(image_path)})
    if not rows:
        raise FileNotFoundError(f"No images found under {path}")
    return rows


def rows_from_images(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for image_path in paths:
        rows.append({"sample_id": image_path.stem, "image_path": str(image_path)})
    return rows


def batched(rows: list[dict[str, str]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def predict(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    label_names = load_label_map(args.label_map)
    model, preprocess = load_model(args.config, args.checkpoint, label_names, device)

    if args.manifest:
        rows = rows_from_manifest(args.manifest)
    elif args.image_dir:
        rows = rows_from_image_dir(args.image_dir)
    else:
        rows = rows_from_images(args.image)

    output_rows = []
    with torch.no_grad():
        for batch_rows in batched(rows, args.batch_size):
            images = []
            for row in batch_rows:
                with Image.open(row["image_path"]) as img:
                    images.append(preprocess(img.convert("RGB")))
            batch = torch.stack(images, dim=0).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for row, prob in zip(batch_rows, probs):
                pred_ids = [str(i + 1) for i, value in enumerate(prob) if value >= args.threshold]
                top_idx = np.argsort(-prob)[: args.topk]
                output_row = {
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "positive_label_ids": "|".join(pred_ids),
                    "topk": "|".join(
                        f"{idx + 1}:{label_names[idx]}:{prob[idx]:.6f}" for idx in top_idx
                    ),
                }
                for idx, value in enumerate(prob):
                    output_row[f"prob_{idx + 1:02d}"] = f"{float(value):.8f}"
                    output_row[f"pred_{idx + 1:02d}"] = int(value >= args.threshold)
                output_rows.append(output_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else ["sample_id", "image_path"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote predictions for {len(output_rows)} images to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="SCVPT inference for FundusSigns-47 labels")
    parser.add_argument("--config", type=Path, required=True, help="SCVPT release config YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Inference checkpoint")
    parser.add_argument("--label-map", type=Path, required=True, help="47-class label map JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="CSV with image_path and optional sample_id")
    source.add_argument("--image-dir", type=Path, help="Directory of images to run recursively")
    source.add_argument("--image", type=Path, action="append", help="One image path; can be repeated")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction CSV")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
