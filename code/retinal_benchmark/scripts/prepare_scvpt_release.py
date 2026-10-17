#!/usr/bin/env python3
"""Prepare a clean SCVPT release bundle without packaging the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_DIR = ROOT / "release" / "FundusSigns-47-SCVPT-20260515"
FINAL_RUN_DIR = (
    ROOT
    / "external/CoOp/output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel"
    / "sig2g_fullinception_candidate_selectiveperclassthr_20ep_cuda3"
)
FINAL_CHECKPOINT = FINAL_RUN_DIR / "visual_residual/model.pth.tar-20"
FINAL_CONFIG = (
    ROOT
    / "external/CoOp/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel"
    / "vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidate_selectiveperclassthr_20ep_vision3.yaml"
)
LABEL_MAP = ROOT / "retinal_benchmark/data_versions/vision3/lesion_label_map.json"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".git",
            ".pytest_cache",
            "cache",
            "output",
            "tensorboard",
            "tmp_*",
        ),
    )


def strip_checkpoint(src: Path, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(src, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
    payload = {
        "state_dict": state_dict,
        "epoch": int(checkpoint.get("epoch", 20)) if isinstance(checkpoint, dict) else 20,
        "model_name": "SCVPT",
        "dataset": "FundusSigns-47",
        "note": "Inference-only checkpoint. Optimizer and scheduler states were removed.",
    }
    torch.save(payload, dst)


def write_release_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """DATASET:
  NAME: "RetinalVision3"
  MANIFEST_PATH: ""
  LABEL_MAP_PATH: "models/lesion_label_map.json"
  MAX_TRAIN_SAMPLES: -1
  MAX_VAL_SAMPLES: -1
  MAX_TEST_SAMPLES: -1
  IMAGE_CACHE_DIR: ""
  IMAGE_CACHE_EXT: "jpg"
  INCEPTION_IMAGE_CACHE_DIR: ""

DATALOADER:
  NUM_WORKERS: 0
  TRAIN_X:
    BATCH_SIZE: 4
  TEST:
    BATCH_SIZE: 4

INPUT:
  SIZE: (256, 256)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["resize", "center_crop", "normalize"]

MODEL:
  BACKBONE:
    NAME: "openclip:ViT-gopt-16-SigLIP2-256:none"

TRAINER:
  STAGE2VIS:
    MODE: "full_inception_candidate_adaptive"
    HIDDEN_DIM: 512
    ALPHA_INIT: 0.05
    PREC: "amp"
    CANDIDATE_THRESHOLD: 0.18
    CANDIDATE_TOPK: 9
    CANDIDATE_SOFT_SCALE: 10.5
    TEACHER_CKPT: ""
    INIT_FROM_TEACHER: False
    INCEPTION_PRETRAINED: False
    RELEASE_INFERENCE: True
  VPT:
    N_CTX: 16
    N_DEEP_LAYERS: -1
    PREC: "amp"
    INIT_STD: 0.02
    DROPOUT: 0.0
    TRAIN_LOGIT_SCALE: True
    TRAIN_PROJ: True
    TRAIN_NORM: True
    LOSS_NAME: "bce"
    FROZEN_MODEL_PATH: ""
""",
        encoding="utf-8",
    )


def write_manifest(release_dir: Path, files: list[Path]) -> None:
    rows = []
    for path in files:
        if path.exists() and path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(release_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "release": "FundusSigns-47-SCVPT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_included": False,
        "files": rows,
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (release_dir / "checksums.sha256").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['sha256']}  {row['path']}\n")


def prepare_release(release_dir: Path, force: bool = False) -> None:
    release_dir.mkdir(parents=True, exist_ok=True)

    # Paper and manuscript assets.
    paper_dst = release_dir / "paper"
    paper_dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "sn-article.pdf",
        "sn-article.tex",
        "sn-bibliography.bib",
        "sn-jnl.cls",
        "sn-mathphys-num.bst",
        "scvpt_mechanism_point_to_point.md",
    ]:
        src = ROOT / "paper/template" / name
        if src.exists():
            copy_file(src, paper_dst / name)

    # Curated code tree. Avoid raw data, caches, and experimental output directories.
    copy_tree(ROOT / "external/CoOp/clip", release_dir / "code/external/CoOp/clip")
    copy_tree(ROOT / "external/CoOp/datasets", release_dir / "code/external/CoOp/datasets")
    copy_tree(ROOT / "external/CoOp/evaluators", release_dir / "code/external/CoOp/evaluators")
    copy_tree(ROOT / "external/CoOp/trainers", release_dir / "code/external/CoOp/trainers")
    copy_tree(ROOT / "external/CoOp/configs", release_dir / "code/external/CoOp/configs")
    copy_tree(ROOT / "external/CoOp/scripts", release_dir / "code/external/CoOp/scripts")
    for name in ["train.py", "requirements.txt", "README.md"]:
        copy_file(ROOT / "external/CoOp" / name, release_dir / "code/external/CoOp" / name)

    copy_tree(ROOT / "external/Dassl.pytorch/dassl", release_dir / "code/external/Dassl.pytorch/dassl")
    for name in ["setup.py", "requirements.txt", "README.md", "LICENSE"]:
        src = ROOT / "external/Dassl.pytorch" / name
        if src.exists():
            copy_file(src, release_dir / "code/external/Dassl.pytorch" / name)

    copy_tree(ROOT / "retinal_benchmark/src", release_dir / "code/retinal_benchmark/src")
    copy_tree(ROOT / "retinal_benchmark/scripts", release_dir / "code/retinal_benchmark/scripts")
    copy_tree(ROOT / "retinal_benchmark/configs", release_dir / "code/retinal_benchmark/configs")
    copy_tree(ROOT / "retinal_benchmark/benchmarks", release_dir / "code/retinal_benchmark/benchmarks")
    for name in ["pyproject.toml", "environment.yml"]:
        copy_file(ROOT / "retinal_benchmark" / name, release_dir / "code/retinal_benchmark" / name)

    # Documentation and schema examples.
    copy_tree(ROOT / "docs/release", release_dir / "docs")

    # Model artifacts and metrics. No raw images or private manifests are copied.
    model_dst = release_dir / "models/scvpt_fundussigns47_inference.pth"
    strip_checkpoint(FINAL_CHECKPOINT, model_dst, force=force)
    copy_file(LABEL_MAP, release_dir / "models/lesion_label_map.json")
    copy_file(FINAL_CONFIG, release_dir / "models/original_training_config.yaml")
    write_release_config(release_dir / "models/scvpt_release_config.yaml")
    for name in ["test_metrics_detailed.json", "test_per_class_metrics.csv", "log.txt"]:
        src = FINAL_RUN_DIR / name
        if src.exists():
            copy_file(src, release_dir / "models" / name)

    # Paper-supporting figures and analysis summaries.
    selected_assets = [
        ROOT / "reports/dataset_analysis/vision3/figures/fundussigns47_dataset_overview.png",
        ROOT / "reports/dataset_analysis/vision3/figures/fundussigns47_lesion_long_tail.png",
        ROOT / "reports/dataset_analysis/vision3/figures/fundussigns47_pareto_labelset_diversity.png",
        ROOT / "reports/method_figures/scvpt_pipeline_final.png",
        ROOT / "reports/mechanism/scvpt/scvpt_mechanism_overview.png",
        ROOT / "reports/mechanism/scvpt/scvpt_case_heatmaps.png",
        ROOT / "reports/mechanism/scvpt/mechanism_summary.json",
        ROOT / "reports/mechanism/scvpt/candidate_control_metrics.csv",
        ROOT / "reports/statistics/scvpt_bootstrap_ci/bootstrap_ci_summary.md",
        ROOT / "reports/statistics/scvpt_bootstrap_ci/bootstrap_ci_summary.csv",
        ROOT / "reports/statistics/scvpt_bootstrap_ci/bootstrap_ci_summary.json",
    ]
    for src in selected_assets:
        if src.exists():
            copy_file(src, release_dir / "analysis" / src.relative_to(ROOT / "reports"))

    files_for_manifest = [p for p in release_dir.rglob("*") if p.is_file()]
    write_manifest(release_dir, files_for_manifest)


def parse_args():
    parser = argparse.ArgumentParser(description="Build the FundusSigns-47 SCVPT release directory")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--force", action="store_true", help="Regenerate the stripped checkpoint if it exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_release(args.release_dir.resolve(), force=args.force)
    print(f"Release directory prepared at {args.release_dir.resolve()}")
