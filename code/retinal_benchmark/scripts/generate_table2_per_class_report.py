#!/usr/bin/env python
"""Generate per-class comparison report for Table 2 (vision3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/path/to/project")
RB_ROOT = ROOT / "retinal_benchmark"
SRC = RB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retinal_benchmark.config import load_yaml
from retinal_benchmark.models import create_model
from retinal_benchmark.training import build_dataloaders


OUT_DIR = RB_ROOT / "reports" / "vision3_table2_per_class"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP_PATH = ROOT / "retinal_benchmark" / "data_versions" / "vision3" / "lesion_label_map.json"


TABLE2_METHODS: List[Dict[str, str]] = [
    {
        "group": "baseline",
        "method": "ResNet50",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__resnet50__42__20260412_081237"),
    },
    {
        "group": "baseline",
        "method": "DenseNet121",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__densenet121__42__20260412_081236"),
    },
    {
        "group": "baseline",
        "method": "EfficientNet-B4",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__efficientnet_b4__42__20260412_090351"),
    },
    {
        "group": "baseline",
        "method": "ConvNeXt-Tiny",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__convnext_tiny__42__20260412_055036"),
    },
    {
        "group": "baseline",
        "method": "InceptionV3",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__inception_v3__42__20260412_055020"),
    },
    {
        "group": "baseline",
        "method": "DeiT-Small",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__deit_small__42__20260412_090351"),
    },
    {
        "group": "baseline",
        "method": "ViT-B/16",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__vit_b16__42__20260412_090351"),
    },
    {
        "group": "baseline",
        "method": "Swin-Tiny",
        "run_dir": str(RB_ROOT / "experiments" / "logs" / "vision3__swin_tiny__42__20260412_055036"),
    },
    {
        "group": "ocl",
        "method": "OCL-dVPT-Tuned",
        "detailed_json": str(
            ROOT
            / "external"
            / "CoOp"
            / "output"
            / "retinal_vision3"
            / "OpenCLIPVPTDeepOnlyMultiLabel"
            / "full_100ep_imgcache288_ctx16_fulldepth_tuned_cuda1"
            / "test_metrics_detailed.json"
        ),
    },
    {
        "group": "ocl",
        "method": "OCL-dVPT-Tuned+S2C+TR",
        "detailed_json": str(
            ROOT
            / "external"
            / "CoOp"
            / "output"
            / "retinal_vision3"
            / "OpenCLIPVPTDeepResidualMultiLabelCached"
            / "coop_taskres_cuda3"
            / "test_metrics_detailed.json"
        ),
    },
    {
        "group": "ocl",
        "method": "OCL-dVPT-T-Conc",
        "detailed_json": str(
            ROOT
            / "external"
            / "CoOp"
            / "output"
            / "retinal_vision3"
            / "OpenCLIPDeepVPTTextConcatMultiLabel"
            / "full_100ep_cuda0"
            / "test_metrics_detailed.json"
        ),
    },
    {
        "group": "ocl",
        "method": "OCL-dVPT-T-Mod",
        "detailed_json": str(
            ROOT
            / "external"
            / "CoOp"
            / "output"
            / "retinal_vision3"
            / "OpenCLIPDeepVPTTextModMultiLabel"
            / "full_100ep_cuda2"
            / "test_metrics_detailed.json"
        ),
    },
    {
        "group": "ocl",
        "method": "OCL-dVPT-T-XAttn",
        "detailed_json": str(
            ROOT
            / "external"
            / "CoOp"
            / "output"
            / "retinal_vision3"
            / "OpenCLIPDeepVPTTextCrossAttnMultiLabel"
            / "full_100ep_cuda3"
            / "test_metrics_detailed.json"
        ),
    },
]


def load_label_map() -> Dict[int, str]:
    raw = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return {int(k) - 1: v for k, v in raw.items()}


def compute_detailed_from_run(run_dir: Path) -> Dict:
    run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    data_cfg = load_yaml(run_meta["data_config"])
    model_cfg = load_yaml(run_meta["model_config"])
    train_cfg = load_yaml(run_meta["train_config"])

    datasets, loaders = build_dataloaders(data_cfg, train_cfg)
    dataset = datasets["test"]
    label_map_path = Path(data_cfg.get("label_map_path", str(LABEL_MAP_PATH)))
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    class_names = [label_map.get(str(i + 1), str(i + 1)) for i in range(dataset.num_classes)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_cfg["name"],
        model_cfg["num_classes"],
        pretrained=False,
    )
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logits_list = []
    targets_list = []
    with torch.no_grad():
        for images, targets, _ in loaders["test"]:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())

    logits = torch.cat(logits_list).numpy()
    targets = torch.cat(targets_list).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    per_class = []
    for class_idx, class_name in enumerate(class_names):
        y_true = targets[:, class_idx]
        y_prob = probs[:, class_idx]
        positive_count = int(y_true.sum())
        try:
            ap = float(average_precision_score(y_true, y_prob)) * 100.0
        except Exception:
            ap = 0.0
        try:
            auc = float(roc_auc_score(y_true, y_prob)) * 100.0 if len(np.unique(y_true)) >= 2 else None
        except Exception:
            auc = None
        per_class.append(
            {
                "class_index": class_idx,
                "class_name": class_name,
                "positive_count": positive_count,
                "ap": ap,
                "auc": auc,
            }
        )

    return {"per_class": per_class}


def load_ocl_detailed(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_long_table() -> pd.DataFrame:
    rows = []
    for spec in TABLE2_METHODS:
        if spec["group"] == "baseline":
            detailed = compute_detailed_from_run(Path(spec["run_dir"]))
        else:
            detailed = load_ocl_detailed(Path(spec["detailed_json"]))
        for item in detailed["per_class"]:
            rows.append(
                {
                    "method": spec["method"],
                    "group": spec["group"],
                    "class_index": item["class_index"],
                    "class_name": item["class_name"],
                    "positive_count": item["positive_count"],
                    "ap": item["ap"],
                    "auc": item["auc"],
                }
            )
    return pd.DataFrame(rows)


def build_winner_table(df: pd.DataFrame) -> pd.DataFrame:
    winners = []
    for class_idx, group in df.groupby("class_index"):
        ap_group = group.sort_values(["ap", "method"], ascending=[False, True]).reset_index(drop=True)
        auc_valid = group.dropna(subset=["auc"]).sort_values(["auc", "method"], ascending=[False, True]).reset_index(drop=True)
        winners.append(
            {
                "class_index": int(class_idx),
                "class_name": ap_group.iloc[0]["class_name"],
                "positive_count": int(ap_group.iloc[0]["positive_count"]),
                "best_ap_method": ap_group.iloc[0]["method"],
                "best_ap": float(ap_group.iloc[0]["ap"]),
                "second_ap_method": ap_group.iloc[1]["method"],
                "second_ap": float(ap_group.iloc[1]["ap"]),
                "best_auc_method": auc_valid.iloc[0]["method"] if not auc_valid.empty else "",
                "best_auc": float(auc_valid.iloc[0]["auc"]) if not auc_valid.empty else np.nan,
                "second_auc_method": auc_valid.iloc[1]["method"] if len(auc_valid) > 1 else "",
                "second_auc": float(auc_valid.iloc[1]["auc"]) if len(auc_valid) > 1 else np.nan,
            }
        )
    return pd.DataFrame(winners)


def render_markdown(df: pd.DataFrame, winners: pd.DataFrame) -> str:
    ap_win_counts = winners["best_ap_method"].value_counts()
    auc_win_counts = winners["best_auc_method"].value_counts()

    lines = []
    lines.append("# Vision3 Table 2 Per-Class Comparison")
    lines.append("")
    lines.append("This report compares all methods currently listed in Table 2 on the 47 lesion classes using per-class AP and AUC.")
    lines.append("")
    lines.append("## AP Winner Counts")
    lines.append("")
    for method, count in ap_win_counts.items():
        lines.append(f"- `{method}`: {int(count)} classes")
    lines.append("")
    lines.append("## AUC Winner Counts")
    lines.append("")
    for method, count in auc_win_counts.items():
        lines.append(f"- `{method}`: {int(count)} classes")
    lines.append("")

    head_classes = winners.sort_values("best_ap", ascending=False).head(10)
    lines.append("## Top 10 Classes by Best AP")
    lines.append("")
    lines.append("| Class | Positives | Best AP Method | Best AP | Second AP Method | Second AP |")
    lines.append("|---|---:|---|---:|---|---:|")
    for _, row in head_classes.iterrows():
        lines.append(
            f"| {row['class_name']} | {int(row['positive_count'])} | {row['best_ap_method']} | {row['best_ap']:.2f} | {row['second_ap_method']} | {row['second_ap']:.2f} |"
        )
    lines.append("")

    tail_classes = winners.sort_values("positive_count").head(10)
    lines.append("## Tail Classes (Fewest Positives)")
    lines.append("")
    lines.append("| Class | Positives | Best AP Method | Best AP | Best AUC Method | Best AUC |")
    lines.append("|---|---:|---|---:|---|---:|")
    for _, row in tail_classes.iterrows():
        best_auc = "" if pd.isna(row["best_auc"]) else f"{row['best_auc']:.2f}"
        lines.append(
            f"| {row['class_name']} | {int(row['positive_count'])} | {row['best_ap_method']} | {row['best_ap']:.2f} | {row['best_auc_method']} | {best_auc} |"
        )
    lines.append("")

    focus_methods = [
        "InceptionV3",
        "OCL-dVPT-Tuned",
        "OCL-dVPT-Tuned+S2C+TR",
        "OCL-dVPT-T-Conc",
        "OCL-dVPT-T-Mod",
        "OCL-dVPT-T-XAttn",
    ]
    pivot = (
        df[df["method"].isin(focus_methods)]
        .pivot_table(index=["class_index", "class_name", "positive_count"], columns="method", values="ap")
        .reset_index()
    )
    lines.append("## Focused AP Comparison (InceptionV3 vs Main OCL Routes)")
    lines.append("")
    lines.append("See `table2_per_class_ap_wide.csv` for the full matrix.")
    lines.append("")
    sample = pivot.head(12)
    lines.append("| Class | Positives | InceptionV3 | OCL-dVPT-Tuned | OCL-dVPT-Tuned+S2C+TR | OCL-dVPT-T-Conc | OCL-dVPT-T-Mod | OCL-dVPT-T-XAttn |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in sample.iterrows():
        lines.append(
            f"| {row['class_name']} | {int(row['positive_count'])} | "
            f"{row.get('InceptionV3', np.nan):.2f} | "
            f"{row.get('OCL-dVPT-Tuned', np.nan):.2f} | "
            f"{row.get('OCL-dVPT-Tuned+S2C+TR', np.nan):.2f} | "
            f"{row.get('OCL-dVPT-T-Conc', np.nan):.2f} | "
            f"{row.get('OCL-dVPT-T-Mod', np.nan):.2f} | "
            f"{row.get('OCL-dVPT-T-XAttn', np.nan):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    df = build_long_table()
    winners = build_winner_table(df)

    long_csv = OUT_DIR / "table2_per_class_ap_auc_long.csv"
    winner_csv = OUT_DIR / "table2_per_class_winners.csv"
    ap_wide_csv = OUT_DIR / "table2_per_class_ap_wide.csv"
    auc_wide_csv = OUT_DIR / "table2_per_class_auc_wide.csv"
    report_md = OUT_DIR / "table2_per_class_summary.md"

    df.sort_values(["class_index", "method"]).to_csv(long_csv, index=False, encoding="utf-8-sig")
    winners.sort_values("class_index").to_csv(winner_csv, index=False, encoding="utf-8-sig")

    ap_wide = (
        df.pivot_table(index=["class_index", "class_name", "positive_count"], columns="method", values="ap")
        .reset_index()
        .sort_values("class_index")
    )
    auc_wide = (
        df.pivot_table(index=["class_index", "class_name", "positive_count"], columns="method", values="auc")
        .reset_index()
        .sort_values("class_index")
    )
    ap_wide.to_csv(ap_wide_csv, index=False, encoding="utf-8-sig")
    auc_wide.to_csv(auc_wide_csv, index=False, encoding="utf-8-sig")

    report_md.write_text(render_markdown(df, winners), encoding="utf-8")

    print(f"Wrote: {long_csv}")
    print(f"Wrote: {winner_csv}")
    print(f"Wrote: {ap_wide_csv}")
    print(f"Wrote: {auc_wide_csv}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
