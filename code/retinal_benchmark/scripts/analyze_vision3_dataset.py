#!/usr/bin/env python
"""Generate descriptive statistics and publication-style figures for FundusSigns-47."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


ROOT = Path("/path/to/project")
DATA_DIR = ROOT / "retinal_benchmark/data_versions/vision3"
OUT_DIR = ROOT / "reports/dataset_analysis/vision3"
FIG_DIR = OUT_DIR / "figures"
PUBLIC_DATASET_NAME = "FundusSigns-47"
FILE_PREFIX = "fundussigns47"


LESION_GROUPS = {
    "Media/Vitreous": [1, 2, 3, 4, 5],
    "Macula": list(range(6, 16)),
    "Optic Disc": list(range(16, 22)),
    "High Myopia": [22, 23, 35],
    "Retinal Detachment/Break": [24, 28, 42],
    "Hemorrhage/Exudation": [25, 26, 27, 37, 38, 39],
    "Laser/Scarring/Pigment": [31, 32, 33, 34, 41],
    "Vascular": [40, 44, 45, 46, 47],
    "Other Retina": [29, 30, 36, 43],
}


EN_SHORT_NAMES = {
    1: "Media opacity",
    2: "Vitreous hemorrhage",
    3: "Asteroid hyalosis",
    4: "PVD",
    5: "Other vitreous",
    6: "Drusen",
    7: "Subretinal fibrosis",
    8: "Myopic macular atrophy",
    9: "Geographic atrophy",
    10: "Epiretinal membrane",
    11: "Macular hemorrhage",
    12: "Submacular hemorrhage",
    13: "Serous macular RD",
    14: "Macular hole",
    15: "Other maculopathy",
    16: "Disc collateral",
    17: "Blurred disc margin",
    18: "Congenital disc anomaly",
    19: "Large cup",
    20: "Optic atrophy",
    21: "Disc neovascularization",
    22: "Myopic disc crescent",
    23: "Myopic disc ring",
    24: "Retinal detachment",
    25: "Quadrant hemorrhage",
    26: "Subretinal hemorrhage",
    27: "Preretinal hemorrhage",
    28: "Retinal break",
    29: "Retinal fibrotic membrane",
    30: "Chorioretinal mass",
    31: "Laser spot",
    32: "Panretinal photocoag.",
    33: "Sectoral laser scars",
    34: "Bone-spicule pigment",
    35: "Tessellated fundus",
    36: "Sunset-glow fundus",
    37: "Retinal hemorrhage",
    38: "Hard exudates",
    39: "Cotton-wool spots",
    40: "Retinal neovascularization",
    41: "Old pigmented lesion",
    42: "Peripheral degeneration",
    43: "Other retinopathy",
    44: "Retinal artery occlusion",
    45: "Vascular white line",
    46: "Arteriolosclerosis",
    47: "Vascular sheathing/vasculitis",
}


def parse_ids(value) -> list[int]:
    if pd.isna(value):
        return []
    return [int(x) for x in str(value).split("|") if str(x).strip()]


def safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def build_label_matrix(df: pd.DataFrame, label_ids: list[int]) -> np.ndarray:
    label_to_col = {label_id: idx for idx, label_id in enumerate(label_ids)}
    mat = np.zeros((len(df), len(label_ids)), dtype=np.uint8)
    for row_idx, ids in enumerate(df["lesion_list"]):
        for label_id in ids:
            if label_id in label_to_col:
                mat[row_idx, label_to_col[label_id]] = 1
    return mat


def collect_image_geometry(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        image_path = Path(str(row["linked_image_path"]))
        width = height = None
        if image_path.exists():
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception:
                pass
        rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "width": width,
                "height": height,
                "pixels": None if width is None or height is None else width * height,
                "aspect_ratio": None if width is None or height is None else width / height,
            }
        )
    return pd.DataFrame(rows)


def plot_overview(df: pd.DataFrame, label_counts: pd.DataFrame, geometry: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    fig.patch.set_facecolor("white")

    cardinality = df["num_lesions"].fillna(0).astype(int)
    sns.countplot(x=cardinality, ax=axes[0], color="#A3BE8C")
    axes[0].set_title("Label cardinality")
    axes[0].set_xlabel("Positive signs per image")
    axes[0].set_ylabel("Images")

    group_counts = compute_group_counts(df)
    sns.barplot(x=group_counts.values, y=group_counts.index, ax=axes[1], color="#D08770")
    axes[1].set_title("Clinical sign groups")
    axes[1].set_xlabel("Images with at least one sign")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        sns.despine(ax=ax)
    plt.tight_layout()
    savefig(FIG_DIR / f"{FILE_PREFIX}_dataset_overview")


def compute_group_counts(df: pd.DataFrame) -> pd.Series:
    group_sets = {name: set(ids) for name, ids in LESION_GROUPS.items()}
    counts = {}
    for name, ids in group_sets.items():
        counts[name] = int(df["lesion_list"].apply(lambda xs: bool(ids.intersection(xs))).sum())
    return pd.Series(counts).sort_values(ascending=False)


def plot_long_tail(label_counts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    data = label_counts.sort_values("total", ascending=True).copy()
    colors = np.where(data["total"] < 20, "#D62828", np.where(data["total"] < 100, "#F77F00", "#2A9D8F"))
    plt.figure(figsize=(9, 12))
    ax = plt.gca()
    ax.hlines(y=data["plot_name"], xmin=0, xmax=data["total"], color=colors, alpha=0.85, linewidth=2.4)
    ax.scatter(data["total"], data["plot_name"], color=colors, s=28, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Positive images (log scale)")
    ax.set_ylabel("")
    ax.set_title("Long-Tailed Lesion Distribution", fontsize=15, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    savefig(FIG_DIR / f"{FILE_PREFIX}_lesion_long_tail")


def plot_split_heatmap(label_counts: pd.DataFrame) -> None:
    data = label_counts.sort_values("total", ascending=False).copy()
    matrix = data[["train", "val", "test"]].to_numpy(dtype=float)
    row_sum = matrix.sum(axis=1, keepdims=True)
    row_pct = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum != 0) * 100.0
    global_pct = np.array([data["train"].sum(), data["val"].sum(), data["test"].sum()], dtype=float)
    global_pct = global_pct / global_pct.sum() * 100.0
    deviation = row_pct - global_pct
    vmax = max(8.0, float(np.nanpercentile(np.abs(deviation), 95)))
    plt.figure(figsize=(7.2, 12.5))
    ax = sns.heatmap(
        deviation,
        yticklabels=data["plot_name"],
        xticklabels=["Train", "Val", "Test"],
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=False,
        cbar_kws={"label": "Deviation from global split (%)"},
        linewidths=0.25,
        linecolor="#F0F0F0",
    )
    ax.set_title("Per-label Split Balance", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6.5)
    savefig(FIG_DIR / f"{FILE_PREFIX}_split_balance_heatmap")


def plot_cooccurrence(label_counts: pd.DataFrame, mat: np.ndarray) -> None:
    top_ids = label_counts.head(20)["label_id"].astype(int).tolist()
    all_ids = label_counts["label_id"].astype(int).tolist()
    cols = [all_ids.index(label_id) for label_id in top_ids]
    sub = mat[:, cols].astype(bool)
    n = len(top_ids)
    jaccard = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            inter = np.logical_and(sub[:, i], sub[:, j]).sum()
            union = np.logical_or(sub[:, i], sub[:, j]).sum()
            jaccard[i, j] = safe_rate(inter, union)
    labels = [f"L{x} {EN_SHORT_NAMES[x]}" for x in top_ids]
    mask = np.eye(n, dtype=bool)
    off_diag = jaccard[~mask]
    vmax = max(0.08, float(np.nanpercentile(off_diag[off_diag > 0], 98)) if np.any(off_diag > 0) else 0.08)
    plt.figure(figsize=(10.5, 9))
    ax = sns.heatmap(
        jaccard,
        mask=mask,
        cmap="RdBu_r",
        norm=PowerNorm(gamma=0.55, vmin=0, vmax=vmax),
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Jaccard co-occurrence"},
        linewidths=0.35,
        linecolor="#F5F5F5",
    )
    ax.set_title("Top-20 Sign Co-occurrence Structure", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    savefig(FIG_DIR / f"{FILE_PREFIX}_top20_cooccurrence_jaccard")


def plot_quality_source(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#fbfaf7")

    match_counts = df["match_type"].fillna("unknown").value_counts()
    sns.barplot(x=match_counts.index, y=match_counts.values, ax=axes[0], palette=["#2A9D8F", "#E9C46A", "#E76F51"])
    axes[0].set_title("Image matching")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Images")

    ann_counts = df["annotation_record_count"].fillna(0).astype(int).value_counts().sort_index()
    sns.barplot(x=ann_counts.index.astype(str), y=ann_counts.values, ax=axes[1], color="#264653")
    axes[1].set_title("Annotation records")
    axes[1].set_xlabel("Records per image")
    axes[1].set_ylabel("Images")

    conflict_counts = df["label_conflict"].fillna(0).astype(int).map({0: "No conflict", 1: "Conflict"}).value_counts()
    sns.barplot(x=conflict_counts.index, y=conflict_counts.values, ax=axes[2], palette=["#2A9D8F", "#D62828"])
    axes[2].set_title("Merged-label conflicts")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Images")

    plt.suptitle("Data Provenance and Annotation Reliability", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    savefig(FIG_DIR / f"{FILE_PREFIX}_source_annotation_quality")


def plot_labelset_pareto(df: pd.DataFrame, label_counts: pd.DataFrame) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    combo_counter = Counter(tuple(xs) for xs in df["lesion_list"])
    combo_sizes = pd.Series(combo_counter.values()).sort_values(ascending=False).reset_index(drop=True)
    cumulative = combo_sizes.cumsum() / combo_sizes.sum() * 100.0
    sorted_counts = label_counts["total"].sort_values(ascending=False).reset_index(drop=True)
    label_cumulative = sorted_counts.cumsum() / sorted_counts.sum() * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(np.arange(1, len(label_cumulative) + 1), label_cumulative, color="#5E81AC", linewidth=2.2)
    axes[0].axhline(80, color="#BF616A", linestyle="--", linewidth=1)
    axes[0].set_title("Label-count concentration")
    axes[0].set_xlabel("Top-k lesion labels")
    axes[0].set_ylabel("Cumulative positives (%)")
    axes[0].set_ylim(0, 105)

    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, color="#A3BE8C", linewidth=2.2)
    axes[1].axhline(80, color="#BF616A", linestyle="--", linewidth=1)
    axes[1].set_title("Labelset diversity")
    axes[1].set_xlabel("Top-k unique labelsets")
    axes[1].set_ylabel("Cumulative images (%)")
    axes[1].set_ylim(0, 105)

    for ax in axes:
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.35)
        sns.despine(ax=ax)
    plt.tight_layout()
    savefig(FIG_DIR / f"{FILE_PREFIX}_pareto_labelset_diversity")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "manifest.csv")
    lesion_map_raw = json.loads((DATA_DIR / "lesion_label_map.json").read_text(encoding="utf-8"))
    lesion_map = {int(k): v for k, v in lesion_map_raw.items()}
    label_ids = sorted(lesion_map)

    df["lesion_list"] = df["lesion_ids"].apply(parse_ids)
    df["diagnosis_list"] = df["diagnosis_ids"].apply(parse_ids)
    mat = build_label_matrix(df, label_ids)

    split_counts = df["split"].value_counts().reindex(["train", "val", "test"]).astype(int)
    label_rows = []
    for col, label_id in enumerate(label_ids):
        total = int(mat[:, col].sum())
        row = {
            "label_id": label_id,
            "label_name_zh": lesion_map[label_id],
            "label_name_en": EN_SHORT_NAMES.get(label_id, f"Lesion {label_id}"),
            "plot_name": f"L{label_id}: {EN_SHORT_NAMES.get(label_id, lesion_map[label_id])}",
            "total": total,
        }
        for split in ["train", "val", "test"]:
            mask = df["split"].eq(split).to_numpy()
            row[split] = int(mat[mask, col].sum())
            row[f"{split}_prevalence_pct"] = round(safe_rate(row[split], int(mask.sum())) * 100, 4)
        row["prevalence_pct"] = round(safe_rate(total, len(df)) * 100, 4)
        label_rows.append(row)
    label_counts = pd.DataFrame(label_rows).sort_values("total", ascending=False).reset_index(drop=True)
    label_counts.to_csv(OUT_DIR / f"{FILE_PREFIX}_lesion_distribution.csv", index=False)

    group_counts = compute_group_counts(df)
    group_counts.to_csv(OUT_DIR / f"{FILE_PREFIX}_lesion_group_counts.csv", header=["image_count"])

    geometry = collect_image_geometry(df)
    geometry.to_csv(OUT_DIR / f"{FILE_PREFIX}_image_geometry.csv", index=False)

    labelset_counter = Counter("|".join(map(str, xs)) if xs else "none" for xs in df["lesion_list"])
    pd.DataFrame(
        [{"labelset": key, "count": value} for key, value in labelset_counter.most_common()]
    ).to_csv(OUT_DIR / f"{FILE_PREFIX}_labelset_distribution.csv", index=False)

    positives_per_image = df["lesion_list"].apply(len)
    samples_with_labels = int((positives_per_image > 0).sum())
    summary = {
        "num_images": int(len(df)),
        "num_train": int(split_counts["train"]),
        "num_val": int(split_counts["val"]),
        "num_test": int(split_counts["test"]),
        "num_lesion_labels": int(len(label_ids)),
        "num_diagnosis_labels": int(len(json.loads((DATA_DIR / "diagnosis_label_map.json").read_text(encoding="utf-8")))),
        "num_unique_patient_like_keys": int(df["patient_id"].dropna().nunique()),
        "num_unique_studies": int(df["study_instance_uid"].dropna().nunique()),
        "num_source_batches": int(df["batch_id"].dropna().nunique()),
        "samples_with_lesion_labels": samples_with_labels,
        "samples_without_lesion_labels": int(len(df) - samples_with_labels),
        "total_positive_lesion_assignments": int(mat.sum()),
        "label_cardinality_mean": round(float(positives_per_image.mean()), 4),
        "label_cardinality_median": round(float(positives_per_image.median()), 4),
        "label_density": round(float(mat.sum() / (len(df) * len(label_ids))), 6),
        "unique_labelsets": int(len(labelset_counter)),
        "most_common_label_count": int(label_counts["total"].max()),
        "rarest_positive_label_count": int(label_counts.loc[label_counts["total"] > 0, "total"].min()),
        "labels_with_lt_20_positives": int((label_counts["total"] < 20).sum()),
        "labels_with_lt_100_positives": int((label_counts["total"] < 100).sum()),
        "match_type_counts": df["match_type"].fillna("unknown").value_counts().to_dict(),
        "annotation_record_count": {str(k): int(v) for k, v in df["annotation_record_count"].value_counts().sort_index().items()},
        "label_conflict_counts": {str(k): int(v) for k, v in df["label_conflict"].value_counts().sort_index().items()},
        "image_width_median": None if geometry["width"].dropna().empty else float(geometry["width"].median()),
        "image_height_median": None if geometry["height"].dropna().empty else float(geometry["height"].median()),
    }
    (OUT_DIR / f"{FILE_PREFIX}_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(OUT_DIR / f"{FILE_PREFIX}_dataset_summary.csv", index=False)

    plot_overview(df, label_counts, geometry)
    plot_long_tail(label_counts)
    plot_split_heatmap(label_counts)
    plot_cooccurrence(label_counts, mat)
    plot_quality_source(df)
    plot_labelset_pareto(df, label_counts)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved analysis to: {OUT_DIR}")


if __name__ == "__main__":
    main()
