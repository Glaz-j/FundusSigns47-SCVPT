#!/usr/bin/env python3
"""Build a versioned manifest and symlinked dataset folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retinal_benchmark.constants import DATASET_SELECTIONS
from retinal_benchmark.manifest import (
    build_union_vision2_records,
    build_union_vision3_records,
    build_dataset_records,
    safe_symlink,
    write_label_maps,
    write_manifest,
)


IMAGE_ROOT = Path("/path/to/private_source/eyedataset/厦大AI")
VISION1_MANIFEST = PROJECT_ROOT / "data_versions" / "vision1" / "manifest.csv"
VISION2_MANIFEST = PROJECT_ROOT / "data_versions" / "vision2" / "manifest.csv"
INCOMING_Q1_ROOT = Path("/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注")
RECOVERED_RETDET_ROOT = Path("/path/to/project/网脱")
RECOVERED_RETDET_REPORT = PROJECT_ROOT / "reports" / "vision2_missing_images" / "203257_蔡司-视网膜脱离补充数据_20260330105644_1_1" / "annotator_2标注_missing_images.csv"


def write_batch_labels(label_root: Path, records: list[dict]) -> None:
    label_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["batch_id"]), []).append(record)
    fieldnames = [
        "sample_id",
        "batch_id",
        "patient_id",
        "lesion_ids",
        "lesion_texts",
        "diagnosis_ids",
        "diagnosis_texts",
        "quality_label",
        "quality_text",
    ]
    for batch_id, batch_records in grouped.items():
        batch_dir = label_root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        with (batch_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in batch_records:
                writer.writerow({k: record.get(k, "") for k in fieldnames})


def write_splits(split_root: Path, records: list[dict]) -> dict[str, int]:
    split_root.mkdir(parents=True, exist_ok=True)
    split_names = ["train", "val", "test"]
    fieldnames = [
        "sample_id",
        "dataset_version",
        "batch_id",
        "split",
        "patient_id",
        "linked_image_path",
        "source_image_path",
        "lesion_ids",
        "lesion_texts",
        "diagnosis_ids",
        "diagnosis_texts",
    ]
    split_summary = {}
    for split in split_names:
        path = split_root / f"{split}.csv"
        split_rows = [record for record in records if record["split"] == split]
        split_summary[split] = len(split_rows)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in split_rows:
                writer.writerow({k: record.get(k, "") for k in fieldnames})
    (split_root / "summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return split_summary


def write_summary(summary_path: Path, records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts.setdefault(record["batch_id"], 0)
        counts[record["batch_id"]] += 1
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_id", "matched_annotation_rows"])
        for batch_id in sorted(counts):
            writer.writerow([batch_id, counts[batch_id]])
    return counts


def write_metadata(version_dir: Path, version: str, counts: dict[str, int], split_summary: dict[str, int]) -> None:
    cfg = DATASET_SELECTIONS[version]
    metadata = {
        "version": version,
        "description": cfg["description"],
        "status": "draft",
        "raw_data": {
            "image_root": str(IMAGE_ROOT),
            "annotation_roots": cfg["annotation_roots"],
        },
        "processed_data": {
            "manifest_csv": str(version_dir / "manifest.csv"),
            "linked_image_root": str(version_dir / "images"),
            "label_root": str(version_dir / "labels"),
            "split_root": str(version_dir / "splits" / "patient_level"),
        },
        "matched_batches": [
            {
                "batch_id": batch_id,
                "matched_annotation_rows": counts[batch_id],
                "selected_annotation_file": Path(cfg["selected_annotations"][batch_id]["annotation_path"]).name,
                "selected_annotation_path": cfg["selected_annotations"][batch_id]["annotation_path"],
            }
            for batch_id in sorted(counts)
        ],
        "totals": {
            "matched_annotation_rows": sum(counts.values()),
        },
        "split_summary": {
            "strategy": "patient_level_hash",
            **split_summary,
        },
        "notes": cfg.get("notes", []),
    }
    (version_dir / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def build_dataset(version: str) -> None:
    if version not in DATASET_SELECTIONS:
        raise KeyError(f"Unsupported dataset version: {version}")

    version_dir = PROJECT_ROOT / "data_versions" / version
    link_root = version_dir / "images"
    label_root = version_dir / "labels"
    split_root = version_dir / "splits" / "patient_level"
    manifest_path = version_dir / "manifest.csv"

    if version == "vision2":
        records = build_union_vision2_records(
            image_root=IMAGE_ROOT,
            vision1_manifest_path=VISION1_MANIFEST,
            incoming_root=INCOMING_Q1_ROOT,
        )
    elif version == "vision3":
        records = build_union_vision3_records(
            vision2_manifest_path=VISION2_MANIFEST,
            recovered_root=RECOVERED_RETDET_ROOT,
            missing_report_path=RECOVERED_RETDET_REPORT,
        )
    else:
        records = build_dataset_records(image_root=IMAGE_ROOT, dataset_version=version)
    for record in records:
        source_path = Path(record["source_image_path"])
        suffix = source_path.suffix.lower()
        link_path = link_root / record["batch_id"] / f"{record['sample_id']}{suffix}"
        safe_symlink(source_path, link_path)
        record["linked_image_path"] = str(link_path)

    write_manifest(records, manifest_path)
    write_label_maps(version_dir)
    write_batch_labels(label_root, records)
    split_summary = write_splits(split_root, records)
    counts = write_summary(version_dir / "summary.csv", records)
    write_metadata(version_dir, version, counts, split_summary)

    print(f"Wrote dataset version: {version}")
    print(f"Manifest: {manifest_path}")
    print(f"Batch counts: {counts}")
    print(f"Split summary: {split_summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=sorted(DATASET_SELECTIONS))
    args = parser.parse_args()
    build_dataset(args.version)


if __name__ == "__main__":
    main()
