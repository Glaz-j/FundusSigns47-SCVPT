"""Dataset classes for retinal benchmark experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _parse_label_vector(label_text: str | float | None, num_classes: int) -> torch.Tensor:
    target = torch.zeros(num_classes, dtype=torch.float32)
    if label_text is None or (isinstance(label_text, float) and pd.isna(label_text)):
        return target
    text = str(label_text).strip()
    if not text:
        return target
    for item in text.split("|"):
        if not item:
            continue
        idx = int(item) - 1
        if 0 <= idx < num_classes:
            target[idx] = 1.0
    return target


class VisionManifestDataset(Dataset):
    """Dataset backed by a frozen manifest csv."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        image_column: str = "linked_image_path",
        label_column: str = "lesion_ids",
        num_classes: int = 47,
        transform=None,
        image_cache_dir: str | Path | None = None,
        image_cache_ext: str = "jpg",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(self.manifest_path)
        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_column = image_column
        self.label_column = label_column
        self.num_classes = num_classes
        self.transform = transform
        self.image_cache_dir = (
            Path(image_cache_dir).expanduser().resolve() if image_cache_dir else None
        )
        self.image_cache_ext = image_cache_ext

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row[self.image_column])
        if self.image_cache_dir is not None:
            sample_id = str(row.get("sample_id", "")).strip()
            split = str(row.get("split", "")).strip()
            if sample_id and split:
                cached_path = self.image_cache_dir / split / f"{sample_id}.{self.image_cache_ext}"
                if cached_path.exists():
                    image_path = cached_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = _parse_label_vector(row[self.label_column], self.num_classes)
        meta = {
            "sample_id": row["sample_id"],
            "batch_id": row["batch_id"],
            "source_image_path": row["source_image_path"],
            "patient_id": row["patient_id"],
        }
        return image, target, meta


def load_label_map(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
