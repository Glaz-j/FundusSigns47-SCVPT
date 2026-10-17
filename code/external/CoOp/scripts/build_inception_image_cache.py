#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from datasets.retinal_vision2 import _resolve_impath


def resolve_image_path(row, image_cache_dir: Path, image_cache_ext: str) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    split = str(row.get("split", "")).strip()
    if sample_id and split:
        cached = image_cache_dir / split / f"{sample_id}.{image_cache_ext}"
        if cached.exists():
            return str(cached)
    return _resolve_impath(row.get("linked_image_path"), row.get("source_image_path"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-cache-ext", default="jpg")
    args = parser.parse_args()

    manifest = Path(os.path.expanduser(args.manifest)).resolve()
    image_cache_dir = Path(os.path.expanduser(args.image_cache_dir)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    tfm = transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    total = 0
    built = 0
    skipped = 0

    for row in tqdm(df.to_dict("records"), desc="Building Inception cache"):
        total += 1
        sample_id = str(row.get("sample_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not sample_id or not split:
            skipped += 1
            continue

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / f"{sample_id}.pt"
        if out_path.exists():
            continue

        impath = resolve_image_path(row, image_cache_dir, args.image_cache_ext)
        if not impath or not Path(impath).exists():
            skipped += 1
            continue

        with Image.open(impath) as img:
            tensor = tfm(img.convert("RGB"))

        tmp_path = out_path.with_suffix(".pt.tmp")
        torch.save(tensor, tmp_path)
        os.replace(tmp_path, out_path)
        built += 1

    print(
        f"Done. total={total} built={built} skipped={skipped} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
