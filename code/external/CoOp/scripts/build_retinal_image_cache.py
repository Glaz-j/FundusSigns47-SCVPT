import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resolve_candidate(path_text: str) -> str:
    path = Path(path_text)
    if path.exists():
        return str(path)

    text = str(path)
    candidates = []
    if "/path/to/private_source/" in text:
        candidates.append(text.replace("/path/to/private_source/", "/path/to/project/"))
    if "/path/to/project/" in text:
        candidates.append(text.replace("/path/to/project/", "/path/to/private_source/"))

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return ""


def _resolve_impath(*paths: str) -> str:
    for impath in paths:
        if not impath or pd.isna(impath):
            continue
        resolved = _resolve_candidate(str(impath))
        if resolved:
            return resolved
    return str(paths[0]) if paths else ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--short-side", type=int, default=288)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--ext", type=str, default="jpg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_keep_ratio(img, short_side):
    width, height = img.size
    scale = short_side / min(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return img.resize((new_width, new_height), Image.BICUBIC)


def build_one(task):
    src, dst, short_side, quality, overwrite = task
    dst = Path(dst)
    if dst.exists() and not overwrite:
        return "skipped", str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = resize_keep_ratio(img, short_side)
        img.save(dst, quality=quality, subsampling=0, optimize=False)
    return "written", str(dst)


def main():
    args = parse_args()
    manifest = Path(os.path.abspath(os.path.expanduser(args.manifest)))
    output_dir = Path(os.path.abspath(os.path.expanduser(args.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)

    tasks = []
    for _, row in df.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if not sample_id or not split:
            continue
        src = _resolve_impath(row.get("linked_image_path"), row.get("source_image_path"))
        dst = output_dir / split / f"{sample_id}.{args.ext}"
        tasks.append((src, str(dst), args.short_side, args.quality, args.overwrite))

    written = 0
    skipped = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(build_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="building_cache"):
            try:
                status, _ = future.result()
                if status == "written":
                    written += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors.append(str(exc))

    print(
        {
            "manifest": str(manifest),
            "output_dir": str(output_dir),
            "samples": len(tasks),
            "written": written,
            "skipped": skipped,
            "errors": len(errors),
            "short_side": args.short_side,
            "quality": args.quality,
        }
    )
    if errors:
        print("sample_errors", errors[:10])


if __name__ == "__main__":
    main()
