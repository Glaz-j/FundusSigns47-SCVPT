import json
import os
from pathlib import Path

import pandas as pd
import torch

from dassl.data.data_manager import DatasetWrapper
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json


def _parse_multilabel(label_text, num_classes):
    target = torch.zeros(num_classes, dtype=torch.float32)
    if label_text is None or pd.isna(label_text):
        return target
    text = str(label_text).strip()
    if not text:
        return target
    for token in text.split("|"):
        if not token:
            continue
        idx = int(token) - 1
        if 0 <= idx < num_classes:
            target[idx] = 1.0
    return target


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


class RetinalDatum(Datum):
    def __init__(self, impath="", label=0, domain=0, classname="", multilabel=None, sample_id=""):
        super().__init__(impath=impath, label=label, domain=domain, classname=classname)
        self._multilabel = multilabel
        self._sample_id = sample_id

    @property
    def multilabel(self):
        return self._multilabel

    @property
    def sample_id(self):
        return self._sample_id


class RetinalMultiLabelWrapper(DatasetWrapper):
    def __getitem__(self, idx):
        item = self.data_source[idx]
        output = super().__getitem__(idx)
        output["label"] = item.multilabel.clone()
        output["sample_id"] = item.sample_id
        return output


@DATASET_REGISTRY.register()
class RetinalVision2(DatasetBase):
    dataset_dir = "retinal_vision2"

    def __init__(self, cfg):
        manifest_path = Path(os.path.abspath(os.path.expanduser(cfg.DATASET.MANIFEST_PATH)))
        label_map_path = Path(os.path.abspath(os.path.expanduser(cfg.DATASET.LABEL_MAP_PATH)))

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

        df = pd.read_csv(manifest_path)
        with label_map_path.open("r", encoding="utf-8") as f:
            label_map = json.load(f)

        classnames = [label_map[str(i)] for i in range(1, len(label_map) + 1)]
        num_classes = len(classnames)

        def build_split(split_name):
            subset = df[df["split"] == split_name].reset_index(drop=True)
            max_samples = {
                "train": cfg.DATASET.MAX_TRAIN_SAMPLES,
                "val": cfg.DATASET.MAX_VAL_SAMPLES,
                "test": cfg.DATASET.MAX_TEST_SAMPLES,
            }[split_name]
            if max_samples is not None and int(max_samples) > 0:
                subset = subset.head(int(max_samples)).reset_index(drop=True)
            items = []
            for _, row in subset.iterrows():
                impath = _resolve_impath(
                    row.get("linked_image_path"),
                    row.get("source_image_path"),
                )
                target = _parse_multilabel(row.get("lesion_ids"), num_classes)
                positive = torch.nonzero(target, as_tuple=False).flatten().tolist()
                primary = positive[0] if positive else 0
                classname = classnames[primary] if classnames else "retinal lesion"
                item = RetinalDatum(
                    impath=impath,
                    label=primary,
                    domain=0,
                    classname=classname,
                    multilabel=target,
                    sample_id=str(row.get("sample_id", "")),
                )
                items.append(item)
            return items

        self._train_x = build_split("train")
        self._val = build_split("val")
        self._test = build_split("test")
        self._train_u = None
        self._num_classes = num_classes
        self._lab2cname = {i: name for i, name in enumerate(classnames)}
        self._classnames = classnames

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes


@DATASET_REGISTRY.register()
class RetinalVision3(RetinalVision2):
    dataset_dir = "retinal_vision3"
