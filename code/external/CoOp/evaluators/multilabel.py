import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dassl.evaluation import EVALUATOR_REGISTRY
from dassl.evaluation.evaluator import EvaluatorBase


@EVALUATOR_REGISTRY.register()
class MultiLabelClassification(EvaluatorBase):
    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self.lab2cname = lab2cname
        self.current_split = "test"
        self.reset()

    def reset(self):
        self.outputs = []
        self.targets = []

    def process(self, mo, gt):
        self.outputs.append(mo.detach().cpu())
        self.targets.append(gt.detach().cpu())

    @staticmethod
    def _round_float(value):
        return round(float(value), 6)

    def _save_details(self, details):
        output_dir = Path(self.cfg.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        split = getattr(self, "current_split", "test")

        json_path = output_dir / f"{split}_metrics_detailed.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

        csv_path = output_dir / f"{split}_per_class_metrics.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("class_index,class_name,positive_count,ap,auc\n")
            for item in details["per_class"]:
                f.write(
                    f"{item['class_index']},{json.dumps(item['class_name'], ensure_ascii=False)},"
                    f"{item['positive_count']},{item['ap']},{item['auc']}\n"
                )

    def evaluate(self):
        if not self.outputs:
            return OrderedDict(
                [
                    ("mAP", 0.0),
                    ("macro_f1", 0.0),
                    ("micro_f1", 0.0),
                    ("macro_auc", 0.0),
                    ("micro_precision", 0.0),
                    ("micro_recall", 0.0),
                    ("sample_f1", 0.0),
                ]
            )

        logits = torch.cat(self.outputs).numpy()
        targets = torch.cat(self.targets).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int32)

        results = OrderedDict()
        try:
            results["mAP"] = float(average_precision_score(targets, probs, average="macro")) * 100.0
        except Exception:
            results["mAP"] = 0.0

        try:
            results["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0)) * 100.0
            results["micro_f1"] = float(f1_score(targets, preds, average="micro", zero_division=0)) * 100.0
            results["micro_precision"] = float(
                precision_score(targets, preds, average="micro", zero_division=0)
            ) * 100.0
            results["micro_recall"] = float(
                recall_score(targets, preds, average="micro", zero_division=0)
            ) * 100.0
            results["sample_f1"] = float(f1_score(targets, preds, average="samples", zero_division=0)) * 100.0
        except Exception:
            results["macro_f1"] = 0.0
            results["micro_f1"] = 0.0
            results["micro_precision"] = 0.0
            results["micro_recall"] = 0.0
            results["sample_f1"] = 0.0

        aucs = []
        per_class = []
        for i in range(targets.shape[1]):
            class_name = self.lab2cname.get(i, str(i)) if self.lab2cname is not None else str(i)
            positive_count = int(targets[:, i].sum())
            ap_val = None
            auc_val = None
            try:
                ap_val = float(average_precision_score(targets[:, i], probs[:, i])) * 100.0
            except Exception:
                ap_val = None
            if len(np.unique(targets[:, i])) < 2:
                auc_val = None
            else:
                try:
                    auc_metric = float(roc_auc_score(targets[:, i], probs[:, i]))
                    aucs.append(auc_metric)
                    auc_val = auc_metric * 100.0
                except Exception:
                    auc_val = None
            per_class.append(
                {
                    "class_index": i,
                    "class_name": class_name,
                    "positive_count": positive_count,
                    "ap": None if ap_val is None else self._round_float(ap_val),
                    "auc": None if auc_val is None else self._round_float(auc_val),
                }
            )
        results["macro_auc"] = (float(np.mean(aucs)) * 100.0) if aucs else 0.0

        details = {
            "split": getattr(self, "current_split", "test"),
            "num_samples": int(targets.shape[0]),
            "num_classes": int(targets.shape[1]),
            "threshold": 0.5,
            "summary": {k: self._round_float(v) for k, v in results.items()},
            "per_class": per_class,
        }
        self._save_details(details)

        print(
            "=> result\n"
            f"* mAP: {results['mAP']:.2f}\n"
            f"* macro_f1: {results['macro_f1']:.2f}\n"
            f"* micro_f1: {results['micro_f1']:.2f}\n"
            f"* macro_auc: {results['macro_auc']:.2f}\n"
            f"* micro_precision: {results['micro_precision']:.2f}\n"
            f"* micro_recall: {results['micro_recall']:.2f}\n"
            f"* sample_f1: {results['sample_f1']:.2f}"
        )
        return results
