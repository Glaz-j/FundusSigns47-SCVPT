import os
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dassl.engine import TRAINER_REGISTRY

from trainers.openclip_coop_multilabel_cached import (
    CachedOpenCLIPPromptModel,
    OpenCLIPCoOpMultiLabelCached,
    parse_openclip_backbone,
)
from trainers.openclip_vpt_deep_only_multilabel import (
    OpenCLIPVPTDeepOnlyModel,
    RetinalImageDataset,
)


@TRAINER_REGISTRY.register()
class OpenCLIPVPTDeepCoOpMultiLabelCached(OpenCLIPCoOpMultiLabelCached):
    """Stage-2 CoOp on top of a frozen tuned Deep VPT image encoder.

    We first rebuild a deterministic feature cache using the tuned Deep VPT
    checkpoint, then freeze the entire image side and train only CoOp text
    prompts on the cached image features.
    """

    def _default_cache_path(self):
        manifest_path = Path(os.path.abspath(os.path.expanduser(self.cfg.DATASET.MANIFEST_PATH)))
        ckpt_path = str(getattr(self.cfg.TRAINER.VPT, "FROZEN_MODEL_PATH", "")).strip()
        ckpt_tag = "vptdeep_frozen"
        if ckpt_path:
            ckpt_tag = Path(ckpt_path).resolve().parent.parent.parent.name.lower()
        backbone = self.cfg.MODEL.BACKBONE.NAME.replace("/", "-").replace(" ", "_").replace(":", "_").lower()
        return manifest_path.parent / f"feature_cache_{backbone}_{ckpt_tag}_center_crop.pt"

    def _load_openclip(self):
        model_name, pretrained_name = parse_openclip_backbone(self.cfg.MODEL.BACKBONE.NAME)
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
            device="cpu",
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess_val, tokenizer

    def _build_or_load_feature_cache(self, clip_model, preprocess):
        cache_path = (
            Path(self.cfg.DATASET.FEATURE_CACHE_PATH)
            if self.cfg.DATASET.FEATURE_CACHE_PATH
            else self._default_cache_path()
        )
        cache_path = Path(os.path.abspath(os.path.expanduser(str(cache_path))))
        if cache_path.exists() and not self.cfg.DATASET.FEATURE_CACHE_REBUILD:
            print(f"Loading frozen Deep VPT feature cache from {cache_path}")
            return torch.load(cache_path, map_location="cpu")

        frozen_model_path = str(getattr(self.cfg.TRAINER.VPT, "FROZEN_MODEL_PATH", "")).strip()
        if not frozen_model_path:
            raise ValueError("TRAINER.VPT.FROZEN_MODEL_PATH must point to a tuned Deep VPT checkpoint")

        frozen_model_path = os.path.abspath(os.path.expanduser(frozen_model_path))
        if not os.path.exists(frozen_model_path):
            raise FileNotFoundError(f"Frozen Deep VPT checkpoint not found: {frozen_model_path}")

        print(f"Building frozen Deep VPT feature cache at {cache_path}")
        print(f"Using frozen Deep VPT checkpoint: {frozen_model_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer = open_clip.get_tokenizer(parse_openclip_backbone(self.cfg.MODEL.BACKBONE.NAME)[0])
        frozen_model = OpenCLIPVPTDeepOnlyModel(self.cfg, self.classnames, clip_model, tokenizer)
        checkpoint = torch.load(frozen_model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        frozen_model.load_state_dict(state_dict, strict=False)
        frozen_model = frozen_model.to(self.device)
        frozen_model.eval()
        for param in frozen_model.parameters():
            param.requires_grad_(False)

        feature_dim = int(frozen_model.text_features.shape[-1])
        result = {
            "meta": {
                "backbone": self.cfg.MODEL.BACKBONE.NAME,
                "frozen_model_path": frozen_model_path,
                "num_classes": self.num_classes,
                "input_resolution": int(self.cfg.INPUT.SIZE[0]),
                "feature_dim": feature_dim,
            },
            "splits": {},
        }

        with torch.no_grad():
            for split in ["train", "val", "test"]:
                subset = self.df[self.df["split"] == split].reset_index(drop=True)
                max_samples = {
                    "train": self.cfg.DATASET.MAX_TRAIN_SAMPLES,
                    "val": self.cfg.DATASET.MAX_VAL_SAMPLES,
                    "test": self.cfg.DATASET.MAX_TEST_SAMPLES,
                }[split]
                if max_samples is not None and int(max_samples) > 0:
                    subset = subset.head(int(max_samples)).reset_index(drop=True)

                dataset = RetinalImageDataset(
                    subset.to_dict("records"),
                    self.num_classes,
                    preprocess,
                    image_cache_dir=self.cfg.DATASET.IMAGE_CACHE_DIR,
                    image_cache_ext=self.cfg.DATASET.IMAGE_CACHE_EXT,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=int(self.cfg.DATASET.FEATURE_CACHE_BATCH_SIZE),
                    shuffle=False,
                    num_workers=int(self.cfg.DATASET.FEATURE_CACHE_NUM_WORKERS),
                    pin_memory=True,
                    persistent_workers=int(self.cfg.DATASET.FEATURE_CACHE_NUM_WORKERS) > 0,
                )

                features = []
                labels = []
                sample_ids = []
                for images, target, ids in loader:
                    images = images.to(self.device, non_blocking=True)
                    feats = frozen_model.image_encoder(images)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    features.append(feats.float().cpu())
                    labels.append(target.float().cpu())
                    sample_ids.extend(list(ids))

                result["splits"][split] = {
                    "features": torch.cat(features, dim=0) if features else torch.empty(0, feature_dim),
                    "labels": torch.cat(labels, dim=0) if labels else torch.empty(0, self.num_classes),
                    "sample_ids": sample_ids,
                }

        torch.save(result, cache_path)
        return result

    def build_model(self):
        cfg = self.cfg
        print(f"Loading OpenCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess, tokenizer = self._load_openclip()

        cache = self._build_or_load_feature_cache(clip_model, preprocess)
        frozen_model_path = os.path.abspath(os.path.expanduser(str(cfg.TRAINER.VPT.FROZEN_MODEL_PATH)))
        checkpoint = torch.load(frozen_model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        if "logit_scale" in state_dict:
            with torch.no_grad():
                clip_model.logit_scale.copy_(state_dict["logit_scale"])

        clip_model = clip_model.cpu()
        self.train_loader_x = DataLoader(
            TensorDataset(cache["splits"]["train"]["features"], cache["splits"]["train"]["labels"]),
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(cache["splits"]["val"]["features"], cache["splits"]["val"]["labels"]),
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(cache["splits"]["test"]["features"], cache["splits"]["test"]["labels"]),
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        print("Building Stage-2 cached CoOp model on frozen tuned Deep VPT features")
        self.model = CachedOpenCLIPPromptModel(cfg, self.classnames, clip_model, tokenizer)
        for name, param in self.model.named_parameters():
            param.requires_grad_("prompt_learner" in name)

        self.model.to(self.device)
        from dassl.optim import build_lr_scheduler, build_optimizer
        from dassl.utils import count_num_param
        from torch.cuda.amp import GradScaler

        print(f"# params: {count_num_param(self.model.prompt_learner):,}")
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
        self.criterion = nn.BCEWithLogitsLoss()
