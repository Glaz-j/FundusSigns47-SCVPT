import json
import os
from pathlib import Path

import open_clip
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_checkpoint, load_pretrained_weights

from datasets.retinal_vision2 import _parse_multilabel, _resolve_impath
from trainers.openclip_coop_multilabel_cached import parse_openclip_backbone
from trainers.openclip_vpt_multilabel import VisualPromptedOpenCLIPImageEncoder


class RetinalImageDataset(Dataset):
    def __init__(self, rows, num_classes, transform, image_cache_dir="", image_cache_ext="jpg"):
        self.rows = rows
        self.num_classes = num_classes
        self.transform = transform
        self.image_cache_dir = Path(image_cache_dir).expanduser().resolve() if image_cache_dir else None
        self.image_cache_ext = image_cache_ext

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        impath = self._resolve_image_path(row)
        with Image.open(impath) as img:
            image = self.transform(img.convert("RGB"))
        label = _parse_multilabel(row.get("lesion_ids"), self.num_classes)
        return image, label, str(row.get("sample_id", ""))

    def _resolve_image_path(self, row):
        if self.image_cache_dir is not None:
            sample_id = str(row.get("sample_id", "")).strip()
            split = str(row.get("split", "")).strip()
            if sample_id and split:
                cached = self.image_cache_dir / split / f"{sample_id}.{self.image_cache_ext}"
                if cached.exists():
                    return str(cached)
        return _resolve_impath(row.get("linked_image_path"), row.get("source_image_path"))


class OpenCLIPVPTOnlyModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.image_encoder = VisualPromptedOpenCLIPImageEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale

        template = "a fundus photograph of {}."
        prompts = [template.format(name.replace("_", " ")) for name in classnames]
        tokenized_prompts = tokenizer(prompts)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", text_features.float())
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.logit_scale.requires_grad_(False)

    def forward(self, image):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class OpenCLIPVPTOnlyMultiLabel(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        manifest_path = Path(os.path.abspath(os.path.expanduser(self.cfg.DATASET.MANIFEST_PATH)))
        label_map_path = Path(os.path.abspath(os.path.expanduser(self.cfg.DATASET.LABEL_MAP_PATH)))
        self.df = pd.read_csv(manifest_path)
        with label_map_path.open("r", encoding="utf-8") as f:
            label_map = json.load(f)
        self.classnames = [label_map[str(i)] for i in range(1, len(label_map) + 1)]
        self.num_classes = len(self.classnames)
        self.lab2cname = {i: name for i, name in enumerate(self.classnames)}
        self.num_source_domains = 0
        self.dm = None
        self.train_loader_x = None
        self.train_loader_u = None
        self.val_loader = None
        self.test_loader = None

    def _load_openclip(self):
        model_name, pretrained_name = parse_openclip_backbone(self.cfg.MODEL.BACKBONE.NAME)
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
            device="cpu",
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess_train, preprocess_val, tokenizer

    def _make_loader(self, split, transform, batch_size, shuffle):
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
            transform,
            image_cache_dir=self.cfg.DATASET.IMAGE_CACHE_DIR,
            image_cache_ext=self.cfg.DATASET.IMAGE_CACHE_EXT,
        )
        num_workers = int(self.cfg.DATALOADER.NUM_WORKERS)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def build_model(self):
        cfg = self.cfg
        print(f"Loading OpenCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess_train, preprocess_val, tokenizer = self._load_openclip()

        self.train_loader_x = self._make_loader(
            "train",
            preprocess_train,
            cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            True,
        )
        self.val_loader = self._make_loader(
            "val",
            preprocess_val,
            cfg.DATALOADER.TEST.BATCH_SIZE,
            False,
        )
        self.test_loader = self._make_loader(
            "test",
            preprocess_val,
            cfg.DATALOADER.TEST.BATCH_SIZE,
            False,
        )

        print("Building OpenCLIP + visual prompt only model for multi-label classification")
        self.model = OpenCLIPVPTOnlyModel(cfg, self.classnames, clip_model, tokenizer)

        for name, param in self.model.named_parameters():
            if "image_encoder.visual_prompt" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"# trainable params: {trainable_count:,}")
        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.VPT.PREC == "amp" else None
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.VPT.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = self.criterion(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = self.criterion(output, label)
            self.model_backward_and_update(loss)

        with torch.no_grad():
            probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()
            match = (preds == label).float().mean().item() * 100.0

        loss_summary = {"loss": loss.item(), "label_match": match}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        image, label, _ = batch
        return image.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True).float()

    def parse_batch_test(self, batch):
        image, label, _ = batch
        return image.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True).float()

    def model_inference(self, input):
        return self.model(input)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = os.path.join(directory, name, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            loaded_epoch = checkpoint["epoch"]
            print(f'Loading weights to {name} from "{model_path}" (epoch = {loaded_epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
