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
from dassl.utils import load_checkpoint, load_pretrained_weights

from datasets.retinal_vision2 import _parse_multilabel, _resolve_impath
from open_clip.transformer import _expand_token
from trainers.openclip_coop_multilabel_cached import parse_openclip_backbone


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)

        pos_loss = targets * torch.log(pos_probs.clamp(min=self.eps))
        neg_loss = (1.0 - targets) * torch.log(neg_probs.clamp(min=self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = pos_probs * targets + neg_probs * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            focal_weight = torch.pow(1.0 - pt, gamma)
            pos_loss = pos_loss * focal_weight
            neg_loss = neg_loss * focal_weight

        loss = -(pos_loss + neg_loss)
        return loss.mean()


class RetinalImageDataset(Dataset):
    def __init__(
        self,
        rows,
        num_classes,
        transform,
        image_cache_dir="",
        image_cache_ext="jpg",
        inception_cache_dir="",
    ):
        self.rows = rows
        self.num_classes = num_classes
        self.transform = transform
        self.image_cache_dir = Path(image_cache_dir).expanduser().resolve() if image_cache_dir else None
        self.image_cache_ext = image_cache_ext
        self.inception_cache_dir = (
            Path(inception_cache_dir).expanduser().resolve() if inception_cache_dir else None
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        impath = self._resolve_image_path(row)
        with Image.open(impath) as img:
            image = self.transform(img.convert("RGB"))
        label = _parse_multilabel(row.get("lesion_ids"), self.num_classes)
        sample_id = str(row.get("sample_id", ""))
        if self.inception_cache_dir is None:
            return image, label, sample_id

        split = str(row.get("split", "")).strip()
        if not sample_id or not split:
            raise ValueError("sample_id and split are required when inception_cache_dir is set")
        inc_path = self.inception_cache_dir / split / f"{sample_id}.pt"
        inc_image = torch.load(inc_path, map_location="cpu")
        return image, label, sample_id, inc_image

    def _resolve_image_path(self, row):
        if self.image_cache_dir is not None:
            sample_id = str(row.get("sample_id", "")).strip()
            split = str(row.get("split", "")).strip()
            if sample_id and split:
                cached = self.image_cache_dir / split / f"{sample_id}.{self.image_cache_ext}"
                if cached.exists():
                    return str(cached)
        return _resolve_impath(row.get("linked_image_path"), row.get("source_image_path"))


class DeepVisualPromptedOpenCLIPImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.n_ctx = int(cfg.TRAINER.VPT.N_CTX)
        self.prompt_dropout = nn.Dropout(float(cfg.TRAINER.VPT.DROPOUT))
        self.uses_timm_visual = hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "blocks")
        if self.uses_timm_visual:
            trunk = self.visual.trunk
            width = int(getattr(trunk, "embed_dim", trunk.num_features))
            depth = len(trunk.blocks)
            dtype = trunk.patch_embed.proj.weight.dtype
            self.num_prefix_tokens = int(getattr(trunk, "num_prefix_tokens", 1) or 1)
        else:
            width = self.visual.conv1.out_channels
            depth = len(self.visual.transformer.resblocks)
            dtype = self.visual.conv1.weight.dtype
            self.num_prefix_tokens = 1
        requested_deep_layers = int(getattr(cfg.TRAINER.VPT, "N_DEEP_LAYERS", -1))
        if requested_deep_layers <= 0:
            requested_deep_layers = depth - 1
        self.n_deep_layers = min(max(requested_deep_layers, 0), max(depth - 1, 0))

        self.shallow_prompt = nn.Parameter(torch.empty(self.n_ctx, width, dtype=dtype))
        self.deep_prompts = nn.Parameter(torch.empty(self.n_deep_layers, self.n_ctx, width, dtype=dtype))
        nn.init.normal_(self.shallow_prompt, std=float(cfg.TRAINER.VPT.INIT_STD))
        if self.deep_prompts.numel() > 0:
            nn.init.normal_(self.deep_prompts, std=float(cfg.TRAINER.VPT.INIT_STD))

        for param in self.visual.parameters():
            param.requires_grad_(False)

    def _prepend_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        return torch.cat([x[:, :self.num_prefix_tokens, :], prompt, x[:, self.num_prefix_tokens:, :]], dim=1)

    def _replace_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        image_tokens = x[:, self.num_prefix_tokens + self.n_ctx:, :]
        return torch.cat([x[:, :self.num_prefix_tokens, :], prompt, image_tokens], dim=1)

    def _forward_openclip_visual(self, image):
        visual = self.visual
        x = image.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = _expand_token(visual.class_embedding, x.shape[0]).to(x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)

        x = self._prepend_prompt(x, self.shallow_prompt)
        x = visual.ln_pre(x)

        blocks = visual.transformer.resblocks
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i < self.n_deep_layers and self.deep_prompts.numel() > 0:
                x = self._replace_prompt(x, self.deep_prompts[i])

        pooled, _ = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled

    def _forward_timm_visual(self, image):
        visual = self.visual
        trunk = visual.trunk
        x = image.to(dtype=trunk.patch_embed.proj.weight.dtype)
        x = trunk.patch_embed(x)
        x = trunk._pos_embed(x)
        x = trunk.patch_drop(x)

        x = self._prepend_prompt(x, self.shallow_prompt)
        x = trunk.norm_pre(x)

        blocks = trunk.blocks
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i < self.n_deep_layers and self.deep_prompts.numel() > 0:
                x = self._replace_prompt(x, self.deep_prompts[i])

        x = trunk.norm(x)
        pooled = trunk.forward_head(x)
        pooled = visual.head(pooled)
        return pooled

    def forward(self, image):
        if self.uses_timm_visual:
            return self._forward_timm_visual(image)
        return self._forward_openclip_visual(image)


class OpenCLIPVPTDeepOnlyModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = DeepVisualPromptedOpenCLIPImageEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale

        template = "a fundus photograph of {}."
        prompts = [template.format(name.replace("_", " ")) for name in classnames]
        tokenized_prompts = tokenizer(prompts)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", text_features.float())
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.logit_scale.requires_grad_(bool(cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE))

    def forward(self, image):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ self.text_features.t()
        return logits


def build_multilabel_criterion(cfg):
    loss_name = str(getattr(cfg.TRAINER.VPT, "LOSS_NAME", "bce")).lower()
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name == "asl":
        return AsymmetricLoss(
            gamma_neg=cfg.TRAINER.VPT.ASL_GAMMA_NEG,
            gamma_pos=cfg.TRAINER.VPT.ASL_GAMMA_POS,
            clip=cfg.TRAINER.VPT.ASL_CLIP,
            eps=cfg.TRAINER.VPT.ASL_EPS,
        )
    raise ValueError(f"Unsupported TRAINER.VPT.LOSS_NAME: {cfg.TRAINER.VPT.LOSS_NAME}")


@TRAINER_REGISTRY.register()
class OpenCLIPVPTDeepOnlyMultiLabel(TrainerX):
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
        if str(pretrained_name).strip().lower() in {"", "none", "null", "false"}:
            pretrained_name = None
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

        print("Building OpenCLIP + deep visual prompt only model for multi-label classification")
        self.model = OpenCLIPVPTDeepOnlyModel(cfg, self.classnames, clip_model, tokenizer)
        print(
            "Deep VPT config: "
            f"n_ctx={cfg.TRAINER.VPT.N_CTX}, "
            f"n_deep_layers={self.model.image_encoder.n_deep_layers}, "
            f"train_logit_scale={cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE}, "
            f"train_proj={cfg.TRAINER.VPT.TRAIN_PROJ}, "
            f"train_norm={cfg.TRAINER.VPT.TRAIN_NORM}, "
            f"loss={cfg.TRAINER.VPT.LOSS_NAME}"
        )

        for name, param in self.model.named_parameters():
            trainable = False
            if "image_encoder.shallow_prompt" in name or "image_encoder.deep_prompts" in name:
                trainable = True
            elif bool(cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE) and name == "logit_scale":
                trainable = True
            elif bool(cfg.TRAINER.VPT.TRAIN_PROJ) and name == "image_encoder.visual.proj":
                trainable = True
            elif bool(cfg.TRAINER.VPT.TRAIN_NORM) and (
                "image_encoder.visual.ln_pre." in name or "image_encoder.visual.ln_post." in name
            ):
                trainable = True
            param.requires_grad_(trainable)

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
        self.criterion = build_multilabel_criterion(cfg)

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
