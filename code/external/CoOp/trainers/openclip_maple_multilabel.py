import json
import os
from copy import deepcopy
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
from dassl.utils import load_pretrained_weights

from datasets.retinal_vision2 import _parse_multilabel, _resolve_impath
from trainers.openclip_coop_multilabel_cached import parse_openclip_backbone


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


class OpenCLIPMaPLeTextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = getattr(clip_model, "attn_mask", None)
        self.n_ctx = n_ctx

    def _replace_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1 + self.n_ctx:, :]
        return torch.cat([prefix, prompt, suffix], dim=1)

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts.to(cast_dtype)
        x = x + self.positional_embedding[:x.shape[1]].to(cast_dtype)

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:x.shape[1], :x.shape[1]].to(x.device)

        for i, blk in enumerate(self.transformer.resblocks):
            if i > 0 and (i - 1) < len(compound_prompts_deeper_text):
                x = self._replace_prompt(x, compound_prompts_deeper_text[i - 1])
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        tokenized_prompts = tokenized_prompts.to(x.device)
        pooled = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        return pooled


class OpenCLIPMaPLeImageEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx):
        super().__init__()
        self.visual = clip_model.visual
        self.n_ctx = n_ctx
        for param in self.visual.parameters():
            param.requires_grad_(False)

    def _append_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        return torch.cat([x, prompt], dim=1)

    def _replace_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prefix = x[:, :-self.n_ctx, :]
        return torch.cat([prefix, prompt], dim=1)

    def forward(self, image, shared_ctx, compound_deeper_prompts):
        visual = self.visual
        x = image.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = visual.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)

        x = self._append_prompt(x, shared_ctx)
        x = visual.ln_pre(x)

        for i, blk in enumerate(visual.transformer.resblocks):
            if i > 0 and (i - 1) < len(compound_deeper_prompts):
                x = self._replace_prompt(x, compound_deeper_prompts[i - 1])
            x = blk(x)

        pooled, _ = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled


class OpenCLIPMultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = int(cfg.TRAINER.MAPLE.N_CTX)
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        prompt_depth = int(cfg.TRAINER.MAPLE.PROMPT_DEPTH)
        assert prompt_depth >= 1, "MaPLe prompt depth must be >= 1"

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.conv1.out_channels
        dtype = clip_model.transformer.get_cast_dtype()

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenizer(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).to(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, text_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=float(cfg.TRAINER.MAPLE.INIT_STD))
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("MaPLe design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        print(f"MaPLe prompt depth: {prompt_depth}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.proj = nn.Linear(text_dim, visual_dim).to(dtype=dtype)

        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, text_dim, dtype=dtype)) for _ in range(prompt_depth - 1)]
        )
        for prompt in self.compound_prompts_text:
            nn.init.normal_(prompt, std=float(cfg.TRAINER.MAPLE.INIT_STD))

        base_proj = nn.Linear(text_dim, visual_dim).to(dtype=dtype)
        self.compound_prompt_projections = nn.ModuleList(
            [deepcopy(base_proj) for _ in range(prompt_depth - 1)]
        )

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [(prompt_prefix + " " + name + ".").strip() for name in classnames]
        tokenized_prompts = tokenizer(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        shared_ctx = self.proj(self.ctx)

        deep_text = list(self.compound_prompts_text)
        deep_vision = [
            proj_layer(text_prompt) for proj_layer, text_prompt in zip(self.compound_prompt_projections, deep_text)
        ]

        return prompts, shared_ctx, deep_text, deep_vision


class OpenCLIPMaPLeModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = OpenCLIPMultiModalPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = OpenCLIPMaPLeImageEncoder(clip_model, self.prompt_learner.n_ctx)
        self.text_encoder = OpenCLIPMaPLeTextEncoder(clip_model, self.prompt_learner.n_ctx)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad_(False)

    def forward(self, image):
        prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_text)
        image_features = self.image_encoder(image, shared_ctx, deep_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ text_features.float().t()
        return logits


@TRAINER_REGISTRY.register()
class OpenCLIPMaPLeMultiLabel(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

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

        print("Building OpenCLIP MaPLe model for multi-label classification")
        self.model = OpenCLIPMaPLeModel(cfg, self.classnames, clip_model, tokenizer)

        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
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
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MAPLE.PREC

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
        image = batch[0].to(self.device)
        label = batch[1].to(self.device)
        return image, label

    def parse_batch_test(self, batch):
        image = batch[0].to(self.device)
        label = batch[1].to(self.device)
        return image, label
