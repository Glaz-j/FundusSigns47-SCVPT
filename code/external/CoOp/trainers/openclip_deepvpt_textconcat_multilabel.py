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


class OpenCLIPPromptedTextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx_text):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = getattr(clip_model, "attn_mask", None)
        self.n_ctx_text = n_ctx_text

    def _replace_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1 + self.n_ctx_text :, :]
        return torch.cat([prefix, prompt, suffix], dim=1)

    def forward(self, prompts, tokenized_prompts, deep_text_prompts):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts.to(cast_dtype)
        x = x + self.positional_embedding[: x.shape[1]].to(cast_dtype)

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[: x.shape[1], : x.shape[1]].to(x.device)

        for i, blk in enumerate(self.transformer.resblocks):
            if i > 0 and (i - 1) < len(deep_text_prompts):
                x = self._replace_prompt(x, deep_text_prompts[i - 1])
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


class TextConcatPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx_text = int(cfg.TRAINER.MAPLE.N_CTX)
        self.prompt_depth = int(cfg.TRAINER.MAPLE.PROMPT_DEPTH)
        assert self.prompt_depth >= 1

        text_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.transformer.get_cast_dtype()
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT

        if ctx_init and self.n_ctx_text <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = tokenizer(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).to(dtype)
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx_text, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(self.n_ctx_text, text_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=float(cfg.TRAINER.MAPLE.INIT_STD))
            prompt_prefix = " ".join(["X"] * self.n_ctx_text)

        self.ctx = nn.Parameter(ctx_vectors)
        self.deep_text_prompts = nn.ParameterList(
            [nn.Parameter(torch.empty(self.n_ctx_text, text_dim, dtype=dtype)) for _ in range(self.prompt_depth - 1)]
        )
        for prompt in self.deep_text_prompts:
            nn.init.normal_(prompt, std=float(cfg.TRAINER.MAPLE.INIT_STD))

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [(prompt_prefix + " " + name + ".").strip() for name in classnames]
        tokenized_prompts = tokenizer(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx_text :, :])
        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)
        deep_text = list(self.deep_text_prompts)
        return prompts, self.ctx, deep_text


class DeepVPTConcatImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.n_ctx_visual = int(cfg.TRAINER.VPT.N_CTX)
        self.n_ctx_text = int(cfg.TRAINER.MAPLE.N_CTX)
        self.prompt_depth = int(cfg.TRAINER.MAPLE.PROMPT_DEPTH)
        self.prompt_dropout = nn.Dropout(float(cfg.TRAINER.VPT.DROPOUT))

        width = self.visual.conv1.out_channels
        text_dim = clip_model.ln_final.weight.shape[0]
        depth = len(self.visual.transformer.resblocks)
        requested_deep_layers = int(getattr(cfg.TRAINER.VPT, "N_DEEP_LAYERS", -1))
        if requested_deep_layers <= 0:
            requested_deep_layers = depth - 1
        self.n_deep_layers = min(max(requested_deep_layers, 0), max(depth - 1, 0))
        self.coupled_deep_layers = min(max(self.prompt_depth - 1, 0), self.n_deep_layers)

        dtype = self.visual.conv1.weight.dtype
        self.shallow_prompt = nn.Parameter(torch.empty(self.n_ctx_visual, width, dtype=dtype))
        self.deep_prompts = nn.Parameter(torch.empty(self.n_deep_layers, self.n_ctx_visual, width, dtype=dtype))
        nn.init.normal_(self.shallow_prompt, std=float(cfg.TRAINER.VPT.INIT_STD))
        if self.deep_prompts.numel() > 0:
            nn.init.normal_(self.deep_prompts, std=float(cfg.TRAINER.VPT.INIT_STD))

        self.text_to_visual_shallow = nn.Linear(text_dim, width).to(dtype=dtype)
        self.text_to_visual_deep = nn.ModuleList(
            [nn.Linear(text_dim, width).to(dtype=dtype) for _ in range(self.coupled_deep_layers)]
        )

        for param in self.visual.parameters():
            param.requires_grad_(False)

    def _prepend_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        return torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)

    def _replace_prompt(self, x, prompt, current_prompt_len):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        image_tokens = x[:, 1 + current_prompt_len :, :]
        return torch.cat([x[:, :1, :], prompt, image_tokens], dim=1)

    def _concat_prompt(self, visual_prompt, text_prompt, proj_layer):
        text_prompt = proj_layer(text_prompt).to(visual_prompt.dtype)
        return torch.cat([visual_prompt, text_prompt], dim=0)

    def forward(self, image, text_ctx, deep_text_prompts):
        visual = self.visual
        x = image.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = _expand_token(visual.class_embedding, x.shape[0]).to(x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)

        shallow_prompt = self._concat_prompt(self.shallow_prompt, text_ctx, self.text_to_visual_shallow)
        current_prompt_len = shallow_prompt.shape[0]
        x = self._prepend_prompt(x, shallow_prompt)
        x = visual.ln_pre(x)

        blocks = visual.transformer.resblocks
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i < self.n_deep_layers:
                prompt = self.deep_prompts[i]
                if i < self.coupled_deep_layers:
                    prompt = self._concat_prompt(prompt, deep_text_prompts[i], self.text_to_visual_deep[i])
                x = self._replace_prompt(x, prompt, current_prompt_len)
                current_prompt_len = prompt.shape[0]

        pooled, _ = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled


class DeepVPTModImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.n_ctx_visual = int(cfg.TRAINER.VPT.N_CTX)
        self.prompt_depth = int(cfg.TRAINER.MAPLE.PROMPT_DEPTH)
        self.prompt_dropout = nn.Dropout(float(cfg.TRAINER.VPT.DROPOUT))

        width = self.visual.conv1.out_channels
        text_dim = clip_model.ln_final.weight.shape[0]
        depth = len(self.visual.transformer.resblocks)
        requested_deep_layers = int(getattr(cfg.TRAINER.VPT, "N_DEEP_LAYERS", -1))
        if requested_deep_layers <= 0:
            requested_deep_layers = depth - 1
        self.n_deep_layers = min(max(requested_deep_layers, 0), max(depth - 1, 0))
        self.coupled_deep_layers = min(max(self.prompt_depth - 1, 0), self.n_deep_layers)

        dtype = self.visual.conv1.weight.dtype
        self.shallow_prompt = nn.Parameter(torch.empty(self.n_ctx_visual, width, dtype=dtype))
        self.deep_prompts = nn.Parameter(torch.empty(self.n_deep_layers, self.n_ctx_visual, width, dtype=dtype))
        nn.init.normal_(self.shallow_prompt, std=float(cfg.TRAINER.VPT.INIT_STD))
        if self.deep_prompts.numel() > 0:
            nn.init.normal_(self.deep_prompts, std=float(cfg.TRAINER.VPT.INIT_STD))

        self.shallow_modulation = nn.Linear(text_dim, self.n_ctx_visual * width).to(dtype=dtype)
        self.shallow_gate = nn.Linear(text_dim, self.n_ctx_visual * width).to(dtype=dtype)
        self.deep_modulation = nn.ModuleList(
            [nn.Linear(text_dim, self.n_ctx_visual * width).to(dtype=dtype) for _ in range(self.coupled_deep_layers)]
        )
        self.deep_gate = nn.ModuleList(
            [nn.Linear(text_dim, self.n_ctx_visual * width).to(dtype=dtype) for _ in range(self.coupled_deep_layers)]
        )

        for param in self.visual.parameters():
            param.requires_grad_(False)

    def _prepend_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        return torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)

    def _replace_prompt(self, x, prompt, current_prompt_len):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        image_tokens = x[:, 1 + current_prompt_len :, :]
        return torch.cat([x[:, :1, :], prompt, image_tokens], dim=1)

    def _apply_modulation(self, visual_prompt, text_prompt, modulation_layer, gate_layer):
        summary = text_prompt.mean(dim=0, keepdim=False).to(visual_prompt.dtype)
        delta = modulation_layer(summary).view(self.n_ctx_visual, -1)
        gate = torch.sigmoid(gate_layer(summary)).view(self.n_ctx_visual, -1)
        return visual_prompt + gate * delta

    def forward(self, image, text_ctx, deep_text_prompts):
        visual = self.visual
        x = image.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = _expand_token(visual.class_embedding, x.shape[0]).to(x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)

        shallow_prompt = self._apply_modulation(
            self.shallow_prompt, text_ctx, self.shallow_modulation, self.shallow_gate
        )
        current_prompt_len = shallow_prompt.shape[0]
        x = self._prepend_prompt(x, shallow_prompt)
        x = visual.ln_pre(x)

        blocks = visual.transformer.resblocks
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i < self.n_deep_layers:
                prompt = self.deep_prompts[i]
                if i < self.coupled_deep_layers:
                    prompt = self._apply_modulation(
                        prompt, deep_text_prompts[i], self.deep_modulation[i], self.deep_gate[i]
                    )
                x = self._replace_prompt(x, prompt, current_prompt_len)
                current_prompt_len = prompt.shape[0]

        pooled, _ = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled


class DeepVPTCrossAttnImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.n_ctx_visual = int(cfg.TRAINER.VPT.N_CTX)
        self.prompt_depth = int(cfg.TRAINER.MAPLE.PROMPT_DEPTH)
        self.prompt_dropout = nn.Dropout(float(cfg.TRAINER.VPT.DROPOUT))

        width = self.visual.conv1.out_channels
        text_dim = clip_model.ln_final.weight.shape[0]
        depth = len(self.visual.transformer.resblocks)
        requested_deep_layers = int(getattr(cfg.TRAINER.VPT, "N_DEEP_LAYERS", -1))
        if requested_deep_layers <= 0:
            requested_deep_layers = depth - 1
        self.n_deep_layers = min(max(requested_deep_layers, 0), max(depth - 1, 0))
        self.total_visual_stages = 1 + self.n_deep_layers

        dtype = self.visual.conv1.weight.dtype
        self.shallow_prompt = nn.Parameter(torch.empty(self.n_ctx_visual, width, dtype=dtype))
        self.deep_prompts = nn.Parameter(torch.empty(self.n_deep_layers, self.n_ctx_visual, width, dtype=dtype))
        nn.init.normal_(self.shallow_prompt, std=float(cfg.TRAINER.VPT.INIT_STD))
        if self.deep_prompts.numel() > 0:
            nn.init.normal_(self.deep_prompts, std=float(cfg.TRAINER.VPT.INIT_STD))

        attn_dim = int(getattr(cfg.TRAINER.MAPLE, "CROSS_ATTN_DIM", 512))
        num_heads = int(getattr(cfg.TRAINER.MAPLE, "CROSS_ATTN_HEADS", 4))
        self.text_stage_embed = nn.Parameter(torch.empty(self.prompt_depth, text_dim, dtype=dtype))
        nn.init.normal_(self.text_stage_embed, std=0.02)

        self.text_bank_proj = nn.Linear(text_dim, attn_dim).to(dtype=dtype)
        self.query_proj = nn.Linear(width, attn_dim).to(dtype=dtype)
        self.cross_attn = nn.MultiheadAttention(attn_dim, num_heads, batch_first=True).to(dtype=dtype)
        self.attn_out = nn.Linear(attn_dim, width).to(dtype=dtype)
        self.gate_layer = nn.Linear(attn_dim, 1).to(dtype=dtype)

        for param in self.visual.parameters():
            param.requires_grad_(False)

    def _prepend_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        return torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)

    def _replace_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        image_tokens = x[:, 1 + prompt.shape[1] :, :]
        return torch.cat([x[:, :1, :], prompt, image_tokens], dim=1)

    def _build_text_bank(self, text_ctx, deep_text_prompts):
        prompt_stages = [text_ctx]
        prompt_stages.extend(list(deep_text_prompts)[: max(self.prompt_depth - 1, 0)])
        bank_tokens = []
        for idx, prompt in enumerate(prompt_stages):
            stage_embed = self.text_stage_embed[idx].unsqueeze(0).to(prompt.dtype)
            bank_tokens.append(prompt + stage_embed)
        text_bank = torch.cat(bank_tokens, dim=0)
        return self.text_bank_proj(text_bank)

    def _cross_attend_prompt(self, visual_prompt, text_bank):
        q = self.query_proj(visual_prompt).unsqueeze(0)
        kv = text_bank.unsqueeze(0)
        attn_output, _ = self.cross_attn(q, kv, kv, need_weights=False)
        attn_output = attn_output.squeeze(0)
        delta = self.attn_out(attn_output).to(visual_prompt.dtype)
        gate = torch.sigmoid(self.gate_layer(attn_output)).to(visual_prompt.dtype)
        return visual_prompt + gate * delta

    def forward(self, image, text_ctx, deep_text_prompts):
        visual = self.visual
        text_bank = self._build_text_bank(text_ctx, deep_text_prompts).to(dtype=visual.conv1.weight.dtype)

        x = image.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = _expand_token(visual.class_embedding, x.shape[0]).to(x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)

        shallow_prompt = self._cross_attend_prompt(self.shallow_prompt, text_bank)
        x = self._prepend_prompt(x, shallow_prompt)
        x = visual.ln_pre(x)

        blocks = visual.transformer.resblocks
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i < self.n_deep_layers:
                prompt = self._cross_attend_prompt(self.deep_prompts[i], text_bank)
                x = self._replace_prompt(x, prompt)

        pooled, _ = visual._pool(x)
        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled


class OpenCLIPDeepVPTTextConcatModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = TextConcatPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = OpenCLIPPromptedTextEncoder(clip_model, self.prompt_learner.n_ctx_text)
        self.image_encoder = DeepVPTConcatImageEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad_(bool(cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE))

    def forward(self, image):
        prompts, text_ctx, deep_text_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_text_prompts)
        image_features = self.image_encoder(image, text_ctx, deep_text_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ text_features.float().t()
        return logits


class OpenCLIPDeepVPTTextModModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = TextConcatPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = OpenCLIPPromptedTextEncoder(clip_model, self.prompt_learner.n_ctx_text)
        self.image_encoder = DeepVPTModImageEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad_(bool(cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE))

    def forward(self, image):
        prompts, text_ctx, deep_text_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_text_prompts)
        image_features = self.image_encoder(image, text_ctx, deep_text_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ text_features.float().t()
        return logits


class OpenCLIPDeepVPTTextCrossAttnModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = TextConcatPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = OpenCLIPPromptedTextEncoder(clip_model, self.prompt_learner.n_ctx_text)
        self.image_encoder = DeepVPTCrossAttnImageEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad_(bool(cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE))

    def forward(self, image):
        prompts, text_ctx, deep_text_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_text_prompts)
        image_features = self.image_encoder(image, text_ctx, deep_text_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ text_features.float().t()
        return logits


@TRAINER_REGISTRY.register()
class OpenCLIPDeepVPTTextConcatMultiLabel(TrainerX):
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
            "train", preprocess_train, cfg.DATALOADER.TRAIN_X.BATCH_SIZE, True
        )
        self.val_loader = self._make_loader(
            "val", preprocess_val, cfg.DATALOADER.TEST.BATCH_SIZE, False
        )
        self.test_loader = self._make_loader(
            "test", preprocess_val, cfg.DATALOADER.TEST.BATCH_SIZE, False
        )

        print("Building OpenCLIP DeepVPT + text-concat modulation model for multi-label classification")
        self.model = OpenCLIPDeepVPTTextConcatModel(cfg, self.classnames, clip_model, tokenizer)
        print(
            "Hybrid config: "
            f"visual_n_ctx={cfg.TRAINER.VPT.N_CTX}, "
            f"n_deep_layers={self.model.image_encoder.n_deep_layers}, "
            f"text_n_ctx={cfg.TRAINER.MAPLE.N_CTX}, "
            f"prompt_depth={cfg.TRAINER.MAPLE.PROMPT_DEPTH}, "
            f"train_logit_scale={cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE}, "
            f"train_proj={cfg.TRAINER.VPT.TRAIN_PROJ}, "
            f"train_norm={cfg.TRAINER.VPT.TRAIN_NORM}"
        )

        for name, param in self.model.named_parameters():
            trainable = False
            if name.startswith("prompt_learner."):
                trainable = True
            elif "image_encoder.shallow_prompt" in name or "image_encoder.deep_prompts" in name:
                trainable = True
            elif "image_encoder.text_to_visual_" in name:
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


@TRAINER_REGISTRY.register()
class OpenCLIPDeepVPTTextModMultiLabel(OpenCLIPDeepVPTTextConcatMultiLabel):
    def build_model(self):
        cfg = self.cfg
        print(f"Loading OpenCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess_train, preprocess_val, tokenizer = self._load_openclip()

        self.train_loader_x = self._make_loader(
            "train", preprocess_train, cfg.DATALOADER.TRAIN_X.BATCH_SIZE, True
        )
        self.val_loader = self._make_loader(
            "val", preprocess_val, cfg.DATALOADER.TEST.BATCH_SIZE, False
        )
        self.test_loader = self._make_loader(
            "test", preprocess_val, cfg.DATALOADER.TEST.BATCH_SIZE, False
        )

        print("Building OpenCLIP DeepVPT + text-modulation model for multi-label classification")
        self.model = OpenCLIPDeepVPTTextModModel(cfg, self.classnames, clip_model, tokenizer)
        print(
            "Hybrid config: "
            f"visual_n_ctx={cfg.TRAINER.VPT.N_CTX}, "
            f"n_deep_layers={self.model.image_encoder.n_deep_layers}, "
            f"text_n_ctx={cfg.TRAINER.MAPLE.N_CTX}, "
            f"prompt_depth={cfg.TRAINER.MAPLE.PROMPT_DEPTH}, "
            f"train_logit_scale={cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE}, "
            f"train_proj={cfg.TRAINER.VPT.TRAIN_PROJ}, "
            f"train_norm={cfg.TRAINER.VPT.TRAIN_NORM}"
        )

        for name, param in self.model.named_parameters():
            trainable = False
            if name.startswith("prompt_learner."):
                trainable = True
            elif "image_encoder.shallow_prompt" in name or "image_encoder.deep_prompts" in name:
                trainable = True
            elif "image_encoder.shallow_modulation" in name or "image_encoder.shallow_gate" in name:
                trainable = True
            elif "image_encoder.deep_modulation." in name or "image_encoder.deep_gate." in name:
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
        self.criterion = nn.BCEWithLogitsLoss()


@TRAINER_REGISTRY.register()
class OpenCLIPDeepVPTTextCrossAttnMultiLabel(OpenCLIPDeepVPTTextConcatMultiLabel):
    def build_model(self):
        cfg = self.cfg
        print(f"Loading OpenCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess_train, preprocess_val, tokenizer = self._load_openclip()

        self.train_loader_x = self._make_loader(
            "train", preprocess_train, cfg.DATALOADER.TRAIN_X.BATCH_SIZE, True
        )
        self.val_loader = self._make_loader(
            "val", preprocess_val, cfg.DATALOADER.TEST.BATCH_SIZE, False
        )
        self.test_loader = self._make_loader(
            "test", preprocess_val, cfg.DATALOADER.TEST.BATCH_SIZE, False
        )

        print("Building OpenCLIP DeepVPT + text cross-attention model for multi-label classification")
        self.model = OpenCLIPDeepVPTTextCrossAttnModel(cfg, self.classnames, clip_model, tokenizer)
        print(
            "Hybrid config: "
            f"visual_n_ctx={cfg.TRAINER.VPT.N_CTX}, "
            f"n_deep_layers={self.model.image_encoder.n_deep_layers}, "
            f"text_n_ctx={cfg.TRAINER.MAPLE.N_CTX}, "
            f"prompt_depth={cfg.TRAINER.MAPLE.PROMPT_DEPTH}, "
            f"cross_attn_dim={cfg.TRAINER.MAPLE.CROSS_ATTN_DIM}, "
            f"cross_attn_heads={cfg.TRAINER.MAPLE.CROSS_ATTN_HEADS}, "
            f"train_logit_scale={cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE}, "
            f"train_proj={cfg.TRAINER.VPT.TRAIN_PROJ}, "
            f"train_norm={cfg.TRAINER.VPT.TRAIN_NORM}"
        )

        for name, param in self.model.named_parameters():
            trainable = False
            if name.startswith("prompt_learner."):
                trainable = True
            elif "image_encoder.shallow_prompt" in name or "image_encoder.deep_prompts" in name:
                trainable = True
            elif "image_encoder.text_stage_embed" in name:
                trainable = True
            elif "image_encoder.text_bank_proj" in name:
                trainable = True
            elif "image_encoder.query_proj" in name:
                trainable = True
            elif "image_encoder.cross_attn" in name:
                trainable = True
            elif "image_encoder.attn_out" in name or "image_encoder.gate_layer" in name:
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
        self.criterion = nn.BCEWithLogitsLoss()
