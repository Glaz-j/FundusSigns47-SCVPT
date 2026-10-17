import json
import math
import os
from pathlib import Path

import open_clip
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.rank = int(rank)
        self.scaling = float(alpha) / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features
        dtype = base.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B) * self.scaling
        return result + lora.to(result.dtype)


class LoRAMultiheadAttention(nn.Module):
    def __init__(self, base: nn.MultiheadAttention, rank: int, alpha: float, apply_in: bool, apply_out: bool):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.rank = int(rank)
        self.scaling = float(alpha) / max(1, self.rank)
        self.apply_in = bool(apply_in)
        self.apply_out = bool(apply_out)

        dtype = base.in_proj_weight.dtype
        embed_dim = base.embed_dim

        if self.apply_in:
            self.in_lora_A = nn.Parameter(torch.empty(self.rank, embed_dim, dtype=dtype))
            self.in_lora_B = nn.Parameter(torch.zeros(embed_dim * 3, self.rank, dtype=dtype))
            nn.init.kaiming_uniform_(self.in_lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("in_lora_A", None)
            self.register_parameter("in_lora_B", None)

        if self.apply_out:
            self.out_lora_A = nn.Parameter(torch.empty(self.rank, embed_dim, dtype=dtype))
            self.out_lora_B = nn.Parameter(torch.zeros(embed_dim, self.rank, dtype=dtype))
            nn.init.kaiming_uniform_(self.out_lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("out_lora_A", None)
            self.register_parameter("out_lora_B", None)

    def _delta_in_proj(self):
        if self.in_lora_A is None:
            return None
        return (self.in_lora_B @ self.in_lora_A) * self.scaling

    def _delta_out_proj(self):
        if self.out_lora_A is None:
            return None
        return (self.out_lora_B @ self.out_lora_A) * self.scaling

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        attn = self.base
        in_proj_weight = attn.in_proj_weight
        delta_in = self._delta_in_proj()
        if delta_in is not None:
            in_proj_weight = in_proj_weight + delta_in.to(in_proj_weight.dtype)

        out_proj_weight = attn.out_proj.weight
        delta_out = self._delta_out_proj()
        if delta_out is not None:
            out_proj_weight = out_proj_weight + delta_out.to(out_proj_weight.dtype)

        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=attn.embed_dim,
            num_heads=attn.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=attn.in_proj_bias,
            bias_k=attn.bias_k,
            bias_v=attn.bias_v,
            add_zero_attn=attn.add_zero_attn,
            dropout_p=attn.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=attn.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=getattr(attn, "average_attn_weights", True),
            is_causal=False,
        )


class LoRAOpenCLIPImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        for param in self.visual.parameters():
            param.requires_grad_(False)

        rank = int(cfg.TRAINER.LORA.RANK)
        alpha = float(cfg.TRAINER.LORA.ALPHA)
        dropout = float(cfg.TRAINER.LORA.DROPOUT)
        n_blocks = int(cfg.TRAINER.LORA.N_BLOCKS)
        attn_in = bool(cfg.TRAINER.LORA.ATTN_IN)
        attn_out = bool(cfg.TRAINER.LORA.ATTN_OUT)
        mlp_fc = bool(cfg.TRAINER.LORA.MLP_FC)
        mlp_proj = bool(cfg.TRAINER.LORA.MLP_PROJ)

        blocks = self.visual.transformer.resblocks
        start = max(0, len(blocks) - n_blocks)
        for idx in range(start, len(blocks)):
            blk = blocks[idx]
            blk.attn = LoRAMultiheadAttention(
                blk.attn,
                rank=rank,
                alpha=alpha,
                apply_in=attn_in,
                apply_out=attn_out,
            )
            if mlp_fc:
                blk.mlp.c_fc = LoRALinear(blk.mlp.c_fc, rank=rank, alpha=alpha, dropout=dropout)
            if mlp_proj:
                blk.mlp.c_proj = LoRALinear(blk.mlp.c_proj, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, image):
        return self.visual(image)


class OpenCLIPLoRAOnlyModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.image_encoder = LoRAOpenCLIPImageEncoder(cfg, clip_model)
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
class OpenCLIPLoRAOnlyMultiLabel(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LORA.PREC in ["fp16", "fp32", "amp"]

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

        print("Building OpenCLIP + visual LoRA only model for multi-label classification")
        self.model = OpenCLIPLoRAOnlyModel(cfg, self.classnames, clip_model, tokenizer)

        for name, param in self.model.named_parameters():
            if "lora_" in name:
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
        self.scaler = GradScaler() if cfg.TRAINER.LORA.PREC == "amp" else None
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.LORA.PREC

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
