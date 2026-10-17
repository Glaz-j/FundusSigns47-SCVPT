import open_clip
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_pretrained_weights

from open_clip.transformer import _expand_token
from trainers.openclip_vpt_deep_only_multilabel import (
    OpenCLIPVPTDeepOnlyMultiLabel,
    build_multilabel_criterion,
)


class DeepVPTClassRegionImageEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.n_ctx = int(cfg.TRAINER.VPT.N_CTX)
        self.prompt_dropout = nn.Dropout(float(cfg.TRAINER.VPT.DROPOUT))

        width = self.visual.conv1.out_channels
        depth = len(self.visual.transformer.resblocks)
        requested_deep_layers = int(getattr(cfg.TRAINER.VPT, "N_DEEP_LAYERS", -1))
        if requested_deep_layers <= 0:
            requested_deep_layers = depth - 1
        self.n_deep_layers = min(max(requested_deep_layers, 0), max(depth - 1, 0))
        dtype = self.visual.conv1.weight.dtype

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
        return torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)

    def _replace_prompt(self, x, prompt):
        prompt = prompt.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        prompt = self.prompt_dropout(prompt)
        image_tokens = x[:, 1 + self.n_ctx :, :]
        return torch.cat([x[:, :1, :], prompt, image_tokens], dim=1)

    def forward(self, image):
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

        pooled, tokens = visual._pool(x)
        patch_tokens = tokens[:, self.n_ctx :, :]
        if visual.proj is not None:
            pooled = pooled @ visual.proj
            patch_tokens = patch_tokens @ visual.proj
        return pooled, patch_tokens


class OpenCLIPDeepVPTClassRegionModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.image_encoder = DeepVPTClassRegionImageEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad_(bool(cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE))

        template = str(cfg.TRAINER.REGION.QUERY_TEMPLATE)
        prompts = [template.format(name.replace("_", " ")) for name in classnames]
        tokenized_prompts = tokenizer(prompts)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embed_dim = text_features.shape[1]
        self.register_buffer("text_features", text_features.float())
        self.class_queries = nn.Parameter(text_features.clone().float())
        self.query_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.region_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=int(cfg.TRAINER.REGION.ATTN_HEADS),
            batch_first=True,
        )
        self.region_gate = nn.Linear(embed_dim, 1)
        self.region_head = nn.Linear(embed_dim, 1)

        nn.init.zeros_(self.region_head.weight)
        nn.init.zeros_(self.region_head.bias)
        nn.init.zeros_(self.region_gate.weight)
        nn.init.constant_(self.region_gate.bias, float(cfg.TRAINER.REGION.GATE_INIT_BIAS))

    def forward(self, image):
        pooled, patch_tokens = self.image_encoder(image)
        image_features = pooled / pooled.norm(dim=-1, keepdim=True)
        base_logits = self.logit_scale.exp() * image_features.float() @ self.text_features.t()

        queries = self.query_norm(self.class_queries).unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        kv = self.patch_norm(patch_tokens.float())
        attn_output, _ = self.region_attn(queries, kv, kv, need_weights=False)
        gate = torch.sigmoid(self.region_gate(attn_output)).squeeze(-1)
        region_delta = self.region_head(attn_output).squeeze(-1)
        return base_logits + gate * region_delta


@TRAINER_REGISTRY.register()
class OpenCLIPDeepVPTClassRegionMultiLabel(OpenCLIPVPTDeepOnlyMultiLabel):
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

        print("Building OpenCLIP DeepVPT + class-aware region head for multi-label classification")
        self.model = OpenCLIPDeepVPTClassRegionModel(cfg, self.classnames, clip_model, tokenizer)
        print(
            "Region config: "
            f"visual_n_ctx={cfg.TRAINER.VPT.N_CTX}, "
            f"n_deep_layers={self.model.image_encoder.n_deep_layers}, "
            f"attn_heads={cfg.TRAINER.REGION.ATTN_HEADS}, "
            f"gate_init_bias={cfg.TRAINER.REGION.GATE_INIT_BIAS}, "
            f"train_logit_scale={cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE}, "
            f"train_proj={cfg.TRAINER.VPT.TRAIN_PROJ}, "
            f"train_norm={cfg.TRAINER.VPT.TRAIN_NORM}"
        )

        for name, param in self.model.named_parameters():
            trainable = False
            if "image_encoder.shallow_prompt" in name or "image_encoder.deep_prompts" in name:
                trainable = True
            elif name == "class_queries":
                trainable = True
            elif name.startswith("query_norm.") or name.startswith("patch_norm."):
                trainable = True
            elif name.startswith("region_attn.") or name.startswith("region_gate.") or name.startswith("region_head."):
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
