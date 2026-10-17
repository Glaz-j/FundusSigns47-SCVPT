"""Model registry."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class NormalizedModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        forward_mode: str = "backbone",
    ):
        super().__init__()
        self.backbone = backbone
        self.forward_mode = forward_mode
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.forward_mode == "features_head":
            feat = self.backbone.forward_features(x)
            return self.backbone.head(feat)
        if self.forward_mode == "forward_then_head":
            feat = self.backbone(x)
            return self.backbone.head(feat)
        return self.backbone(x)


class SqueezeExcitation2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.avg_pool(x))
        return x * scale


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = inputs @ self.weight
        outputs = adj @ support
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


class ResNetMultiLabelBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet101LinearHead(ResNetMultiLabelBase):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__(backbone, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.forward_features(x)
        pooled = self.pool(feat_map).flatten(1)
        return self.fc(pooled)


class ResNet101MLGCN(ResNetMultiLabelBase):
    def __init__(self, backbone: nn.Module, num_classes: int, use_attention: bool = False):
        super().__init__(backbone, num_classes)
        self.use_attention = use_attention
        self.attention = SqueezeExcitation2d(2048) if use_attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_embed = nn.Parameter(torch.empty(num_classes, 256))
        self.gc1 = GraphConvolution(256, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.gc_relu = nn.LeakyReLU(0.2, inplace=True)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("adjacency", torch.eye(num_classes, dtype=torch.float32))
        nn.init.trunc_normal_(self.class_embed, std=0.02)

    def _normalized_adjacency(self) -> torch.Tensor:
        adj = self.adjacency
        degree = adj.sum(dim=1).clamp_min(1e-6)
        d_inv_sqrt = degree.pow(-0.5)
        norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
        return norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.forward_features(x)
        feat_map = self.attention(feat_map)
        pooled = self.pool(feat_map).flatten(1)
        adj = self._normalized_adjacency()
        class_nodes = self.gc1(self.class_embed, adj)
        class_nodes = self.gc_relu(class_nodes)
        class_nodes = self.gc2(class_nodes, adj)
        logits = pooled @ class_nodes.t()
        return logits * self.logit_scale.exp().clamp(max=100.0) / math.sqrt(class_nodes.shape[1])

    def initialize_from_loader(self, loader, device):
        cooc = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, targets, _ in loader:
            targets = targets.float()
            counts += targets.sum(dim=0)
            cooc += targets.t() @ targets
        denom = counts.unsqueeze(1).clamp_min(1.0)
        conditional = cooc / denom
        conditional = conditional.clamp(min=0.0, max=1.0)
        conditional.fill_diagonal_(1.0)
        adj = 0.5 * (conditional + conditional.t())
        adj.fill_diagonal_(1.0)
        self.adjacency.copy_(adj.to(self.adjacency.device))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class ConvNeXtTinyMLDecoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_queries: int = 16,
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.features = backbone.features
        self.input_proj = nn.Linear(768, decoder_dim)
        self.query_embed = nn.Parameter(torch.randn(num_queries, decoder_dim) * 0.02)
        self.decoder = nn.ModuleList(
            [CrossAttentionBlock(decoder_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        class_to_query = torch.arange(num_classes) % num_queries
        self.register_buffer("class_to_query", class_to_query, persistent=False)
        self.class_proj = nn.Parameter(torch.randn(num_classes, decoder_dim) * 0.02)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.input_proj(tokens)
        q = self.query_embed.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for block in self.decoder:
            q = block(q, tokens)
        q = self.decoder_norm(q)
        per_class_q = q[:, self.class_to_query, :]
        logits = (per_class_q * self.class_proj.unsqueeze(0)).sum(dim=-1) + self.class_bias
        return logits


class RetFoundBase(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def classify_from_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # RETFound with global_pool=True averages patch tokens only.
        pooled = x[:, 1:, :].mean(dim=1)
        pooled = self.backbone.fc_norm(pooled)
        return self.backbone.head(pooled)

    def extract_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def parameter_groups(self, train_cfg: dict):
        lr = float(train_cfg["learning_rate"])
        wd = float(train_cfg["weight_decay"])
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr, "weight_decay": wd}]


class RetFoundFullFinetune(RetFoundBase):
    def extract_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_pooled_features(x)
        return self.backbone.head(feat)

    def parameter_groups(self, train_cfg: dict):
        backbone_lr = float(train_cfg.get("backbone_learning_rate", train_cfg["learning_rate"]))
        head_lr = float(train_cfg.get("head_learning_rate", train_cfg["learning_rate"]))
        wd = float(train_cfg["weight_decay"])
        layer_decay = float(train_cfg.get("layer_decay", 1.0))

        groups = []
        stem_params = []
        named_backbone = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head.") or name.startswith("fc_norm."):
                continue
            if name.startswith("blocks."):
                block_idx = int(name.split(".")[1])
                named_backbone.append((block_idx, param))
            else:
                stem_params.append(param)

        if stem_params:
            stem_scale = layer_decay ** len(self.backbone.blocks)
            groups.append({"params": stem_params, "lr": backbone_lr * stem_scale, "weight_decay": wd})

        block_buckets = {}
        for block_idx, param in named_backbone:
            block_buckets.setdefault(block_idx, []).append(param)
        for block_idx, params in sorted(block_buckets.items()):
            scale = layer_decay ** (len(self.backbone.blocks) - 1 - block_idx)
            groups.append({"params": params, "lr": backbone_lr * scale, "weight_decay": wd})

        head_params = []
        for module_name in ["fc_norm", "head"]:
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                head_params.extend([p for p in module.parameters() if p.requires_grad])
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": wd})
        return groups


class RetFoundPartialFinetune(RetFoundBase):
    def __init__(self, backbone: nn.Module, train_last_n_blocks: int):
        super().__init__(backbone)
        self.train_last_n_blocks = train_last_n_blocks
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.blocks[-train_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for module_name in ["fc_norm", "head"]:
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def extract_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_pooled_features(x)
        return self.backbone.head(feat)

    def parameter_groups(self, train_cfg: dict):
        backbone_lr = float(train_cfg.get("backbone_learning_rate", train_cfg["learning_rate"]))
        head_lr = float(train_cfg.get("head_learning_rate", train_cfg["learning_rate"]))
        wd = float(train_cfg["weight_decay"])
        groups = []

        block_params = []
        for block in self.backbone.blocks[-self.train_last_n_blocks:]:
            block_params.extend([p for p in block.parameters() if p.requires_grad])
        if block_params:
            groups.append({"params": block_params, "lr": backbone_lr, "weight_decay": wd})

        head_params = []
        for module_name in ["fc_norm", "head"]:
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                head_params.extend([p for p in module.parameters() if p.requires_grad])
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": wd})
        return groups


class DeepVisualPromptTuning(RetFoundBase):
    def __init__(self, backbone: nn.Module, prompt_len: int):
        super().__init__(backbone)
        self.prompt_len = prompt_len
        embed_dim = backbone.embed_dim
        num_layers = len(backbone.blocks)
        self.prompts = nn.Parameter(torch.zeros(num_layers, prompt_len, embed_dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.head.parameters():
            param.requires_grad = True
        if hasattr(self.backbone, "fc_norm"):
            for param in self.backbone.fc_norm.parameters():
                param.requires_grad = True

    def extract_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        bsz = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        for idx, block in enumerate(self.backbone.blocks):
            prompts = self.prompts[idx].unsqueeze(0).expand(bsz, -1, -1)
            x = torch.cat([x[:, :1, :], prompts, x[:, 1:, :]], dim=1)
            x = block(x)
            x = torch.cat([x[:, :1, :], x[:, 1 + self.prompt_len :, :]], dim=1)
        pooled = x[:, 1:, :].mean(dim=1)
        return self.backbone.fc_norm(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_pooled_features(x)
        return self.backbone.head(feat)

    def parameter_groups(self, train_cfg: dict):
        lr = float(train_cfg.get("prompt_learning_rate", train_cfg["learning_rate"]))
        head_lr = float(train_cfg.get("head_learning_rate", lr))
        wd = float(train_cfg["weight_decay"])
        groups = [{"params": [self.prompts], "lr": lr, "weight_decay": wd}]
        head_params = []
        for module_name in ["fc_norm", "head"]:
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                head_params.extend([p for p in module.parameters() if p.requires_grad])
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": wd})
        return groups


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base_out + lora_out


class RetFoundLoRA(RetFoundBase):
    def __init__(self, backbone: nn.Module, rank: int, alpha: float):
        super().__init__(backbone)
        self.rank = rank
        self.alpha = alpha
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
        for module_name in ["fc_norm", "head"]:
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def extract_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_pooled_features(x)
        return self.backbone.head(feat)

    def parameter_groups(self, train_cfg: dict):
        lr = float(train_cfg.get("lora_learning_rate", train_cfg["learning_rate"]))
        head_lr = float(train_cfg.get("head_learning_rate", lr))
        wd = float(train_cfg["weight_decay"])
        lora_params = []
        head_params = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_a, module.lora_b])
        for module_name in ["fc_norm", "head"]:
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                head_params.extend([p for p in module.parameters() if p.requires_grad])
        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": lr, "weight_decay": wd})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": wd})
        return groups


class PerClassAffineHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return logits * self.scale + self.bias


class LabelCorrelationHead(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return logits + self.net(logits)


class RetrievalPrototypeHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.zeros(num_classes, feat_dim), requires_grad=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.mix = nn.Parameter(torch.tensor(0.5))

    def initialize(self, prototypes: torch.Tensor):
        self.prototypes.data.copy_(prototypes)

    def forward(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(features, dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)
        proto_logits = feats @ protos.t() * self.logit_scale.exp().clamp(max=100)
        mix = torch.sigmoid(self.mix)
        return logits + mix * proto_logits


class RetFoundExperimentWrapper(nn.Module):
    def __init__(self, base: RetFoundBase, aux_head: nn.Module | None = None):
        super().__init__()
        self.base = base
        self.aux_head = aux_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.base.extract_pooled_features(x)
        logits = self.base.backbone.head(feats)
        if self.aux_head is not None:
            logits = self.aux_head(logits, feats)
        return logits

    def parameter_groups(self, train_cfg: dict):
        groups = self.base.parameter_groups(train_cfg)
        if self.aux_head is not None:
            aux_params = [p for p in self.aux_head.parameters() if p.requires_grad]
            if aux_params:
                groups.append(
                    {
                        "params": aux_params,
                        "lr": float(train_cfg.get("head_learning_rate", train_cfg["learning_rate"])),
                        "weight_decay": float(train_cfg["weight_decay"]),
                    }
                )
        return groups

    def initialize_from_loader(self, loader, device):
        if not isinstance(self.aux_head, RetrievalPrototypeHead):
            return
        self.base.eval()
        feats_sum = torch.zeros_like(self.aux_head.prototypes.data)
        counts = torch.zeros(self.aux_head.prototypes.shape[0], device=device)
        with torch.no_grad():
            for images, targets, _ in loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                feats = self.base.extract_pooled_features(images)
                for cls_idx in range(targets.shape[1]):
                    mask = targets[:, cls_idx] > 0.5
                    if mask.any():
                        feats_sum[cls_idx] += feats[mask].sum(dim=0)
                        counts[cls_idx] += mask.sum()
        counts = counts.clamp_min(1.0).unsqueeze(1)
        self.aux_head.initialize((feats_sum / counts).detach().cpu().to(self.aux_head.prototypes.device))
        self.base.train()


@lru_cache(maxsize=1)
def _retfound_modules():
    repo_dir = PROJECT_ROOT / ".model_cache" / "RETFound_MAE"
    if not repo_dir.exists():
        raise FileNotFoundError(f"RETFound repo not found at {repo_dir}")
    models_vit = _load_module("retfound_models_vit_local", repo_dir / "models_vit.py")
    pos_embed = _load_module("retfound_pos_embed_local", repo_dir / "util" / "pos_embed.py")
    return models_vit, pos_embed


def _load_retfound_backbone(num_classes: int) -> nn.Module:
    if timm is None:
        raise ValueError("timm is required for RETFound but is not installed")
    models_vit, pos_embed = _retfound_modules()
    weight_path = PROJECT_ROOT / ".model_cache" / "RETFound_MAE" / "weights" / "RETFound_cfp_weights.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"RETFound weights not found at {weight_path}")

    model = models_vit.vit_large_patch16(num_classes=num_classes, drop_path_rate=0.2, global_pool=True)
    checkpoint = torch.load(weight_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]
    pos_embed.interpolate_pos_embed(model, checkpoint_model)
    model.load_state_dict(checkpoint_model, strict=False)
    return model


def _create_retfound_model(num_classes: int) -> nn.Module:
    return NormalizedModel(
        _load_retfound_backbone(num_classes),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        forward_mode="features_head",
    )


def _create_retfound_variant(model_cfg: dict) -> nn.Module:
    variant = model_cfg.get("variant", "full")
    num_classes = model_cfg["num_classes"]
    backbone = _load_retfound_backbone(num_classes)
    if variant == "full":
        return RetFoundFullFinetune(backbone)
    if variant == "partial":
        return RetFoundPartialFinetune(backbone, train_last_n_blocks=int(model_cfg.get("train_last_n_blocks", 4)))
    if variant == "dvpt":
        return DeepVisualPromptTuning(backbone, prompt_len=int(model_cfg.get("prompt_len", 8)))
    if variant == "lora":
        return RetFoundLoRA(
            backbone,
            rank=int(model_cfg.get("lora_rank", 8)),
            alpha=float(model_cfg.get("lora_alpha", 16.0)),
        )
    raise ValueError(f"Unsupported RETFound variant: {variant}")


def _create_retfound_experiment(model_cfg: dict) -> nn.Module:
    base = _create_retfound_variant(model_cfg)
    aux_type = model_cfg.get("aux_head", "none")
    aux_head = None
    if aux_type == "affine":
        aux_head = PerClassAffineHead(int(model_cfg["num_classes"]))
    elif aux_type == "labelcorr":
        aux_head = LabelCorrelationHead(
            int(model_cfg["num_classes"]),
            hidden_dim=int(model_cfg.get("labelcorr_hidden_dim", 96)),
        )
    elif aux_type == "retrievalproto":
        aux_head = RetrievalPrototypeHead(base.backbone.embed_dim, int(model_cfg["num_classes"]))
    elif aux_type == "none":
        aux_head = None
    else:
        raise ValueError(f"Unsupported aux_head: {aux_type}")
    return RetFoundExperimentWrapper(base, aux_head)


@lru_cache(maxsize=1)
def _visionfm_modules():
    repo_dir = PROJECT_ROOT / "external" / "model_sources" / "VisionFM"
    if not repo_dir.exists():
        raise FileNotFoundError(f"VisionFM repo not found at {repo_dir}")
    utils_stub = types.ModuleType("utils")
    utils_stub.trunc_normal_ = nn.init.trunc_normal_
    sys.modules.setdefault("utils", utils_stub)
    vit = _load_module("visionfm_vit_local", repo_dir / "models" / "vision_transformer.py")
    return vit


def _visionfm_resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = nn.functional.interpolate(posemb_grid, size=gs_new, mode="bicubic", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


def _visionfm_stats():
    return (
        (0.423737496137619, 0.2609460651874542, 0.128403902053833),
        (0.29482534527778625, 0.20167365670204163, 0.13668020069599152),
    )


def _create_visionfm_model(num_classes: int) -> nn.Module:
    vit = _visionfm_modules()
    weight_path = PROJECT_ROOT / ".model_cache" / "VisionFM" / "VFM_Fundus_weights.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"VisionFM weights not found at {weight_path}")

    model = vit.vit_base(patch_size=16, num_classes=num_classes)
    state = torch.load(weight_path, map_location="cpu")
    checkpoint_model = state["model"] if "model" in state else state
    checkpoint_model = {k.replace("module.", "").replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    if "pos_embed" in checkpoint_model and checkpoint_model["pos_embed"].shape != model.pos_embed.shape:
        checkpoint_model["pos_embed"] = _visionfm_resize_pos_embed(
            checkpoint_model["pos_embed"],
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )
    model.load_state_dict(checkpoint_model, strict=False)
    fundus_mean, fundus_std = _visionfm_stats()
    return NormalizedModel(
        model,
        mean=tuple(fundus_mean),
        std=tuple(fundus_std),
        forward_mode="forward_then_head",
    )


def _build_resnet101_backbone(pretrained: bool) -> nn.Module:
    weights = models.ResNet101_Weights.DEFAULT if pretrained else None
    return models.resnet101(weights=weights)


def _build_convnext_tiny_backbone(pretrained: bool) -> nn.Module:
    if not pretrained:
        return models.convnext_tiny(weights=None)
    weight_path = PROJECT_ROOT / ".model_cache" / "torch" / "hub" / "checkpoints" / "convnext_tiny-983f1562.pth"
    model = models.convnext_tiny(weights=None)
    if weight_path.exists():
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        return model
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    return models.convnext_tiny(weights=weights)


def _create_convnext_mldecoder(model_cfg: dict) -> nn.Module:
    backbone = _build_convnext_tiny_backbone(bool(model_cfg.get("pretrained", True)))
    return ConvNeXtTinyMLDecoder(
        backbone,
        num_classes=int(model_cfg["num_classes"]),
        num_queries=int(model_cfg.get("num_queries", 16)),
        decoder_dim=int(model_cfg.get("decoder_dim", 256)),
        num_heads=int(model_cfg.get("decoder_heads", 8)),
        num_layers=int(model_cfg.get("decoder_layers", 2)),
    )


def create_model(name: str | dict, num_classes: int | None = None, pretrained: bool = True) -> nn.Module:
    if isinstance(name, dict):
        model_cfg = dict(name)
        model_name = model_cfg["name"]
        model_num_classes = int(model_cfg["num_classes"])
        if model_name == "convnext_tiny_mldecoder":
            return _create_convnext_mldecoder(model_cfg)
        if model_name == "retfound_vitl16_ft":
            return _create_retfound_variant(model_cfg)
        if model_name == "retfound_vitl16_exp":
            return _create_retfound_experiment(model_cfg)
        name = model_name
        num_classes = model_num_classes
    if num_classes is None:
        raise ValueError("num_classes must be provided")
    if name == "retfound_vitl16":
        return _create_retfound_model(num_classes)
    if name == "visionfm_vitb16":
        return _create_visionfm_model(num_classes)
    if name == "convnext_tiny_mldecoder":
        return ConvNeXtTinyMLDecoder(_build_convnext_tiny_backbone(pretrained), num_classes)
    if name == "resnet101":
        return ResNet101LinearHead(_build_resnet101_backbone(pretrained), num_classes)
    if name == "resnet101_mlgcn":
        return ResNet101MLGCN(_build_resnet101_backbone(pretrained), num_classes, use_attention=False)
    if name == "resnet101_attn_mlgcn":
        return ResNet101MLGCN(_build_resnet101_backbone(pretrained), num_classes, use_attention=True)
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if name == "swin_tiny":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if name == "inception_v3":
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True if pretrained else False)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "vit_b16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    if name == "deit_small":
        if timm is None:
            raise ValueError("timm is required for deit_small but is not installed")
        return timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported model: {name}")
