import os

import open_clip
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param

from trainers.openclip_coop_multilabel_cached import (
    OpenCLIPPromptLearner,
    OpenCLIPTextEncoder,
)
from trainers.openclip_vptdeep_coop_multilabel_cached import (
    OpenCLIPVPTDeepCoOpMultiLabelCached,
)


class CachedStage2ResidualModel(nn.Module):
    def __init__(
        self,
        cfg,
        clip_model,
        text_features,
        classnames=None,
        tokenizer=None,
        train_features=None,
        train_labels=None,
    ):
        super().__init__()
        self.mode = str(cfg.TRAINER.STAGE2.MODE).lower()
        self.rank = int(cfg.TRAINER.STAGE2.RANK)
        self.feature_dim = int(text_features.shape[-1])
        self.num_classes = int(text_features.shape[0])
        self.tail_power = float(cfg.TRAINER.STAGE2.TAIL_POWER)
        self.tail_threshold = int(cfg.TRAINER.STAGE2.TAIL_THRESHOLD)
        self.head_threshold = int(cfg.TRAINER.STAGE2.HEAD_THRESHOLD)
        self.candidate_threshold = float(cfg.TRAINER.STAGE2.CANDIDATE_THRESHOLD)
        self.candidate_topk = int(cfg.TRAINER.STAGE2.CANDIDATE_TOPK)
        self.candidate_soft_scale = float(cfg.TRAINER.STAGE2.CANDIDATE_SOFT_SCALE)
        self.recall_helper_alpha = float(cfg.TRAINER.STAGE2.RECALL_HELPER_ALPHA)
        self.hardclass_use_mid = bool(cfg.TRAINER.STAGE2.HARDCLASS_USE_MID)

        self.register_buffer("base_text_features", text_features.float())
        self.logit_scale = nn.Parameter(
            clip_model.logit_scale.detach().float().clone(), requires_grad=False
        )
        if train_labels is not None:
            class_counts = train_labels.float().sum(dim=0)
        else:
            class_counts = torch.ones(self.num_classes)
        safe_counts = class_counts.clamp(min=1.0)
        max_count = float(safe_counts.max().item())
        tail_weights = torch.pow(max_count / safe_counts, self.tail_power)
        tail_weights = tail_weights / tail_weights.mean().clamp(min=1e-6)
        tail_mask = (class_counts <= self.tail_threshold).float()
        head_mask = (class_counts >= self.head_threshold).float()
        mid_mask = 1.0 - torch.clamp(tail_mask + head_mask, max=1.0)
        self.register_buffer("class_counts", class_counts.float())
        self.register_buffer("tail_weights", tail_weights.float())
        self.register_buffer("tail_mask", tail_mask)
        self.register_buffer("head_mask", head_mask)
        self.register_buffer("mid_mask", mid_mask)

        if self.mode == "affine":
            self.scale = nn.Parameter(torch.ones(self.num_classes))
            self.bias = nn.Parameter(torch.zeros(self.num_classes))
        elif self.mode == "taskres":
            self.delta_text = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
        elif self.mode == "hardneg_taskres":
            self.delta_text = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
        elif self.mode == "lowrank":
            self.text_u = nn.Parameter(torch.zeros(self.num_classes, self.rank))
            self.text_v = nn.Parameter(torch.zeros(self.rank, self.feature_dim))
            nn.init.normal_(self.text_u, std=0.02)
            nn.init.normal_(self.text_v, std=0.02)
        elif self.mode == "taskres_lowrank":
            self.delta_text = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
            self.text_u = nn.Parameter(torch.zeros(self.num_classes, self.rank))
            self.text_v = nn.Parameter(torch.zeros(self.rank, self.feature_dim))
            nn.init.normal_(self.text_u, std=0.02)
            nn.init.normal_(self.text_v, std=0.02)
        elif self.mode == "mlp":
            hidden_dim = int(cfg.TRAINER.STAGE2.HIDDEN_DIM)
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(cfg.TRAINER.STAGE2.DROPOUT)),
                nn.Linear(hidden_dim, self.num_classes),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        elif self.mode in {
            "coop_taskres",
            "coop_taskres_candidate",
            "coop_taskres_hardclass",
            "coop_taskres_recall",
        }:
            if classnames is None or tokenizer is None:
                raise ValueError(f"{self.mode} requires classnames and tokenizer")
            self.prompt_learner = OpenCLIPPromptLearner(cfg, classnames, clip_model, tokenizer)
            self.text_encoder = OpenCLIPTextEncoder(clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.delta_text = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
        elif self.mode == "coop_lowrank":
            if classnames is None or tokenizer is None:
                raise ValueError("coop_lowrank requires classnames and tokenizer")
            self.prompt_learner = OpenCLIPPromptLearner(cfg, classnames, clip_model, tokenizer)
            self.text_encoder = OpenCLIPTextEncoder(clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.text_u = nn.Parameter(torch.zeros(self.num_classes, self.rank))
            self.text_v = nn.Parameter(torch.zeros(self.rank, self.feature_dim))
            nn.init.normal_(self.text_u, std=0.02)
            nn.init.normal_(self.text_v, std=0.02)
        elif self.mode == "labelcorr":
            hidden_dim = int(cfg.TRAINER.STAGE2.HIDDEN_DIM)
            num_heads = int(cfg.TRAINER.STAGE2.CORR_HEADS)
            num_layers = int(cfg.TRAINER.STAGE2.CORR_LAYERS)
            self.logit_proj = nn.Linear(1, hidden_dim)
            self.class_embed = nn.Parameter(torch.empty(self.num_classes, hidden_dim))
            nn.init.normal_(self.class_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=float(cfg.TRAINER.STAGE2.DROPOUT),
                batch_first=True,
                activation="gelu",
            )
            self.corr_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.corr_out = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.corr_out.weight)
            nn.init.zeros_(self.corr_out.bias)
        elif self.mode == "tip":
            if train_features is None or train_labels is None:
                raise ValueError("tip mode requires train_features and train_labels")
            train_features = train_features.float()
            train_features = train_features / train_features.norm(dim=-1, keepdim=True)
            self.register_buffer("cache_keys", train_features)
            self.register_buffer("cache_values", train_labels.float())
            alpha_init = float(cfg.TRAINER.STAGE2.TIP_ALPHA_INIT)
            beta_init = float(cfg.TRAINER.STAGE2.TIP_BETA_INIT)
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init)))
            self.log_beta = nn.Parameter(torch.log(torch.tensor(beta_init)))
        elif self.mode == "bilinear":
            self.image_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
            nn.init.zeros_(self.image_proj.weight)
            self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        elif self.mode == "tail_taskres":
            self.delta_text = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
        elif self.mode == "tail_calib":
            self.delta_scale = nn.Parameter(torch.zeros(self.num_classes))
            self.bias = nn.Parameter(torch.zeros(self.num_classes))
        elif self.mode == "tail_labelcorr":
            hidden_dim = int(cfg.TRAINER.STAGE2.HIDDEN_DIM)
            num_heads = int(cfg.TRAINER.STAGE2.CORR_HEADS)
            num_layers = int(cfg.TRAINER.STAGE2.CORR_LAYERS)
            self.logit_proj = nn.Linear(1, hidden_dim)
            self.class_embed = nn.Parameter(torch.empty(self.num_classes, hidden_dim))
            nn.init.normal_(self.class_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=float(cfg.TRAINER.STAGE2.DROPOUT),
                batch_first=True,
                activation="gelu",
            )
            self.corr_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.corr_out = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.corr_out.weight)
            nn.init.zeros_(self.corr_out.bias)
        elif self.mode == "tail_tip":
            if train_features is None or train_labels is None:
                raise ValueError("tail_tip mode requires train_features and train_labels")
            train_features = train_features.float()
            train_features = train_features / train_features.norm(dim=-1, keepdim=True)
            prototypes = []
            for class_idx in range(self.num_classes):
                class_mask = train_labels[:, class_idx] > 0
                if class_mask.any():
                    proto = train_features[class_mask].mean(dim=0)
                    proto = proto / proto.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                else:
                    proto = torch.zeros(self.feature_dim)
                prototypes.append(proto)
            prototype_bank = torch.stack(prototypes, dim=0)
            self.register_buffer("prototype_bank", prototype_bank.float())
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(cfg.TRAINER.STAGE2.TIP_ALPHA_INIT))))
        elif self.mode == "headtail_resid":
            self.delta_shared = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
            self.delta_head = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
            self.delta_tail = nn.Parameter(torch.zeros(self.num_classes, self.feature_dim))
        else:
            raise ValueError(f"Unsupported TRAINER.STAGE2.MODE: {cfg.TRAINER.STAGE2.MODE}")

    def _base_logits(self, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features.float() @ self.base_text_features.t()

    def _residual_text_features(self):
        if self.mode == "taskres":
            text_features = self.base_text_features + self.delta_text
        elif self.mode == "hardneg_taskres":
            text_features = self.base_text_features + self.delta_text
        elif self.mode == "lowrank":
            text_features = self.base_text_features + self.text_u @ self.text_v
        elif self.mode == "taskres_lowrank":
            text_features = self.base_text_features + self.delta_text + self.text_u @ self.text_v
        elif self.mode in {
            "coop_taskres",
            "coop_taskres_candidate",
            "coop_taskres_hardclass",
            "coop_taskres_recall",
        }:
            prompts = self.prompt_learner()
            tokenized = self.tokenized_prompts.to(prompts.device)
            prompt_text = self.text_encoder(prompts, tokenized)
            prompt_text = prompt_text / prompt_text.norm(dim=-1, keepdim=True)
            text_features = prompt_text + self.delta_text
        elif self.mode == "coop_lowrank":
            prompts = self.prompt_learner()
            tokenized = self.tokenized_prompts.to(prompts.device)
            prompt_text = self.text_encoder(prompts, tokenized)
            prompt_text = prompt_text / prompt_text.norm(dim=-1, keepdim=True)
            text_features = prompt_text + self.text_u @ self.text_v
        elif self.mode == "tail_taskres":
            text_features = self.base_text_features + self.tail_weights.unsqueeze(-1) * self.delta_text
        elif self.mode == "headtail_resid":
            text_features = (
                self.base_text_features
                + self.delta_shared
                + self.head_mask.unsqueeze(-1) * self.delta_head
                + self.tail_weights.unsqueeze(-1) * self.tail_mask.unsqueeze(-1) * self.delta_tail
            )
        else:
            raise RuntimeError("Residual text features are only defined for text-residual modes")

        return text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, image_features):
        base_logits = self._base_logits(image_features)

        if self.mode == "affine":
            return base_logits * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)

        if self.mode == "tail_calib":
            scale = 1.0 + self.tail_weights * self.tail_mask * self.delta_scale
            bias = self.tail_weights * self.tail_mask * self.bias
            return base_logits * scale.unsqueeze(0) + bias.unsqueeze(0)

        if self.mode in {
            "taskres",
            "hardneg_taskres",
            "lowrank",
            "taskres_lowrank",
            "coop_taskres",
            "coop_lowrank",
            "tail_taskres",
            "headtail_resid",
        }:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self._residual_text_features()
            return self.logit_scale.exp() * image_features.float() @ text_features.t()

        if self.mode in {
            "coop_taskres_candidate",
            "coop_taskres_hardclass",
            "coop_taskres_recall",
        }:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self._residual_text_features()
            coop_logits = self.logit_scale.exp() * image_features.float() @ text_features.t()
            base_probs = torch.sigmoid(base_logits)

            topk = min(max(self.candidate_topk, 1), self.num_classes)
            candidate_mask = (base_probs >= self.candidate_threshold).float()
            topk_idx = torch.topk(base_probs, k=topk, dim=1).indices
            topk_mask = torch.zeros_like(candidate_mask).scatter_(1, topk_idx, 1.0)
            candidate_mask = torch.clamp(candidate_mask + topk_mask, max=1.0)

            if self.mode == "coop_taskres_candidate":
                return base_logits + candidate_mask * (coop_logits - base_logits)

            hard_class_mask = self.tail_mask
            if self.hardclass_use_mid:
                hard_class_mask = torch.clamp(self.tail_mask + self.mid_mask, max=1.0)

            if self.mode == "coop_taskres_hardclass":
                gate = candidate_mask * hard_class_mask.unsqueeze(0)
                return base_logits + gate * (coop_logits - base_logits)

            soft_gate = torch.sigmoid(
                self.candidate_soft_scale * (base_probs - self.candidate_threshold)
            )
            soft_gate = torch.maximum(soft_gate, topk_mask)
            positive_delta = torch.relu(coop_logits - base_logits)
            return base_logits + self.recall_helper_alpha * soft_gate * positive_delta

        if self.mode == "mlp":
            return base_logits + self.mlp(image_features.float())

        if self.mode == "labelcorr":
            corr_tokens = self.logit_proj(base_logits.unsqueeze(-1)) + self.class_embed.unsqueeze(0)
            corr_tokens = self.corr_encoder(corr_tokens)
            return base_logits + self.corr_out(corr_tokens).squeeze(-1)

        if self.mode == "tail_labelcorr":
            corr_tokens = self.logit_proj(base_logits.unsqueeze(-1)) + self.class_embed.unsqueeze(0)
            corr_tokens = self.corr_encoder(corr_tokens)
            residual = self.corr_out(corr_tokens).squeeze(-1)
            residual = residual * (1.0 + self.tail_weights * self.tail_mask).unsqueeze(0)
            return base_logits + residual

        if self.mode == "tip":
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            affinity = image_features.float() @ self.cache_keys.t()
            alpha = self.log_alpha.exp()
            beta = self.log_beta.exp()
            cache_logits = torch.exp(beta * (affinity - 1.0)) @ self.cache_values
            return base_logits + alpha * cache_logits

        if self.mode == "tail_tip":
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            proto_logits = self.logit_scale.exp() * image_features.float() @ self.prototype_bank.t()
            alpha = self.log_alpha.exp()
            return base_logits + alpha * proto_logits * (self.tail_weights * self.tail_mask).unsqueeze(0)

        if self.mode == "bilinear":
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            residual_features = self.image_proj(image_features.float())
            residual_logits = self.logit_scale.exp() * residual_features @ self.base_text_features.t()
            return base_logits + residual_logits + self.class_bias.unsqueeze(0)

        raise RuntimeError(f"Unexpected mode: {self.mode}")


@TRAINER_REGISTRY.register()
class OpenCLIPVPTDeepResidualMultiLabelCached(OpenCLIPVPTDeepCoOpMultiLabelCached):
    """Stage-2 residual/classifier adaptation on top of frozen tuned Deep VPT features."""

    def build_model(self):
        cfg = self.cfg
        print(f"Loading OpenCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess, _ = self._load_openclip()

        cache = self._build_or_load_feature_cache(clip_model, preprocess)
        frozen_model_path = os.path.abspath(os.path.expanduser(str(cfg.TRAINER.VPT.FROZEN_MODEL_PATH)))
        checkpoint = torch.load(frozen_model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        if "logit_scale" in state_dict:
            with torch.no_grad():
                clip_model.logit_scale.copy_(state_dict["logit_scale"])

        text_template = str(cfg.TRAINER.STAGE2.TEXT_TEMPLATE).strip()
        prompts = [text_template.format(name.replace("_", " ")) for name in self.classnames]
        model_name = cfg.MODEL.BACKBONE.NAME.split(":")[1]
        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            tokenized_prompts = tokenizer(prompts)
            text_features = clip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        clip_model = clip_model.cpu()
        self.train_loader_x = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(cache["splits"]["train"]["features"], cache["splits"]["train"]["labels"]),
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(cache["splits"]["val"]["features"], cache["splits"]["val"]["labels"]),
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(cache["splits"]["test"]["features"], cache["splits"]["test"]["labels"]),
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        print(
            f"Building Stage-2 residual model on frozen tuned Deep VPT features "
            f"(mode={cfg.TRAINER.STAGE2.MODE})"
        )
        self.model = CachedStage2ResidualModel(
            cfg,
            clip_model,
            text_features.float(),
            classnames=self.classnames,
            tokenizer=tokenizer,
            train_features=cache["splits"]["train"]["features"],
            train_labels=cache["splits"]["train"]["labels"],
        )
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("stage2_residual", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.STAGE2.PREC == "amp" else None
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward_backward(self, batch):
        features, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.STAGE2.PREC
        if prec == "amp":
            with autocast():
                output = self.model(features)
                loss = self._compute_loss(features, output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(features)
            loss = self._compute_loss(features, output, label)
            self.model_backward_and_update(loss)

        with torch.no_grad():
            probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()
            match = (preds == label).float().mean().item() * 100.0

        loss_summary = {"loss": loss.item(), "label_match": match}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def _compute_loss(self, features, output, label):
        loss = self.criterion(output, label)
        if self.cfg.TRAINER.STAGE2.MODE.lower() == "hardneg_taskres":
            with torch.no_grad():
                base_logits = self.model._base_logits(features)
                base_probs = torch.sigmoid(base_logits)
                thr = float(self.cfg.TRAINER.STAGE2.HARDNEG_THRESHOLD)
                alpha = float(self.cfg.TRAINER.STAGE2.HARDNEG_ALPHA)
                gamma = float(self.cfg.TRAINER.STAGE2.HARDNEG_GAMMA)
                hard_strength = ((base_probs - thr) / max(1e-6, 1.0 - thr)).clamp(min=0.0)
                weights = 1.0 + (1.0 - label) * alpha * torch.pow(hard_strength, gamma)
            loss = loss * weights
        return loss.mean()

    def parse_batch_train(self, batch):
        features, label = batch
        return features.to(self.device, non_blocking=True), label.to(
            self.device, non_blocking=True
        ).float()

    def parse_batch_test(self, batch):
        features, label = batch
        return features.to(self.device, non_blocking=True), label.to(
            self.device, non_blocking=True
        ).float()

    def model_inference(self, input):
        return self.model(input)
