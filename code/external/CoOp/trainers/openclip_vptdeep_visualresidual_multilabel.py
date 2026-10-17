import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param

from trainers.openclip_vpt_deep_only_multilabel import (
    OpenCLIPVPTDeepOnlyModel,
    OpenCLIPVPTDeepOnlyMultiLabel,
    RetinalImageDataset,
)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SmallCNNBackbone(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, 3, stride=2),
            ConvBNAct(32, 64, 3, stride=2),
            ConvBNAct(64, 128, 3, stride=2),
            ConvBNAct(128, 256, 3, stride=2),
            ConvBNAct(256, 256, 3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class InceptionLiteBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4
        self.branch1 = ConvBNAct(in_ch, branch_ch, kernel_size=1, padding=0)
        self.branch3 = nn.Sequential(
            ConvBNAct(in_ch, branch_ch, kernel_size=1, padding=0),
            ConvBNAct(branch_ch, branch_ch, kernel_size=3),
        )
        self.branch5 = nn.Sequential(
            ConvBNAct(in_ch, branch_ch, kernel_size=1, padding=0),
            ConvBNAct(branch_ch, branch_ch, kernel_size=5),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(in_ch, branch_ch, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return torch.cat(
            [
                self.branch1(x),
                self.branch3(x),
                self.branch5(x),
                self.branch_pool(x),
            ],
            dim=1,
        )


class InceptionLiteBackbone(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, 3, stride=2),
            ConvBNAct(32, 64, 3, stride=2),
            ConvBNAct(64, 96, 3, stride=1),
        )
        self.block1 = InceptionLiteBlock(96, 128)
        self.reduce1 = ConvBNAct(128, 160, 3, stride=2)
        self.block2 = InceptionLiteBlock(160, 192)
        self.reduce2 = ConvBNAct(192, 224, 3, stride=2)
        self.block3 = InceptionLiteBlock(224, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.reduce1(x)
        x = self.block2(x)
        x = self.reduce2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class FullInceptionV3Backbone(nn.Module):
    def __init__(self, num_classes, pretrained=True, checkpoint_path=""):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True if pretrained else False)
        model.aux_logits = False
        model.AuxLogits = None
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.model = model
        self.extractor = create_feature_extractor(self.model, return_nodes={"Mixed_7c": "featmap"})
        self.feat_dim = feat_dim
        self.head = nn.Linear(feat_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        checkpoint_path = str(checkpoint_path).strip()
        if checkpoint_path:
            self.load_classifier_checkpoint(checkpoint_path)

    def forward_features(self, x):
        featmap = self.forward_feature_map(x)
        return F.adaptive_avg_pool2d(featmap, 1).flatten(1)

    def forward_feature_map(self, x):
        return self.extractor(x)["featmap"]

    def forward_logits(self, x):
        feats = self.forward_features(x)
        logits = self.head(feats)
        return feats, logits

    def load_classifier_checkpoint(self, checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("fc."):
                mapped["head." + key[3:]] = value
            else:
                mapped["model." + key] = value
        self.load_state_dict(mapped, strict=False)


class FrozenDeepVPTVisualResidualModel(nn.Module):
    def __init__(self, cfg, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad_(False)

        self.mode = str(cfg.TRAINER.STAGE2VIS.MODE).lower()
        self.feature_dim = int(base_model.text_features.shape[-1])
        self.num_classes = int(base_model.text_features.shape[0])
        hidden_dim = int(cfg.TRAINER.STAGE2VIS.HIDDEN_DIM)
        self.candidate_threshold = float(cfg.TRAINER.STAGE2VIS.CANDIDATE_THRESHOLD)
        self.candidate_topk = int(cfg.TRAINER.STAGE2VIS.CANDIDATE_TOPK)
        self.candidate_soft_scale = float(cfg.TRAINER.STAGE2VIS.CANDIDATE_SOFT_SCALE)
        self.distill_lambda = float(cfg.TRAINER.STAGE2VIS.DISTILL_LAMBDA)
        self.distill_temperature = float(cfg.TRAINER.STAGE2VIS.DISTILL_TEMPERATURE)
        self.hardneg_threshold = float(cfg.TRAINER.STAGE2VIS.HARDNEG_THRESHOLD)
        self.hardneg_alpha = float(cfg.TRAINER.STAGE2VIS.HARDNEG_ALPHA)
        self.hardneg_gamma = float(cfg.TRAINER.STAGE2VIS.HARDNEG_GAMMA)
        self.local_patch_ratio = float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_RATIO)
        self.local_patch_weight = float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)
        self.classspec_temp = float(cfg.TRAINER.STAGE2VIS.CLASSSPEC_TEMP)
        self.cascade_low_threshold = float(cfg.TRAINER.STAGE2VIS.CASCADE_LOW_THRESHOLD)
        self.cascade_high_threshold = float(cfg.TRAINER.STAGE2VIS.CASCADE_HIGH_THRESHOLD)
        self.confusion_mix = float(cfg.TRAINER.STAGE2VIS.CONFUSION_MIX)
        self.roi_topk = int(getattr(cfg.TRAINER.STAGE2VIS, "ROI_TOPK", 4))
        self.roi_ratio_large = float(getattr(cfg.TRAINER.STAGE2VIS, "ROI_RATIO_LARGE", 0.65))
        self.roi_ratio_medium = float(getattr(cfg.TRAINER.STAGE2VIS, "ROI_RATIO_MEDIUM", 0.45))
        self.roi_ratio_small = float(getattr(cfg.TRAINER.STAGE2VIS, "ROI_RATIO_SMALL", 0.30))
        self.memory_topk = int(getattr(cfg.TRAINER.STAGE2VIS, "MEMORY_TOPK", 32))
        self.memory_temp = float(getattr(cfg.TRAINER.STAGE2VIS, "MEMORY_TEMP", 10.0))
        self.memory_tail_threshold = int(getattr(cfg.TRAINER.STAGE2VIS, "MEMORY_TAIL_THRESHOLD", 40))
        self.multi_proto_k = int(getattr(cfg.TRAINER.STAGE2VIS, "MULTI_PROTO_K", 4))
        self.confusion_exemplar_topk = int(getattr(cfg.TRAINER.STAGE2VIS, "CONFUSION_EXEMPLAR_TOPK", 16))
        self.retrieval_alpha_init = float(getattr(cfg.TRAINER.STAGE2VIS, "RETRIEVAL_ALPHA_INIT", 0.05))
        teacher_ckpt = str(cfg.TRAINER.STAGE2VIS.TEACHER_CKPT).strip()
        inception_pretrained = bool(getattr(cfg.TRAINER.STAGE2VIS, "INCEPTION_PRETRAINED", True))
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("confusion_prior", torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer("fp_prior", torch.ones(self.num_classes))
        self.register_buffer("memory_bank", torch.empty(0, self.feature_dim))
        self.register_buffer("memory_labels", torch.empty(0, self.num_classes))
        self.register_buffer("prototype_bank", torch.empty(self.num_classes, self.feature_dim))
        self.register_buffer("subprototype_bank", torch.empty(self.num_classes, self.multi_proto_k, self.feature_dim))
        self.register_buffer("subprototype_mask", torch.zeros(self.num_classes, self.multi_proto_k))
        self.register_buffer("tail_prototype_bank", torch.empty(self.num_classes, self.feature_dim))
        self.register_buffer("tail_class_mask", torch.zeros(self.num_classes))

        if self.mode == "cnn_logit":
            self.branch = SmallCNNBackbone(hidden_dim)
            self.head = nn.Linear(hidden_dim, self.num_classes)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "cnn_feature":
            self.branch = SmallCNNBackbone(hidden_dim)
            self.feature_proj = nn.Linear(hidden_dim, self.feature_dim)
            nn.init.zeros_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "patch_logit":
            self.branch = SmallCNNBackbone(hidden_dim)
            self.head = nn.Linear(hidden_dim, self.num_classes)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.patch_ratio = float(cfg.TRAINER.STAGE2VIS.PATCH_RATIO)
        elif self.mode == "inception_aux":
            self.branch = InceptionLiteBackbone(hidden_dim)
            self.head = nn.Linear(hidden_dim, self.num_classes)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_logit":
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_feature":
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True)
            self.feature_proj = nn.Linear(self.branch.feat_dim, self.feature_dim)
            nn.init.zeros_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_gated":
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True)
            self.gate = nn.Sequential(
                nn.Linear(self.feature_dim + self.branch.feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_classes),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        elif self.mode == "full_inception_candidate":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(
                self.num_classes, pretrained=inception_pretrained, checkpoint_path=ckpt
            )
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_adaptive":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(
                self.num_classes, pretrained=inception_pretrained, checkpoint_path=ckpt
            )
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            init_thr = torch.full((self.num_classes,), float(cfg.TRAINER.STAGE2VIS.CANDIDATE_THRESHOLD))
            init_thr = torch.logit(init_thr.clamp(1e-4, 1 - 1e-4))
            self.candidate_threshold_param = nn.Parameter(init_thr)
            init_scale = torch.full((self.num_classes,), float(cfg.TRAINER.STAGE2VIS.CANDIDATE_SOFT_SCALE))
            self.candidate_scale_param = nn.Parameter(init_scale)
        elif self.mode == "full_inception_candidate_gated":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.gate = nn.Sequential(
                nn.Linear(self.feature_dim + self.branch.feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_classes),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        elif self.mode == "full_inception_candidate_classspec":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.class_alpha = nn.Parameter(torch.zeros(self.num_classes))
            self.class_bias = nn.Parameter(torch.zeros(self.num_classes))
        elif self.mode == "full_inception_candidate_localpatch":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.local_alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)))
        elif self.mode == "full_inception_candidate_multiscale":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.local_alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)))
        elif self.mode == "full_inception_candidate_topkpatch":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_dualbranch":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.local_alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)))
        elif self.mode == "full_inception_candidate_mivote":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_hardneg":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_fpfocus":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_confusion":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_calibrated":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.log_scale = nn.Parameter(torch.zeros(self.num_classes))
            self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        elif self.mode == "full_inception_candidate_cascade":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_cascadecal":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.log_scale = nn.Parameter(torch.zeros(self.num_classes))
            self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        elif self.mode == "full_inception_candidate_distill":
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            if not teacher_ckpt:
                raise ValueError("full_inception_candidate_distill requires TRAINER.STAGE2VIS.TEACHER_CKPT")
            self.teacher = FullInceptionV3Backbone(
                self.num_classes, pretrained=False, checkpoint_path=teacher_ckpt
            )
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        elif self.mode == "full_inception_candidate_camdual":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.local_alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)))
        elif self.mode == "full_inception_candidate_cammil4":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_cammil8":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
        elif self.mode == "full_inception_candidate_campyramid":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.local_alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)))
        elif self.mode == "full_inception_candidate_retrievalknn":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_retrievalproto":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_retrievalhybrid":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.local_alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_tailproto":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_retrievalproto_cs":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_retrievalprotogated":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
            self.retrieval_gate = nn.Sequential(
                nn.Linear(self.feature_dim + self.num_classes, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_classes),
            )
            nn.init.zeros_(self.retrieval_gate[-1].weight)
            nn.init.zeros_(self.retrieval_gate[-1].bias)
        elif self.mode == "full_inception_candidate_retrievalconfusion":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_retrievalexemplar":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_confusionprototopk":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_confusionexemplartopk":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_multiprotoretrieval":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.retrieval_alpha = nn.Parameter(torch.tensor(self.retrieval_alpha_init))
        elif self.mode == "full_inception_candidate_candidateroi":
            ckpt = teacher_ckpt if bool(cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER) else ""
            self.branch = FullInceptionV3Backbone(self.num_classes, pretrained=True, checkpoint_path=ckpt)
            self.alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.ALPHA_INIT)))
            self.local_alpha = nn.Parameter(torch.tensor(float(cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT)))
        else:
            raise ValueError(f"Unsupported TRAINER.STAGE2VIS.MODE: {cfg.TRAINER.STAGE2VIS.MODE}")

    def _base_features_logits(self, image):
        image_features = self.base_model.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.base_model.logit_scale.exp() * image_features.float() @ self.base_model.text_features.t()
        return image_features, logits

    def _center_crop_tensor(self, image):
        b, c, h, w = image.shape
        crop_h = max(16, int(h * self.patch_ratio))
        crop_w = max(16, int(w * self.patch_ratio))
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        patch = image[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)

    def _center_crop_tensor_ratio(self, image, ratio):
        b, c, h, w = image.shape
        crop_h = max(16, int(h * ratio))
        crop_w = max(16, int(w * ratio))
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        patch = image[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)

    def _corner_crop_tensor_ratio(self, image, ratio, vertical="top", horizontal="left"):
        b, c, h, w = image.shape
        crop_h = max(16, int(h * ratio))
        crop_w = max(16, int(w * ratio))
        top = 0 if vertical == "top" else max(0, h - crop_h)
        left = 0 if horizontal == "left" else max(0, w - crop_w)
        patch = image[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)

    def _local_views(self, image, ratio):
        return [
            self._center_crop_tensor_ratio(image, ratio),
            self._corner_crop_tensor_ratio(image, ratio, "top", "left"),
            self._corner_crop_tensor_ratio(image, ratio, "top", "right"),
            self._corner_crop_tensor_ratio(image, ratio, "bottom", "left"),
            self._corner_crop_tensor_ratio(image, ratio, "bottom", "right"),
        ]

    def _local_patch_logits(self, image, ratio):
        logits = []
        for view in self._local_views(image.float(), ratio):
            view_input = self._to_imagenet_input(view, size=299)
            _, view_logits = self.branch.forward_logits(view_input)
            logits.append(view_logits.unsqueeze(1))
        return torch.cat(logits, dim=1)

    def _to_imagenet_input(self, image, size=299):
        clip_mean = self.clip_mean.to(device=image.device, dtype=image.dtype)
        clip_std = self.clip_std.to(device=image.device, dtype=image.dtype)
        imnet_mean = self.imnet_mean.to(device=image.device, dtype=image.dtype)
        imnet_std = self.imnet_std.to(device=image.device, dtype=image.dtype)
        image = image * clip_std + clip_mean
        image = image.clamp(0.0, 1.0)
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
        image = (image - imnet_mean) / imnet_std
        return image

    def _topk_mask(self, probs):
        mask = torch.zeros_like(probs)
        if self.candidate_topk > 0:
            topk = min(self.candidate_topk, self.num_classes)
            idx = torch.topk(probs, k=topk, dim=1).indices
            mask = mask.scatter_(1, idx, 1.0)
        return mask

    def _candidate_mask(self, base_logits):
        probs = torch.sigmoid(base_logits)
        mask = (probs >= self.candidate_threshold).float()
        mask = torch.maximum(mask, self._topk_mask(probs))
        return mask

    def _adaptive_candidate_weight(self, base_logits):
        probs = torch.sigmoid(base_logits)
        threshold = torch.sigmoid(self.candidate_threshold_param).unsqueeze(0)
        scale = F.softplus(self.candidate_scale_param).unsqueeze(0)
        weight = torch.sigmoid(scale * (probs - threshold))
        return torch.maximum(weight, self._topk_mask(probs))

    def set_confusion_prior(self, prior):
        self.confusion_prior.copy_(prior.to(self.confusion_prior.device, dtype=self.confusion_prior.dtype))

    def set_fp_prior(self, prior):
        self.fp_prior.copy_(prior.to(self.fp_prior.device, dtype=self.fp_prior.dtype))

    def set_memory_bank(self, features, labels, prototypes, tail_prototypes, tail_mask, subprototypes=None, subproto_mask=None):
        self.memory_bank = features.to(self.memory_bank.device, dtype=self.memory_bank.dtype)
        self.memory_labels = labels.to(self.memory_labels.device, dtype=self.memory_labels.dtype)
        self.prototype_bank = prototypes.to(self.prototype_bank.device, dtype=self.prototype_bank.dtype)
        self.tail_prototype_bank = tail_prototypes.to(self.tail_prototype_bank.device, dtype=self.tail_prototype_bank.dtype)
        self.tail_class_mask = tail_mask.to(self.tail_class_mask.device, dtype=self.tail_class_mask.dtype)
        if subprototypes is not None:
            self.subprototype_bank = subprototypes.to(self.subprototype_bank.device, dtype=self.subprototype_bank.dtype)
        if subproto_mask is not None:
            self.subprototype_mask = subproto_mask.to(self.subprototype_mask.device, dtype=self.subprototype_mask.dtype)

    def _candidate_attention_map(self, featmap, base_logits, candidate_mask):
        cams = torch.einsum("cf,bfhw->bchw", self.branch.head.weight, featmap)
        cams = F.relu(cams)
        cams = cams / cams.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        candidate_weight = candidate_mask * torch.sigmoid(base_logits)
        attn = (cams * candidate_weight.unsqueeze(-1).unsqueeze(-1)).amax(dim=1)
        attn = attn / attn.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return attn

    def _top_candidate_class_mask(self, base_logits, candidate_mask):
        masked_logits = base_logits.masked_fill(candidate_mask <= 0, float("-inf"))
        top_idx = masked_logits.argmax(dim=1, keepdim=True)
        mask = torch.zeros_like(base_logits)
        mask.scatter_(1, top_idx, 1.0)
        return mask

    def _single_candidate_attention_map(self, featmap, base_logits, candidate_mask):
        cams = torch.einsum("cf,bfhw->bchw", self.branch.head.weight, featmap)
        cams = F.relu(cams)
        cams = cams / cams.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        top_candidate_mask = self._top_candidate_class_mask(base_logits, candidate_mask)
        attn = (cams * top_candidate_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        attn = attn / attn.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return attn

    def _topk_coords_from_attention(self, attn, k):
        b, h, w = attn.shape
        flat = attn.view(b, -1)
        k = min(k, flat.size(1))
        idx = torch.topk(flat, k=k, dim=1).indices
        rows = idx // w
        cols = idx % w
        return rows, cols, h, w

    def _crop_from_grid_coords(self, image, rows, cols, grid_h, grid_w, ratio):
        b, c, h, w = image.shape
        crop_h = max(16, int(h * ratio))
        crop_w = max(16, int(w * ratio))
        patches = []
        for i in range(b):
            cy = int((float(rows[i]) + 0.5) * h / grid_h)
            cx = int((float(cols[i]) + 0.5) * w / grid_w)
            top = max(0, min(h - crop_h, cy - crop_h // 2))
            left = max(0, min(w - crop_w, cx - crop_w // 2))
            patch = image[i : i + 1, :, top : top + crop_h, left : left + crop_w]
            patch = F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)
            patches.append(patch)
        return torch.cat(patches, dim=0)

    def _cam_roi_logits(self, image, featmap, base_logits, candidate_mask, ratio, k=1):
        attn = self._candidate_attention_map(featmap, base_logits, candidate_mask)
        rows, cols, grid_h, grid_w = self._topk_coords_from_attention(attn, k)
        roi_logits = []
        for j in range(rows.size(1)):
            patch = self._crop_from_grid_coords(image.float(), rows[:, j], cols[:, j], grid_h, grid_w, ratio)
            patch_input = self._to_imagenet_input(patch, size=299)
            _, patch_logits = self.branch.forward_logits(patch_input)
            roi_logits.append(patch_logits.unsqueeze(1))
        return torch.cat(roi_logits, dim=1), attn

    def _candidate_roi_logits(self, image, featmap, base_logits, candidate_mask, ratio):
        attn = self._single_candidate_attention_map(featmap, base_logits, candidate_mask)
        rows, cols, grid_h, grid_w = self._topk_coords_from_attention(attn, 1)
        patch = self._crop_from_grid_coords(image.float(), rows[:, 0], cols[:, 0], grid_h, grid_w, ratio)
        patch_input = self._to_imagenet_input(patch, size=299)
        _, patch_logits = self.branch.forward_logits(patch_input)
        return patch_logits, attn

    def _knn_retrieval_logits(self, base_features):
        if self.memory_bank.numel() == 0:
            return torch.zeros(base_features.size(0), self.num_classes, device=base_features.device, dtype=base_features.dtype)
        sims = base_features.float() @ self.memory_bank.t()
        topk = min(self.memory_topk, sims.size(1))
        vals, idx = torch.topk(sims, k=topk, dim=1)
        weights = torch.softmax(vals * self.memory_temp, dim=1)
        labels = self.memory_labels[idx]
        probs = torch.sum(weights.unsqueeze(-1) * labels, dim=1).clamp(1e-4, 1 - 1e-4)
        return torch.logit(probs)

    def _prototype_retrieval_logits(self, base_features, use_tail=False):
        proto = self.tail_prototype_bank if use_tail else self.prototype_bank
        logits = base_features.float() @ proto.t()
        if use_tail:
            mask = self.tail_class_mask.unsqueeze(0)
            logits = logits * mask
        return logits

    def _multi_prototype_retrieval_logits(self, base_features):
        if self.subprototype_bank.numel() == 0:
            return torch.zeros(base_features.size(0), self.num_classes, device=base_features.device, dtype=base_features.dtype)
        sims = torch.einsum("bd,ckd->bck", base_features.float(), self.subprototype_bank.float())
        sims = sims.masked_fill(self.subprototype_mask.unsqueeze(0) <= 0, float("-inf"))
        return torch.logsumexp(sims, dim=-1)

    def _candidate_specific_proto_logits(self, base_features, candidate_mask):
        sims = self._prototype_retrieval_logits(base_features, use_tail=False)
        denom = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        masked_mean = (sims * candidate_mask).sum(dim=1, keepdim=True) / denom
        return candidate_mask * (sims - masked_mean)

    def _confusion_aware_proto_logits(self, base_features, candidate_mask):
        expanded = torch.clamp(candidate_mask + candidate_mask @ self.confusion_prior, 0.0, 1.0)
        sims = self._prototype_retrieval_logits(base_features, use_tail=False)
        denom = expanded.sum(dim=1, keepdim=True).clamp_min(1.0)
        masked_mean = (sims * expanded).sum(dim=1, keepdim=True) / denom
        return candidate_mask * (sims - masked_mean)

    def _confusion_proto_topk_logits(self, base_features, candidate_mask):
        sims = self._prototype_retrieval_logits(base_features, use_tail=False)
        expanded = torch.clamp(candidate_mask + candidate_mask @ self.confusion_prior, 0.0, 1.0)
        topk = min(max(self.roi_topk, 1), self.num_classes)
        selected_idx = torch.topk(expanded * sims, k=topk, dim=1).indices
        selected = torch.zeros_like(expanded)
        selected.scatter_(1, selected_idx, 1.0)
        selected = torch.maximum(selected, candidate_mask)
        denom = selected.sum(dim=1, keepdim=True).clamp_min(1.0)
        masked_mean = (sims * selected).sum(dim=1, keepdim=True) / denom
        return candidate_mask * (sims - masked_mean)

    def _confusion_exemplar_topk_logits(self, base_features, candidate_mask):
        if self.memory_bank.numel() == 0:
            return torch.zeros(base_features.size(0), self.num_classes, device=base_features.device, dtype=base_features.dtype)
        sims = base_features.float() @ self.memory_bank.t()
        labels = self.memory_labels > 0.5
        outputs = []
        conf_topk = min(max(self.roi_topk, 1), self.num_classes)
        for c in range(self.num_classes):
            pos_mask = labels[:, c]
            if pos_mask.any():
                pos_vals = sims[:, pos_mask]
                pos_k = min(self.confusion_exemplar_topk, pos_vals.size(1))
                pos_score = torch.topk(pos_vals, k=pos_k, dim=1).values.mean(dim=1)
            else:
                pos_score = torch.zeros(sims.size(0), device=sims.device, dtype=sims.dtype)
            conf_idx = torch.topk(self.confusion_prior[c], k=conf_topk).indices
            conf_mask = labels[:, conf_idx].any(dim=1) & (~labels[:, c])
            if conf_mask.any():
                neg_vals = sims[:, conf_mask]
                neg_k = min(self.confusion_exemplar_topk, neg_vals.size(1))
                neg_score = torch.topk(neg_vals, k=neg_k, dim=1).values.mean(dim=1)
            else:
                neg_score = torch.zeros(sims.size(0), device=sims.device, dtype=sims.dtype)
            outputs.append((pos_score - neg_score).unsqueeze(1))
        logits = torch.cat(outputs, dim=1)
        return logits * candidate_mask

    def _exemplar_retrieval_logits(self, base_features, candidate_mask):
        if self.memory_bank.numel() == 0:
            return torch.zeros(base_features.size(0), self.num_classes, device=base_features.device, dtype=base_features.dtype)
        sims = base_features.float() @ self.memory_bank.t()
        topk = min(self.memory_topk, sims.size(1))
        logits = []
        labels = self.memory_labels > 0.5
        for c in range(self.num_classes):
            pos_mask = labels[:, c]
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_vals = sims[:, pos_mask]
                pos_k = min(topk, pos_vals.size(1))
                pos_score = torch.topk(pos_vals, k=pos_k, dim=1).values.mean(dim=1)
            else:
                pos_score = torch.zeros(sims.size(0), device=sims.device, dtype=sims.dtype)
            if neg_mask.any():
                neg_vals = sims[:, neg_mask]
                neg_k = min(topk, neg_vals.size(1))
                neg_score = torch.topk(neg_vals, k=neg_k, dim=1).values.mean(dim=1)
            else:
                neg_score = torch.zeros(sims.size(0), device=sims.device, dtype=sims.dtype)
            logits.append((pos_score - neg_score).unsqueeze(1))
        logits = torch.cat(logits, dim=1)
        return logits * candidate_mask

    def forward(self, image, inception_input=None):
        output, _ = self.forward_with_aux(image, inception_input=inception_input)
        return output

    def forward_with_aux(self, image, inception_input=None):
        with torch.no_grad():
            base_features, base_logits = self._base_features_logits(image)

        if self.mode == "cnn_logit":
            residual_logits = self.head(self.branch(image.float()))
            return base_logits + self.alpha * residual_logits, {}

        if self.mode == "cnn_feature":
            residual_feature = self.feature_proj(self.branch(image.float()))
            fused = base_features + self.alpha * residual_feature
            fused = fused / fused.norm(dim=-1, keepdim=True)
            return self.base_model.logit_scale.exp() * fused.float() @ self.base_model.text_features.t(), {}

        if self.mode == "patch_logit":
            patch = self._center_crop_tensor(image.float())
            residual_logits = self.head(self.branch(patch))
            return base_logits + self.alpha * residual_logits, {}

        if self.mode == "inception_aux":
            residual_logits = self.head(self.branch(image.float()))
            return base_logits + self.alpha * residual_logits, {}

        if self.mode == "full_inception_logit":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            return base_logits + self.alpha * residual_logits, {}

        if self.mode == "full_inception_feature":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            inc_features = self.branch.forward_features(inc_input)
            residual_feature = self.feature_proj(inc_features)
            fused = base_features + self.alpha * residual_feature
            fused = fused / fused.norm(dim=-1, keepdim=True)
            return self.base_model.logit_scale.exp() * fused.float() @ self.base_model.text_features.t(), {}

        if self.mode == "full_inception_gated":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            inc_features, inc_logits = self.branch.forward_logits(inc_input)
            gate = torch.sigmoid(self.gate(torch.cat([base_features.float(), inc_features.float()], dim=-1)))
            return base_logits + gate * inc_logits, {}

        if self.mode == "full_inception_candidate":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            return base_logits + self.alpha * candidate_mask * residual_logits, {}

        if self.mode == "full_inception_candidate_adaptive":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_weight = self._adaptive_candidate_weight(base_logits)
            return base_logits + self.alpha * candidate_weight * residual_logits, {
                "candidate_weight": candidate_weight,
            }

        if self.mode == "full_inception_candidate_gated":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            inc_features, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            gate = torch.sigmoid(self.gate(torch.cat([base_features.float(), inc_features.float()], dim=-1)))
            return base_logits + self.alpha * candidate_mask * gate * residual_logits, {
                "candidate_mask": candidate_mask,
                "gate": gate,
            }

        if self.mode == "full_inception_candidate_classspec":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            class_gate = torch.sigmoid(self.classspec_temp * torch.sigmoid(base_logits) + self.class_bias.unsqueeze(0))
            class_scale = F.softplus(self.class_alpha).unsqueeze(0)
            output = base_logits + self.alpha * candidate_mask * class_gate * class_scale * residual_logits
            return output, {
                "candidate_mask": candidate_mask,
                "class_gate": class_gate,
            }

        if self.mode == "full_inception_candidate_localpatch":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            patch = self._center_crop_tensor_ratio(image.float(), self.local_patch_ratio)
            patch_input = self._to_imagenet_input(patch, size=299)
            _, local_logits = self.branch.forward_logits(patch_input)
            candidate_mask = self._candidate_mask(base_logits)
            combined = residual_logits + self.local_alpha * local_logits
            output = base_logits + self.alpha * candidate_mask * combined
            return output, {
                "candidate_mask": candidate_mask,
                "local_logits": local_logits,
            }

        if self.mode == "full_inception_candidate_multiscale":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            patch_mid = self._center_crop_tensor_ratio(image.float(), self.local_patch_ratio)
            patch_small = self._center_crop_tensor_ratio(image.float(), max(0.35, self.local_patch_ratio * 0.72))
            patch_mid_input = self._to_imagenet_input(patch_mid, size=299)
            patch_small_input = self._to_imagenet_input(patch_small, size=299)
            _, local_mid_logits = self.branch.forward_logits(patch_mid_input)
            _, local_small_logits = self.branch.forward_logits(patch_small_input)
            candidate_mask = self._candidate_mask(base_logits)
            combined = residual_logits + self.local_alpha * (0.65 * local_mid_logits + 0.35 * local_small_logits)
            output = base_logits + self.alpha * candidate_mask * combined
            return output, {
                "candidate_mask": candidate_mask,
                "local_mid_logits": local_mid_logits,
                "local_small_logits": local_small_logits,
            }

        if self.mode == "full_inception_candidate_topkpatch":
            local_logits_all = self._local_patch_logits(image, self.local_patch_ratio)
            topk_local = torch.topk(local_logits_all, k=min(2, local_logits_all.size(1)), dim=1).values
            pooled_local_logits = topk_local.mean(dim=1)
            candidate_mask = self._candidate_mask(base_logits)
            output = base_logits + self.alpha * candidate_mask * pooled_local_logits
            return output, {
                "candidate_mask": candidate_mask,
                "local_logits_all": local_logits_all,
                "pooled_local_logits": pooled_local_logits,
            }

        if self.mode == "full_inception_candidate_dualbranch":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            local_logits_all = self._local_patch_logits(image, self.local_patch_ratio)
            topk_local = torch.topk(local_logits_all, k=min(2, local_logits_all.size(1)), dim=1).values
            pooled_local_logits = topk_local.mean(dim=1)
            candidate_mask = self._candidate_mask(base_logits)
            combined = residual_logits + self.local_alpha * pooled_local_logits
            output = base_logits + self.alpha * candidate_mask * combined
            return output, {
                "candidate_mask": candidate_mask,
                "local_logits_all": local_logits_all,
                "pooled_local_logits": pooled_local_logits,
            }

        if self.mode == "full_inception_candidate_mivote":
            local_logits_all = self._local_patch_logits(image, self.local_patch_ratio)
            pooled_local_logits = torch.logsumexp(local_logits_all, dim=1) - math.log(float(local_logits_all.size(1)))
            candidate_mask = self._candidate_mask(base_logits)
            output = base_logits + self.alpha * candidate_mask * pooled_local_logits
            return output, {
                "candidate_mask": candidate_mask,
                "local_logits_all": local_logits_all,
                "pooled_local_logits": pooled_local_logits,
            }

        if self.mode == "full_inception_candidate_hardneg":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            output = base_logits + self.alpha * candidate_mask * residual_logits
            return output, {
                "base_logits": base_logits,
                "candidate_mask": candidate_mask,
            }

        if self.mode == "full_inception_candidate_fpfocus":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            output = base_logits + self.alpha * candidate_mask * residual_logits
            return output, {
                "base_logits": base_logits,
                "candidate_mask": candidate_mask,
            }

        if self.mode == "full_inception_candidate_confusion":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            output = base_logits + self.alpha * candidate_mask * residual_logits
            return output, {
                "base_logits": base_logits,
                "candidate_mask": candidate_mask,
            }

        if self.mode == "full_inception_candidate_calibrated":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            fused_logits = base_logits + self.alpha * candidate_mask * residual_logits
            scale = torch.exp(self.log_scale).unsqueeze(0)
            output = fused_logits * scale + self.logit_bias.unsqueeze(0)
            return output, {
                "fused_logits": fused_logits,
            }

        if self.mode == "full_inception_candidate_cascade":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            probs = torch.sigmoid(base_logits)
            uncertain_mask = ((probs >= self.cascade_low_threshold) & (probs <= self.cascade_high_threshold)).float()
            uncertain_mask = torch.maximum(uncertain_mask, self._topk_mask(probs))
            output = base_logits + self.alpha * uncertain_mask * residual_logits
            return output, {
                "uncertain_mask": uncertain_mask,
            }

        if self.mode == "full_inception_candidate_cascadecal":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            probs = torch.sigmoid(base_logits)
            uncertain_mask = ((probs >= self.cascade_low_threshold) & (probs <= self.cascade_high_threshold)).float()
            uncertain_mask = torch.maximum(uncertain_mask, self._topk_mask(probs))
            fused_logits = base_logits + self.alpha * uncertain_mask * residual_logits
            scale = torch.exp(self.log_scale).unsqueeze(0)
            output = fused_logits * scale + self.logit_bias.unsqueeze(0)
            return output, {
                "uncertain_mask": uncertain_mask,
                "fused_logits": fused_logits,
            }

        if self.mode == "full_inception_candidate_distill":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            with torch.no_grad():
                _, teacher_logits = self.teacher.forward_logits(inc_input)
            output = base_logits + self.alpha * candidate_mask * residual_logits
            return output, {
                "student_logits": residual_logits,
                "teacher_logits": teacher_logits,
                "candidate_mask": candidate_mask,
            }

        if self.mode == "full_inception_candidate_camdual":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            featmap = self.branch.forward_feature_map(inc_input)
            residual_logits = self.branch.head(F.adaptive_avg_pool2d(featmap, 1).flatten(1))
            candidate_mask = self._candidate_mask(base_logits)
            roi_logits_all, attn = self._cam_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_medium, k=1)
            roi_logits = roi_logits_all[:, 0]
            combined = residual_logits + self.local_alpha * roi_logits
            output = base_logits + self.alpha * candidate_mask * combined
            return output, {"candidate_mask": candidate_mask, "roi_logits": roi_logits, "attn": attn}

        if self.mode == "full_inception_candidate_cammil4":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            featmap = self.branch.forward_feature_map(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            roi_logits_all, attn = self._cam_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_medium, k=4)
            pooled = torch.logsumexp(roi_logits_all, dim=1) - math.log(float(roi_logits_all.size(1)))
            output = base_logits + self.alpha * candidate_mask * pooled
            return output, {"candidate_mask": candidate_mask, "roi_logits_all": roi_logits_all, "attn": attn}

        if self.mode == "full_inception_candidate_cammil8":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            featmap = self.branch.forward_feature_map(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            roi_logits_all, attn = self._cam_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_small, k=8)
            pooled = torch.logsumexp(roi_logits_all, dim=1) - math.log(float(roi_logits_all.size(1)))
            output = base_logits + self.alpha * candidate_mask * pooled
            return output, {"candidate_mask": candidate_mask, "roi_logits_all": roi_logits_all, "attn": attn}

        if self.mode == "full_inception_candidate_campyramid":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            featmap = self.branch.forward_feature_map(inc_input)
            residual_logits = self.branch.head(F.adaptive_avg_pool2d(featmap, 1).flatten(1))
            candidate_mask = self._candidate_mask(base_logits)
            large_logits, attn = self._cam_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_large, k=1)
            medium_logits, _ = self._cam_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_medium, k=1)
            small_logits, _ = self._cam_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_small, k=1)
            pyramid = 0.5 * large_logits[:, 0] + 0.3 * medium_logits[:, 0] + 0.2 * small_logits[:, 0]
            combined = residual_logits + self.local_alpha * pyramid
            output = base_logits + self.alpha * candidate_mask * combined
            return output, {"candidate_mask": candidate_mask, "pyramid_logits": pyramid, "attn": attn}

        if self.mode == "full_inception_candidate_retrievalknn":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._knn_retrieval_logits(base_features)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * candidate_mask * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_retrievalproto":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._prototype_retrieval_logits(base_features, use_tail=False)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * candidate_mask * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_retrievalhybrid":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            featmap = self.branch.forward_feature_map(inc_input)
            residual_logits = self.branch.head(F.adaptive_avg_pool2d(featmap, 1).flatten(1))
            candidate_mask = self._candidate_mask(base_logits)
            roi_logits_all, attn = self._cam_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_medium, k=1)
            retrieval_logits = self._knn_retrieval_logits(base_features)
            combined = residual_logits + self.local_alpha * roi_logits_all[:, 0]
            output = base_logits + self.alpha * candidate_mask * combined + self.retrieval_alpha * candidate_mask * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits, "attn": attn}

        if self.mode == "full_inception_candidate_tailproto":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._prototype_retrieval_logits(base_features, use_tail=True)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_retrievalproto_cs":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._candidate_specific_proto_logits(base_features, candidate_mask)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_retrievalprotogated":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._prototype_retrieval_logits(base_features, use_tail=False)
            gate = torch.sigmoid(self.retrieval_gate(torch.cat([base_features.float(), retrieval_logits.float()], dim=-1)))
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * candidate_mask * gate * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits, "gate": gate}

        if self.mode == "full_inception_candidate_retrievalconfusion":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._confusion_aware_proto_logits(base_features, candidate_mask)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_retrievalexemplar":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._exemplar_retrieval_logits(base_features, candidate_mask)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_confusionprototopk":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._confusion_proto_topk_logits(base_features, candidate_mask)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_confusionexemplartopk":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._confusion_exemplar_topk_logits(base_features, candidate_mask)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_multiprotoretrieval":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            _, residual_logits = self.branch.forward_logits(inc_input)
            candidate_mask = self._candidate_mask(base_logits)
            retrieval_logits = self._multi_prototype_retrieval_logits(base_features)
            output = base_logits + self.alpha * candidate_mask * residual_logits + self.retrieval_alpha * candidate_mask * retrieval_logits
            return output, {"candidate_mask": candidate_mask, "retrieval_logits": retrieval_logits}

        if self.mode == "full_inception_candidate_candidateroi":
            inc_input = inception_input if inception_input is not None else self._to_imagenet_input(image.float(), size=299)
            featmap = self.branch.forward_feature_map(inc_input)
            residual_logits = self.branch.head(F.adaptive_avg_pool2d(featmap, 1).flatten(1))
            candidate_mask = self._candidate_mask(base_logits)
            roi_logits, attn = self._candidate_roi_logits(image, featmap, base_logits, candidate_mask, self.roi_ratio_medium)
            combined = residual_logits + self.local_alpha * roi_logits
            output = base_logits + self.alpha * candidate_mask * combined
            return output, {"candidate_mask": candidate_mask, "roi_logits": roi_logits, "attn": attn}

        raise RuntimeError(f"Unexpected mode: {self.mode}")


@TRAINER_REGISTRY.register()
class OpenCLIPVPTDeepVisualResidualMultiLabel(OpenCLIPVPTDeepOnlyMultiLabel):
    """Stage-2 visual residual branch on top of a frozen tuned Deep VPT model."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.STAGE2VIS.PREC in ["fp16", "fp32", "amp"]

    def _use_inception_cache(self):
        return str(self.cfg.TRAINER.STAGE2VIS.MODE).lower().startswith("full_inception")

    def _make_loader(self, split, transform, batch_size, shuffle):
        subset = self.df[self.df["split"] == split].reset_index(drop=True)
        max_samples = {
            "train": self.cfg.DATASET.MAX_TRAIN_SAMPLES,
            "val": self.cfg.DATASET.MAX_VAL_SAMPLES,
            "test": self.cfg.DATASET.MAX_TEST_SAMPLES,
        }[split]
        if max_samples is not None and int(max_samples) > 0:
            subset = subset.head(int(max_samples)).reset_index(drop=True)

        inception_cache_dir = ""
        if self._use_inception_cache():
            inception_cache_dir = str(
                Path(
                    os.path.expanduser(
                        getattr(
                            self.cfg.DATASET,
                            "INCEPTION_IMAGE_CACHE_DIR",
                            "/path/to/project/external/CoOp/cache/retinal_inception299_imnet_v3",
                        )
                    )
                ).resolve()
            )

        dataset = RetinalImageDataset(
            subset.to_dict("records"),
            self.num_classes,
            transform,
            image_cache_dir=self.cfg.DATASET.IMAGE_CACHE_DIR,
            image_cache_ext=self.cfg.DATASET.IMAGE_CACHE_EXT,
            inception_cache_dir=inception_cache_dir,
        )
        num_workers = int(self.cfg.DATALOADER.NUM_WORKERS)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def build_model(self):
        cfg = self.cfg
        cfg.defrost()
        cfg.TEST.EVALUATOR = "MultiLabelClassification"
        cfg.freeze()
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

        base_model = OpenCLIPVPTDeepOnlyModel(cfg, self.classnames, clip_model, tokenizer)
        raw_frozen_model_path = str(cfg.TRAINER.VPT.FROZEN_MODEL_PATH).strip()
        frozen_model_path = os.path.abspath(os.path.expanduser(raw_frozen_model_path)) if raw_frozen_model_path else ""
        release_inference = bool(getattr(cfg.TRAINER.STAGE2VIS, "RELEASE_INFERENCE", False))
        if frozen_model_path and os.path.isfile(frozen_model_path):
            checkpoint = torch.load(frozen_model_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            base_model.load_state_dict(state_dict, strict=False)
        elif release_inference:
            print(
                "Frozen Deep VPT checkpoint was not provided. "
                "Continuing because RELEASE_INFERENCE=True and the final SCVPT checkpoint "
                "is expected to contain base_model weights."
            )
        else:
            raise FileNotFoundError(f"Frozen Deep VPT checkpoint not found: {frozen_model_path}")

        print(
            f"Building Stage-2 visual residual model "
            f"(mode={cfg.TRAINER.STAGE2VIS.MODE}) on frozen Deep VPT features"
        )
        self.model = FrozenDeepVPTVisualResidualModel(cfg, base_model)
        self.model.to(self.device)
        mode = cfg.TRAINER.STAGE2VIS.MODE.lower()
        if mode in {
            "full_inception_candidate_confusion",
            "full_inception_candidate_retrievalconfusion",
            "full_inception_candidate_confusionprototopk",
            "full_inception_candidate_confusionexemplartopk",
        }:
            prior = self._compute_confusion_prior(self.model.base_model)
            self.model.set_confusion_prior(prior)
        elif mode == "full_inception_candidate_fpfocus":
            prior = self._compute_fp_prior(self.model.base_model)
            self.model.set_fp_prior(prior)
        if "retrieval" in mode or mode in {
            "full_inception_candidate_confusionprototopk",
            "full_inception_candidate_confusionexemplartopk",
        }:
            memory_loader = self._make_loader("train", preprocess_val, 32, False)
            features, labels, prototypes, tail_prototypes, tail_mask, subprototypes, subproto_mask = self._build_memory_bank(self.model.base_model, memory_loader)
            self.model.set_memory_bank(features, labels, prototypes, tail_prototypes, tail_mask, subprototypes, subproto_mask)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"# trainable params: {sum(p.numel() for p in trainable_params):,}")

        self.optim = build_optimizer(trainable_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("visual_residual", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.STAGE2VIS.PREC == "amp" else None
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def _build_memory_bank(self, base_model, memory_loader):
        feats_all = []
        labels_all = []
        counts = torch.zeros(self.num_classes, device=self.device)
        proto_sum = torch.zeros(self.num_classes, base_model.text_features.shape[-1], device=self.device)
        tail_proto_sum = torch.zeros_like(proto_sum)
        tail_counts = torch.zeros(self.num_classes, device=self.device)
        base_model.eval()
        with torch.no_grad():
            for batch in memory_loader:
                if len(batch) == 4:
                    image, label = batch[0], batch[1]
                else:
                    image, label = batch[0], batch[1]
                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True).float()
                image_features = base_model.image_encoder(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                feats_all.append(image_features.float().cpu())
                labels_all.append(label.float().cpu())
                counts += label.sum(dim=0)
                proto_sum += label.t() @ image_features.float()
        counts_cpu = counts.cpu()
        tail_mask = (counts_cpu <= self.model.memory_tail_threshold).float()
        feats = torch.cat(feats_all, dim=0).to(self.device)
        labels = torch.cat(labels_all, dim=0).to(self.device)
        prototypes = proto_sum / counts.clamp_min(1.0).unsqueeze(1)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tail_proto_sum = labels.t() @ feats
        tail_counts = labels.sum(dim=0)
        tail_prototypes = tail_proto_sum / tail_counts.clamp_min(1.0).unsqueeze(1)
        tail_prototypes = tail_prototypes / tail_prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        subprototypes = torch.zeros(
            self.num_classes, self.model.multi_proto_k, base_model.text_features.shape[-1], device=self.device
        )
        subproto_mask = torch.zeros(self.num_classes, self.model.multi_proto_k, device=self.device)
        for c in range(self.num_classes):
            class_mask = labels[:, c] > 0.5
            if not class_mask.any():
                continue
            class_feats = feats[class_mask]
            num_pos = class_feats.size(0)
            k = min(self.model.multi_proto_k, num_pos)
            if k <= 0:
                continue
            if num_pos <= k:
                selected = class_feats[:k]
                subprototypes[c, :k] = selected / selected.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                subproto_mask[c, :k] = 1.0
                continue
            global_proto = prototypes[c]
            sims = class_feats @ global_proto
            order = torch.argsort(sims)
            anchor_ids = []
            for i in range(k):
                pos = int(round(i * (num_pos - 1) / max(k - 1, 1)))
                anchor_ids.append(order[pos].item())
            anchors = class_feats[torch.tensor(anchor_ids, device=self.device)]
            assign = (class_feats @ anchors.t()).argmax(dim=1)
            for j in range(k):
                cluster = class_feats[assign == j]
                if cluster.numel() == 0:
                    cluster = anchors[j : j + 1]
                proto = cluster.mean(dim=0)
                proto = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                subprototypes[c, j] = proto
                subproto_mask[c, j] = 1.0

        return feats, labels, prototypes, tail_prototypes, tail_mask.to(self.device), subprototypes, subproto_mask

    def _compute_confusion_prior(self, base_model):
        num_classes = self.num_classes
        confusion = torch.zeros(num_classes, num_classes, device=self.device)
        counts = torch.zeros(num_classes, device=self.device)
        topk = int(self.cfg.TRAINER.STAGE2VIS.CONFUSION_TOPK)
        mix = float(self.cfg.TRAINER.STAGE2VIS.CONFUSION_MIX)
        base_model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 4:
                    image, label = batch[0], batch[1]
                else:
                    image, label = batch[0], batch[1]
                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True).float()
                image_features = base_model.image_encoder(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = base_model.logit_scale.exp() * image_features.float() @ base_model.text_features.t()
                probs = torch.sigmoid(logits)
                pos_mask = label > 0.5
                neg_probs = probs * (1.0 - label)
                confusion += pos_mask.float().transpose(0, 1) @ neg_probs
                counts += pos_mask.float().sum(dim=0)
        confusion = confusion / counts.clamp_min(1.0).unsqueeze(1)
        confusion.fill_diagonal_(0.0)
        if topk > 0 and topk < num_classes:
            vals, idx = torch.topk(confusion, k=topk, dim=1)
            sparse = torch.zeros_like(confusion)
            sparse.scatter_(1, idx, vals)
            confusion = sparse
        confusion = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return confusion * mix

    def _compute_fp_prior(self, base_model):
        num_classes = self.num_classes
        fp_sum = torch.zeros(num_classes, device=self.device)
        neg_counts = torch.zeros(num_classes, device=self.device)
        thr = float(self.cfg.TRAINER.STAGE2VIS.HARDNEG_THRESHOLD)
        base_model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 4:
                    image, label = batch[0], batch[1]
                else:
                    image, label = batch[0], batch[1]
                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True).float()
                image_features = base_model.image_encoder(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = base_model.logit_scale.exp() * image_features.float() @ base_model.text_features.t()
                probs = torch.sigmoid(logits)
                neg_mask = (label < 0.5).float()
                fp_mask = (probs >= thr).float() * neg_mask
                fp_sum += fp_mask.sum(dim=0)
                neg_counts += neg_mask.sum(dim=0)
        prior = fp_sum / neg_counts.clamp_min(1.0)
        prior = prior / prior.mean().clamp_min(1e-6)
        return prior.clamp_min(0.25)

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        if isinstance(input, (tuple, list)):
            image, inception_input = input
        else:
            image, inception_input = input, None
        prec = self.cfg.TRAINER.STAGE2VIS.PREC

        if prec == "amp":
            with autocast():
                output, aux = self.model.forward_with_aux(image, inception_input=inception_input)
                loss = self._compute_loss(output, label, aux)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, aux = self.model.forward_with_aux(image, inception_input=inception_input)
            loss = self._compute_loss(output, label, aux)
            self.model_backward_and_update(loss)

        with torch.no_grad():
            probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()
            match = (preds == label).float().mean().item() * 100.0

        loss_summary = {"loss": loss.item(), "label_match": match}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def _compute_loss(self, output, label, aux):
        loss = self.criterion(output, label)
        if self.cfg.TRAINER.STAGE2VIS.MODE.lower() == "full_inception_candidate_hardneg":
            base_probs = torch.sigmoid(aux["base_logits"])
            hard_strength = ((base_probs - self.model.hardneg_threshold) / (1.0 - self.model.hardneg_threshold)).clamp(min=0.0)
            weights = 1.0 + (1.0 - label) * self.model.hardneg_alpha * torch.pow(hard_strength, self.model.hardneg_gamma)
            loss = (loss * weights).mean()
        elif self.cfg.TRAINER.STAGE2VIS.MODE.lower() == "full_inception_candidate_fpfocus":
            base_probs = torch.sigmoid(aux["base_logits"])
            candidate_mask = aux["candidate_mask"]
            hard_strength = ((base_probs - self.model.hardneg_threshold) / (1.0 - self.model.hardneg_threshold)).clamp(min=0.0)
            fp_weight = self.model.fp_prior.unsqueeze(0)
            weights = 1.0 + (1.0 - label) * candidate_mask * fp_weight * self.model.hardneg_alpha * torch.pow(hard_strength, self.model.hardneg_gamma)
            loss = (loss * weights).mean()
        elif self.cfg.TRAINER.STAGE2VIS.MODE.lower() == "full_inception_candidate_confusion":
            base_probs = torch.sigmoid(aux["base_logits"])
            confusion_weight = label @ self.model.confusion_prior
            hard_strength = ((base_probs - self.model.hardneg_threshold) / (1.0 - self.model.hardneg_threshold)).clamp(min=0.0)
            weights = 1.0 + (1.0 - label) * confusion_weight * torch.pow(hard_strength, self.model.hardneg_gamma)
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        if self.cfg.TRAINER.STAGE2VIS.MODE.lower() == "full_inception_candidate_distill":
            temp = float(self.cfg.TRAINER.STAGE2VIS.DISTILL_TEMPERATURE)
            lam = float(self.cfg.TRAINER.STAGE2VIS.DISTILL_LAMBDA)
            student = torch.sigmoid(aux["student_logits"] / temp)
            teacher = torch.sigmoid(aux["teacher_logits"] / temp)
            distill = F.mse_loss(student, teacher)
            loss = loss + lam * distill
        return loss

    def parse_batch_train(self, batch):
        if len(batch) == 4:
            image, label, _, inception_input = batch
            return (
                (
                    image.to(self.device, non_blocking=True),
                    inception_input.to(self.device, non_blocking=True).float(),
                ),
                label.to(self.device, non_blocking=True).float(),
            )
        image, label = super().parse_batch_train(batch)
        return image, label

    def parse_batch_test(self, batch):
        if len(batch) == 4:
            image, label, _, inception_input = batch
            return (
                (
                    image.to(self.device, non_blocking=True),
                    inception_input.to(self.device, non_blocking=True).float(),
                ),
                label.to(self.device, non_blocking=True).float(),
            )
        image, label = super().parse_batch_test(batch)
        return image, label

    def model_inference(self, input, inception_input=None):
        if isinstance(input, (tuple, list)):
            input, inception_input = input
        return self.model(input, inception_input=inception_input)
