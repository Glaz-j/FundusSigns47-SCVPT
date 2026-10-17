import json
import os
from pathlib import Path

import open_clip
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_checkpoint, load_pretrained_weights

from datasets.retinal_vision2 import _parse_multilabel, _resolve_impath
from open_clip.model import text_global_pool


def parse_openclip_backbone(spec: str):
    parts = spec.split(":")
    if len(parts) != 3 or parts[0].lower() != "openclip":
        raise ValueError(
            "OpenCLIP backbone must use format "
            "'openclip:<model_name>:<pretrained_tag>', "
            f"got: {spec}"
        )
    return parts[1], parts[2]


def get_openclip_text_tower(clip_model):
    """Support both classic CLIP-like models and CustomTextCLIP / SigLIP-style models."""
    return getattr(clip_model, "text", clip_model)


class RetinalFeatureBuildDataset(Dataset):
    def __init__(self, rows, num_classes, transform):
        self.rows = rows
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        impath = _resolve_impath(row.get("linked_image_path"), row.get("source_image_path"))
        with Image.open(impath) as img:
            image = self.transform(img)
        label = _parse_multilabel(row.get("lesion_ids"), self.num_classes)
        return image, label, str(row.get("sample_id", ""))


class OpenCLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        text_tower = get_openclip_text_tower(clip_model)
        self.transformer = text_tower.transformer
        self.token_embedding = text_tower.token_embedding
        self.positional_embedding = text_tower.positional_embedding
        self.ln_final = text_tower.ln_final
        self.text_projection = text_tower.text_projection
        self.text_pool_type = getattr(text_tower, "pool_type", getattr(clip_model, "text_pool_type", "argmax"))
        self.text_eos_id = getattr(clip_model, "text_eos_id", None)
        self.attn_mask = getattr(text_tower, "attn_mask", getattr(clip_model, "attn_mask", None))

    def forward(self, prompts, tokenized_prompts):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts + self.positional_embedding.to(cast_dtype)
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device)
        tokenized_prompts = tokenized_prompts.to(x.device)
        x = self.transformer(x, attn_mask=attn_mask)
        x = self.ln_final(x)
        # open_clip changed text_global_pool() signature across releases.
        try:
            x = text_global_pool(
                x,
                tokenized_prompts,
                self.text_pool_type,
                eos_token_id=self.text_eos_id,
            )
        except TypeError:
            x = text_global_pool(x, tokenized_prompts, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return x


class OpenCLIPPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        text_tower = get_openclip_text_tower(clip_model)
        dtype = text_tower.transformer.get_cast_dtype()
        ctx_dim = text_tower.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.image_size if isinstance(clip_model.visual.image_size, int) else clip_model.visual.image_size[0]
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenizer(ctx_init)
            with torch.no_grad():
                embedding = text_tower.token_embedding(prompt).to(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = []
        prompts = []
        for name in classnames:
            prompts.append((prompt_prefix + " " + name + ".").strip())
            try:
                name_lens.append(len(tokenizer.encode(name)))
            except Exception:
                toks = tokenizer(name)
                name_lens.append(int((toks[0] != 0).sum().item()) - 2)

        tokenized_prompts = tokenizer(prompts)
        with torch.no_grad():
            embedding = text_tower.token_embedding(tokenized_prompts).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


class CachedOpenCLIPPromptModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, tokenizer):
        super().__init__()
        self.prompt_learner = OpenCLIPPromptLearner(cfg, classnames, clip_model, tokenizer)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = OpenCLIPTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ text_features.float().t()
        return logits


@TRAINER_REGISTRY.register()
class OpenCLIPCoOpMultiLabelCached(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

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

    def _default_cache_path(self):
        manifest_path = Path(os.path.abspath(os.path.expanduser(self.cfg.DATASET.MANIFEST_PATH)))
        backbone = self.cfg.MODEL.BACKBONE.NAME.replace('/', '-').replace(' ', '_').replace(':', '_').lower()
        return manifest_path.parent / f'feature_cache_{backbone}_center_crop.pt'

    def _load_openclip(self):
        model_name, pretrained_name = parse_openclip_backbone(self.cfg.MODEL.BACKBONE.NAME)
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
            device='cpu',
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess_val, tokenizer

    def _build_or_load_feature_cache(self, model, preprocess):
        cache_path = Path(self.cfg.DATASET.FEATURE_CACHE_PATH) if self.cfg.DATASET.FEATURE_CACHE_PATH else self._default_cache_path()
        cache_path = Path(os.path.abspath(os.path.expanduser(str(cache_path))))
        if cache_path.exists() and not self.cfg.DATASET.FEATURE_CACHE_REBUILD:
            print(f'Loading feature cache from {cache_path}')
            return torch.load(cache_path, map_location='cpu')

        print(f'Building feature cache at {cache_path}')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        model = model.to(self.device)
        model.eval()

        feature_dim = None
        text_proj = getattr(model, "text_projection", None)
        if isinstance(text_proj, torch.Tensor):
            feature_dim = int(text_proj.shape[-1])
        elif isinstance(text_proj, nn.Parameter):
            feature_dim = int(text_proj.shape[-1])
        else:
            feature_dim = 768

        result = {
            'meta': {
                'backbone': self.cfg.MODEL.BACKBONE.NAME,
                'num_classes': self.num_classes,
                'input_resolution': int(self.cfg.INPUT.SIZE[0]),
                'feature_dim': feature_dim,
            },
            'splits': {},
        }

        with torch.no_grad():
            for split in ['train', 'val', 'test']:
                subset = self.df[self.df['split'] == split].reset_index(drop=True)
                max_samples = {
                    'train': self.cfg.DATASET.MAX_TRAIN_SAMPLES,
                    'val': self.cfg.DATASET.MAX_VAL_SAMPLES,
                    'test': self.cfg.DATASET.MAX_TEST_SAMPLES,
                }[split]
                if max_samples is not None and int(max_samples) > 0:
                    subset = subset.head(int(max_samples)).reset_index(drop=True)
                rows = subset.to_dict('records')
                ds = RetinalFeatureBuildDataset(rows, self.num_classes, preprocess)
                loader = DataLoader(
                    ds,
                    batch_size=int(self.cfg.DATASET.FEATURE_CACHE_BATCH_SIZE),
                    shuffle=False,
                    num_workers=int(self.cfg.DATASET.FEATURE_CACHE_NUM_WORKERS),
                    pin_memory=True,
                )
                features = []
                labels = []
                sample_ids = []
                for images, target, ids in loader:
                    images = images.to(self.device, non_blocking=True)
                    feats = model.encode_image(images, normalize=True)
                    features.append(feats.float().cpu())
                    labels.append(target.float().cpu())
                    sample_ids.extend(list(ids))
                result['splits'][split] = {
                    'features': torch.cat(features, dim=0) if features else torch.empty(0, feature_dim),
                    'labels': torch.cat(labels, dim=0) if labels else torch.empty(0, self.num_classes),
                    'sample_ids': sample_ids,
                }
        torch.save(result, cache_path)
        return result

    def build_model(self):
        cfg = self.cfg
        print(f"Loading OpenCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, preprocess, tokenizer = self._load_openclip()

        cache = self._build_or_load_feature_cache(clip_model, preprocess)
        clip_model = clip_model.cpu()
        self.train_loader_x = DataLoader(
            TensorDataset(cache['splits']['train']['features'], cache['splits']['train']['labels']),
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(cache['splits']['val']['features'], cache['splits']['val']['labels']),
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(cache['splits']['test']['features'], cache['splits']['test']['labels']),
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        print('Building cached OpenCLIP CoOp model for multi-label classification')
        self.model = CachedOpenCLIPPromptModel(cfg, self.classnames, clip_model, tokenizer)
        for name, param in self.model.named_parameters():
            if 'prompt_learner' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model.prompt_learner):,}")
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('prompt_learner', self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == 'amp' else None
        self.criterion = nn.BCEWithLogitsLoss()

    def forward_backward(self, batch):
        features, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == 'amp':
            with autocast():
                output = self.model(features)
                loss = self.criterion(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(features)
            loss = self.criterion(output, label)
            self.model_backward_and_update(loss)

        with torch.no_grad():
            probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()
            match = (preds == label).float().mean().item() * 100.0

        loss_summary = {'loss': loss.item(), 'label_match': match}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        features, label = batch
        return features.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True).float()

    def parse_batch_test(self, batch):
        features, label = batch
        return features.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True).float()

    def model_inference(self, input):
        return self.model(input)

    def load_model(self, directory, epoch=None):
        if not directory:
            print('Note that load_model() is skipped as no pretrained model is given')
            return

        names = self.get_model_names()
        model_file = 'model-best.pth.tar'
        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = os.path.join(directory, name, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']
            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
