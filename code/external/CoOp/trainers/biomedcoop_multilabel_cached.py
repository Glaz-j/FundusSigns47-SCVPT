import json
import os
from pathlib import Path

import open_clip
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_checkpoint, load_pretrained_weights

from datasets.retinal_vision2 import _parse_multilabel, _resolve_impath


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


class BiomedPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, model, tokenizer):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        ctx_dim = model.text.transformer.config.hidden_size
        clip_imsize = model.visual.image_size if isinstance(model.visual.image_size, int) else model.visual.image_size[0]
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        word_embeddings = model.text.transformer.embeddings.word_embeddings

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ").strip()
            ctx_tokens = tokenizer([ctx_init])
            valid = (ctx_tokens[0] != model.text.config.pad_token_id).nonzero(as_tuple=False).flatten()
            # drop special tokens at the front/back when present
            usable = ctx_tokens[0, 1: valid[-1]] if len(valid) > 1 else ctx_tokens[0, 1:1]
            with torch.no_grad():
                init_embed = word_embeddings(usable)
            n_ctx = init_embed.shape[0]
            ctx_vectors = init_embed.clone().detach()
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["x"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.register_buffer("token_embedding_weight", word_embeddings.weight.detach().clone())
        self.pad_token_id = model.text.config.pad_token_id
        self.classnames = [name.replace("_", " ") for name in classnames]
        prompts = [f"{prompt_prefix} {name}".strip() for name in self.classnames]
        tokenized_prompts = tokenizer(prompts)
        attention_mask = (tokenized_prompts != self.pad_token_id).long()
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("attention_mask", attention_mask)

    def forward(self):
        tokenized = self.tokenized_prompts
        embeddings = F.embedding(tokenized, self.token_embedding_weight)
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        embeddings[:, 1:1 + self.n_ctx, :] = ctx
        return embeddings, self.attention_mask


class BiomedTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.text.transformer
        self.pooler = model.text.pooler
        self.proj = model.text.proj

    def forward(self, input_embeds, attention_mask):
        out = self.transformer(inputs_embeds=input_embeds, attention_mask=attention_mask)
        pooled = self.pooler(out, attention_mask)
        projected = self.proj(pooled)
        return projected


class CachedBiomedPromptModel(nn.Module):
    def __init__(self, cfg, classnames, model, tokenizer):
        super().__init__()
        self.prompt_learner = BiomedPromptLearner(cfg, classnames, model, tokenizer)
        self.text_encoder = BiomedTextEncoder(model)
        self.logit_scale = model.logit_scale

    def forward(self, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompt_embeds, attention_mask = self.prompt_learner()
        text_features = self.text_encoder(prompt_embeds, attention_mask)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.float() @ text_features.float().t()
        return logits


@TRAINER_REGISTRY.register()
class BiomedCoOpMultiLabelCached(TrainerX):
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

    def _load_biomedclip(self):
        model_name = self.cfg.MODEL.BACKBONE.NAME
        model, preprocess = open_clip.create_model_from_pretrained(model_name, device='cpu')
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess, tokenizer

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
        result = {
            'meta': {
                'backbone': self.cfg.MODEL.BACKBONE.NAME,
                'num_classes': self.num_classes,
                'input_resolution': int(self.cfg.INPUT.SIZE[0]),
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
                    'features': torch.cat(features, dim=0) if features else torch.empty(0, 512),
                    'labels': torch.cat(labels, dim=0) if labels else torch.empty(0, self.num_classes),
                    'sample_ids': sample_ids,
                }
        torch.save(result, cache_path)
        return result

    def build_model(self):
        cfg = self.cfg
        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomed_model, preprocess, tokenizer = self._load_biomedclip()

        cache = self._build_or_load_feature_cache(biomed_model, preprocess)
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

        print('Building cached BiomedCoOp model for multi-label classification')
        self.model = CachedBiomedPromptModel(cfg, self.classnames, biomed_model, tokenizer)
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
            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
