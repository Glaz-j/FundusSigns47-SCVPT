import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import count_num_param, load_checkpoint, load_pretrained_weights

from clip import clip as clip_lib
from datasets.retinal_vision2 import _parse_multilabel, _resolve_impath
from trainers.coop import PromptLearner, TextEncoder, load_clip_to_cpu


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


class CachedPromptModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features.type(self.dtype) @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class CoOpMultiLabelCached(TrainerX):
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
        backbone = self.cfg.MODEL.BACKBONE.NAME.replace('/', '-').replace(' ', '_').lower()
        return manifest_path.parent / f'feature_cache_{backbone}_center_crop.pt'

    def _build_or_load_feature_cache(self, clip_model):
        cache_path = Path(self.cfg.DATASET.FEATURE_CACHE_PATH) if self.cfg.DATASET.FEATURE_CACHE_PATH else self._default_cache_path()
        cache_path = Path(os.path.abspath(os.path.expanduser(str(cache_path))))
        if cache_path.exists() and not self.cfg.DATASET.FEATURE_CACHE_REBUILD:
            print(f'Loading feature cache from {cache_path}')
            return torch.load(cache_path, map_location='cpu')

        print(f'Building feature cache at {cache_path}')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        preprocess = clip_lib._transform(clip_model.visual.input_resolution)
        visual = clip_model.visual.to(self.device)
        visual.eval()
        dtype = clip_model.dtype
        result = {
            'meta': {
                'backbone': self.cfg.MODEL.BACKBONE.NAME,
                'num_classes': self.num_classes,
                'input_resolution': int(clip_model.visual.input_resolution),
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
                    feats = visual(images.type(dtype))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
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
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC in ['fp32', 'amp']:
            clip_model.float()

        cache = self._build_or_load_feature_cache(clip_model)
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

        print('Building cached CoOp model for multi-label classification')
        self.model = CachedPromptModel(cfg, self.classnames, clip_model)
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
