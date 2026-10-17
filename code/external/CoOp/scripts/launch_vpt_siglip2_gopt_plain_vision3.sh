#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/project"
COOP_DIR="$ROOT/external/CoOp"
OUT_DIR="$COOP_DIR/output/retinal_vision3/OpenCLIPVPTDeepOnlyMultiLabel/vit_gopt16_siglip2_256_ctx16_fulldepth_plain_cuda2"

mkdir -p "$OUT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$COOP_DIR:$ROOT/external/Dassl.pytorch:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.model_cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/.model_cache"
export CUDA_VISIBLE_DEVICES=2

cd "$COOP_DIR"
python -u train.py \
  --root "$ROOT" \
  --trainer OpenCLIPVPTDeepOnlyMultiLabel \
  --dataset-config-file "$COOP_DIR/configs/datasets/retinal_vision3.yaml" \
  --config-file "$COOP_DIR/configs/trainers/OpenCLIPVPTMultiLabel/vit_gopt16_siglip2_256_vptdeeponly_imgcache288_ctx16_fulldepth_vision3.yaml" \
  --output-dir "$OUT_DIR"
