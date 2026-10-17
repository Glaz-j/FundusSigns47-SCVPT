#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/project"
COOP_ROOT="$ROOT/external/CoOp"
OUTDIR="$COOP_ROOT/output/retinal_vision3/OpenCLIPCoOpMultiLabelCached/siglip2_so400m_vision3_1000ep_stable"
LOGFILE="$OUTDIR/launch.log"

mkdir -p "$OUTDIR"
cd "$COOP_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/external/Dassl.pytorch:$ROOT/external/CoOp:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.model_cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/.model_cache"
export CUDA_VISIBLE_DEVICES=3

echo "[$(date -u +%F' '%T)] starting SigLIP2 SO400M-14 cached CoOp on vision3" >> "$LOGFILE"

python "$COOP_ROOT/train.py" \
  --root "$ROOT" \
  --trainer OpenCLIPCoOpMultiLabelCached \
  --dataset-config-file "$COOP_ROOT/configs/datasets/retinal_vision3.yaml" \
  --config-file "$COOP_ROOT/configs/trainers/OpenCLIPCoOpMultiLabelCached/vit_so400m14_siglip2_cached_vision3_1000ep.yaml" \
  --output-dir "$OUTDIR" \
  >> "$LOGFILE" 2>&1
