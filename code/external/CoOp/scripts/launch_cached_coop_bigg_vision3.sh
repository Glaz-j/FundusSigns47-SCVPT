#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/project"
COOP_ROOT="$ROOT/external/CoOp"
OUTDIR="$COOP_ROOT/output/retinal_vision3/OpenCLIPCoOpMultiLabelCached/vit_bigg14_vision3_1000ep_stable"
LOGFILE="$OUTDIR/launch.log"
PIDFILE="$OUTDIR/run.pid"

mkdir -p "$OUTDIR"
cd "$COOP_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/external/Dassl.pytorch:$ROOT/external/CoOp:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.model_cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/.model_cache"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

echo "[$(date -u +%F' '%T)] starting bigG cached CoOp on vision3" >> "$LOGFILE"
echo "[$(date -u +%F' '%T)] cwd=$COOP_ROOT" >> "$LOGFILE"

python "$COOP_ROOT/train.py" \
  --root "$ROOT" \
  --trainer OpenCLIPCoOpMultiLabelCached \
  --dataset-config-file "$COOP_ROOT/configs/datasets/retinal_vision3.yaml" \
  --config-file "$COOP_ROOT/configs/trainers/OpenCLIPCoOpMultiLabelCached/vit_bigg14_laion2b_cached_vision3_1000ep.yaml" \
  --output-dir "$OUTDIR" \
  >> "$LOGFILE" 2>&1
