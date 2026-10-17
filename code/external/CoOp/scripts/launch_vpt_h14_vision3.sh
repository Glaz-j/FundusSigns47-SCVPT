#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/project"
COOP_ROOT="$ROOT/external/CoOp"
OUTDIR="$COOP_ROOT/output/retinal_vision3/OpenCLIPVPTDeepOnlyMultiLabel/vit_h14_ctx16_fulldepth_tuned_cuda0"
LOGFILE="$OUTDIR/launch.log"

mkdir -p "$OUTDIR"
cd "$COOP_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/external/Dassl.pytorch:$ROOT/external/CoOp:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.model_cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/.model_cache"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "[$(date -u +%F' '%T)] starting ViT-H-14 Deep VPT tuned on vision3" >> "$LOGFILE"
echo "[$(date -u +%F' '%T)] cwd=$COOP_ROOT" >> "$LOGFILE"

python "$COOP_ROOT/train.py" \
  --root "$ROOT" \
  --trainer OpenCLIPVPTDeepOnlyMultiLabel \
  --dataset-config-file "$COOP_ROOT/configs/datasets/retinal_vision3.yaml" \
  --config-file "$COOP_ROOT/configs/trainers/OpenCLIPVPTMultiLabel/vit_h14_laion2b_vptdeeponly_imgcache288_ctx16_fulldepth_tuned_vision3.yaml" \
  --output-dir "$OUTDIR" \
  >> "$LOGFILE" 2>&1
