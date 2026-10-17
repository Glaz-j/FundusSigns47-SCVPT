#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/project"
COOP_DIR="$ROOT/external/CoOp"
OUT_ROOT="$COOP_DIR/output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel"
MANIFEST="$ROOT/retinal_benchmark/data_versions/vision3/manifest.csv"
IMG_CACHE="$COOP_DIR/cache/retinal_image_cache_288_v3"
INC_CACHE="$COOP_DIR/cache/retinal_inception299_imnet_v3"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$COOP_DIR:$ROOT/external/Dassl.pytorch:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.model_cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/.model_cache"
export CUDA_VISIBLE_DEVICES=2

cd "$COOP_DIR"
python -u scripts/build_inception_image_cache.py \
  --manifest "$MANIFEST" \
  --image-cache-dir "$IMG_CACHE" \
  --output-dir "$INC_CACHE"

run_exp () {
  local config_path="$1"
  local out_name="$2"
  local out_dir="$OUT_ROOT/$out_name"
  mkdir -p "$out_dir"
  cd "$COOP_DIR"
  python -u train.py \
    --root "$ROOT" \
    --trainer OpenCLIPVPTDeepVisualResidualMultiLabel \
    --dataset-config-file "$COOP_DIR/configs/datasets/retinal_vision3.yaml" \
    --config-file "$config_path" \
    --output-dir "$out_dir" | tee "$out_dir/launch.log"
}

run_exp "$COOP_DIR/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_l14_datacomp_vptdeep_stage2vis_fullinceptionfeature_20ep_vision3.yaml" "fullinception_feature_cuda2"
