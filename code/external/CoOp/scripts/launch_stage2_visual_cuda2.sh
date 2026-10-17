#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/project"
COOP_DIR="$ROOT/external/CoOp"
OUT_ROOT="$COOP_DIR/output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$COOP_DIR:$ROOT/external/Dassl.pytorch:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.model_cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/.model_cache"
export CUDA_VISIBLE_DEVICES=2

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
    --output-dir "$out_dir" \
    DATALOADER.NUM_WORKERS 2 | tee "$out_dir/launch.log"
}

run_exp "$COOP_DIR/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_l14_datacomp_vptdeep_stage2vis_patchlogit_30ep_vision3.yaml" "patchlogit_cuda2"
run_exp "$COOP_DIR/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_l14_datacomp_vptdeep_stage2vis_inceptionaux_30ep_vision3.yaml" "inceptionaux_cuda2"
