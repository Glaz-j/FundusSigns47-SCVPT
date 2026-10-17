#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/project"
COOP_DIR="$ROOT/external/CoOp"
OUT_ROOT="$COOP_DIR/output/retinal_vision3/OpenCLIPVPTDeepResidualMultiLabelCached"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$COOP_DIR:$ROOT/external/Dassl.pytorch:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.model_cache"
export HUGGINGFACE_HUB_CACHE="$ROOT/.model_cache"
export CUDA_VISIBLE_DEVICES=1

run_exp () {
  local config_path="$1"
  local out_name="$2"
  local out_dir="$OUT_ROOT/$out_name"
  mkdir -p "$out_dir"
  cd "$COOP_DIR"
  python -u train.py \
    --root "$ROOT" \
    --trainer OpenCLIPVPTDeepResidualMultiLabelCached \
    --dataset-config-file "$COOP_DIR/configs/datasets/retinal_vision3.yaml" \
    --config-file "$config_path" \
    --output-dir "$out_dir" | tee "$out_dir/launch.log"
}

run_exp "$COOP_DIR/configs/trainers/OpenCLIPVPTDeepResidualMultiLabelCached/vit_l14_datacomp_vptdeep_stage2_tail_taskres_1000ep_vision3.yaml" "tail_taskres_cuda1"
run_exp "$COOP_DIR/configs/trainers/OpenCLIPVPTDeepResidualMultiLabelCached/vit_l14_datacomp_vptdeep_stage2_tail_labelcorr_1000ep_vision3.yaml" "tail_labelcorr_cuda1"
run_exp "$COOP_DIR/configs/trainers/OpenCLIPVPTDeepResidualMultiLabelCached/vit_l14_datacomp_vptdeep_stage2_headtail_resid_1000ep_vision3.yaml" "headtail_resid_cuda1"
