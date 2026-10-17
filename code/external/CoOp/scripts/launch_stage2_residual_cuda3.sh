#!/usr/bin/env bash
set -u

ROOT="/path/to/project"
COOP_DIR="$ROOT/external/CoOp"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$COOP_DIR:$ROOT/external/Dassl.pytorch:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=3

run_exp() {
  local config_name="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  cd "$COOP_DIR" || exit 1
  python -u train.py \
    --root "$ROOT" \
    --trainer OpenCLIPVPTDeepResidualMultiLabelCached \
    --dataset-config-file "$COOP_DIR/configs/datasets/retinal_vision3.yaml" \
    --config-file "$COOP_DIR/configs/trainers/OpenCLIPVPTDeepResidualMultiLabelCached/$config_name" \
    --output-dir "$out_dir" \
    > "$out_dir/launch.log" 2>&1
}

run_exp "vit_l14_datacomp_vptdeep_stage2_coop_taskres_1000ep_vision3.yaml" \
  "$COOP_DIR/output/retinal_vision3/OpenCLIPVPTDeepResidualMultiLabelCached/coop_taskres_cuda3"

run_exp "vit_l14_datacomp_vptdeep_stage2_bilinear_1000ep_vision3.yaml" \
  "$COOP_DIR/output/retinal_vision3/OpenCLIPVPTDeepResidualMultiLabelCached/bilinear_cuda3"
