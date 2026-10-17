#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/project
COOP=$ROOT/external/CoOp
DASSL=$ROOT/external/Dassl.pytorch
DATASET_CFG=$COOP/configs/datasets/retinal_vision3.yaml

export PYTHONPATH="$COOP:$DASSL"
export CUDA_VISIBLE_DEVICES=0

run_exp() {
  local cfg="$1"
  local outdir="$2"
  mkdir -p "$outdir"
  python -u "$COOP/train.py" \
    --root "$ROOT" \
    --trainer OpenCLIPVPTDeepVisualResidualMultiLabel \
    --dataset-config-file "$DATASET_CFG" \
    --config-file "$cfg" \
    --output-dir "$outdir" \
    > "$outdir/launch.log" 2>&1
}

run_exp \
  "$COOP/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidateclassspec_20ep_vision3.yaml" \
  "$COOP/output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel/sig2g_fullinception_candidate_classspec_20ep_cuda0"

run_exp \
  "$COOP/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidatecascade_20ep_vision3.yaml" \
  "$COOP/output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel/sig2g_fullinception_candidate_cascade_20ep_cuda0"
