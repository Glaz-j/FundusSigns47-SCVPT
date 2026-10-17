#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/project
COOP=$ROOT/external/CoOp
DASSL=$ROOT/external/Dassl.pytorch
DATASET_CFG=$COOP/configs/datasets/retinal_vision3.yaml

export PYTHONPATH="$COOP:$DASSL"
export CUDA_VISIBLE_DEVICES=1

OUTDIR="$COOP/output/retinal_vision3/OpenCLIPVPTDeepVisualResidualMultiLabel/sig2g_fullinception_candidate_localpatch_20ep_cuda1"
mkdir -p "$OUTDIR"
python -u "$COOP/train.py" \
  --root "$ROOT" \
  --trainer OpenCLIPVPTDeepVisualResidualMultiLabel \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$COOP/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidatelocalpatch_20ep_vision3.yaml" \
  --output-dir "$OUTDIR" \
  > "$OUTDIR/launch.log" 2>&1
