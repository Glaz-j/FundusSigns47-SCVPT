FundusSigns-47: SCVPT research code
Version 1.0.0 | Research source code | 2026-10-17

Associated manuscript:
FundusSigns-47: Prompt-Tuned Residual Visual Verification for Multi-Label
Ophthalmic Sign Recognition

Authors: Zhaoye Wu, Xiuju Chen, Xiaoyu Xu, Chenxi Huang, Xiaoxin Li
Contact: supermonkeyxi@xmu.edu.cn

SCOPE
This repository provides training, inference, evaluation and analysis code for
FundusSigns-47 and SCVPT, together with experiment configurations and the
47-label mapping. Clinical data, model checkpoints and per-image predictions
are managed separately. The final configuration is identified below; additional
configurations support related experimental variants.

LAYOUT
code/external/CoOp: adapted prompt-tuning trainers, datasets and configurations.
code/external/Dassl.pytorch: vendored framework with its original MIT license.
code/retinal_benchmark: benchmark, inference and analysis scripts.
models: label mapping and configuration files ONLY, no checkpoints.
docs: data format and illustrative manifest, not actual patient data.
VALIDATION.json: source-package validation and preparation record.

ENVIRONMENT
Python 3.10 is the documented original major/minor version. The requirements
below specify the packages needed to configure a working environment.
Use an isolated environment and select a compatible torch/torchvision pair for
your system. CUDA and GPU memory requirements depend on the model and task.

From this archive's root:
  python -m venv .venv
  # Activate the environment using your platform's activation command.
  python -m pip install -r requirements.txt

The local source directories are added by the inference script. For scripts
that import retinal_benchmark as an installed package:
  python -m pip install --no-deps -e code/retinal_benchmark

INFERENCE (requires an independently obtained compatible checkpoint)
Run from the archive root, replacing all /path/to values:
  python code/retinal_benchmark/scripts/scvpt_predict.py --config models/scvpt_release_config.yaml --checkpoint /path/to/scvpt_fundussigns47_inference.pth --label-map models/lesion_label_map.json --manifest /path/to/manifest.csv --output predictions.csv --device cuda:0

The manifest requires image_path; sample_id is optional in the inference CLI.
docs/manifest_example.csv contains placeholders, not runnable example images.
Only load trusted checkpoints. Checkpoint loading may use Python pickle.
Obtain compatible checkpoints separately and follow their applicable terms.

TRAINING AND EVALUATION
Entry point: code/external/CoOp/train.py
Final trainer: OpenCLIPVPTDeepVisualResidualMultiLabel
Final configuration:
code/external/CoOp/configs/trainers/OpenCLIPVPTDeepVisualResidualMultiLabel/vit_gopt16_siglip2_256_vptdeep_stage2vis_fullinceptioncandidate_selectiveperclassthr_20ep_vision3.yaml
Dataset loader: code/external/CoOp/datasets/retinal_vision2.py
Historical training settings: models/original_training_config.yaml

Training requires a compatible private manifest, image data, and the initial
prompt-tuned base/teacher checkpoints referenced by the configuration. Replace
all placeholder paths and inspect dataset-loader fields before running. The
inference release configuration is NOT a substitute for the training config.

ANALYSIS
scvpt_bootstrap_ci.py requires cached per-image targets and logits, which are
not supplied. scvpt_reviewer_fusion_controls.py and run_scvpt_mechanism_analysis.py
also depend on experiment artifacts outside this source-only archive.
Historical scripts contain placeholder paths and must be adapted to the local
environment.

PUBLICATION CHANGES
Server roots and staff names in source annotation filenames were replaced by
placeholder paths/names. Bytecode caches and non-source artifacts were omitted.
Missing upstream license notices were restored from their upstream repositories.
No algorithmic modifications were intended. See VALIDATION.json for changed files.

DATA ACCESS
Clinical images and individual-level annotations are not publicly released due
to patient privacy and institutional data-governance restrictions. De-identified
aggregate statistics and derived evaluation summaries reported in the manuscript
may be requested from the corresponding author, subject to institutional approval.

LICENSE AND ATTRIBUTION
Project-owned code: MIT, see LICENSE. Third-party code retains its own notices;
see THIRD_PARTY_NOTICES.txt. The license does not grant rights to clinical data,
third-party model weights or third-party trademarks.

Research use: this software is not a validated clinical diagnostic device.
Citation metadata is provided in CITATION.cff. No DOI has been assigned yet.
