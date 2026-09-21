#!/bin/bash
# Fits post-hoc race/sex/view probes on a frozen no_adapter/mlp_adapter checkpoint
# (see training/fit_attribute_probes.py for why this runs after, not during,
# pathology training). Toggle CHECKPOINT_NAME below and run once per checkpoint.
#
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:0
#SBATCH --output=outputs/logs/fit_attribute_probes_%j.log

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

CHECKPOINT_NAME="no_adapter"
# CHECKPOINT_NAME="mlp_adapter"

METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
EMBEDDING_CACHE_DIR="${ADAPTER_EMBEDDING_CACHE_DIR:-./cache/raddino}"
CHECKPOINT_DIR="${ADAPTER_CHECKPOINT_DIR:-./checkpoints}"

python -m training.fit_attribute_probes \
    --checkpoint_dir "$CHECKPOINT_DIR/$CHECKPOINT_NAME" \
    --csv_filepath "$METADATA_CSV" \
    --cache_dir "$EMBEDDING_CACHE_DIR" \
    --dataset_filter=1
