#!/bin/bash
# Trains a frozen race/sex/view attribute probe on top of one pathology-trained
# Attention Pooling checkpoint (see training/train_attn_probe.py). Toggle
# PRETRAINED_CHECKPOINT below and run once per layer config — 4 runs total.
#
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/train_attn_probe_%j.log

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

PRETRAINED_CHECKPOINT="attn_even_layers"
# PRETRAINED_CHECKPOINT="attn_early_layers"
# PRETRAINED_CHECKPOINT="attn_late_layers"
# PRETRAINED_CHECKPOINT="attn_split_layers"

IMAGE_DIR="${ADAPTER_IMAGE_DIR:-/path/to/mimic-cxr-jpg}"
METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
CHECKPOINT_DIR="${ADAPTER_CHECKPOINT_DIR:-./checkpoints}"

python -m training.train_attn_probe \
    --pretrained_checkpoint_name "$PRETRAINED_CHECKPOINT" \
    --checkpoint_root "$CHECKPOINT_DIR" \
    --csv_filepath "$METADATA_CSV" \
    --data_dir "$IMAGE_DIR" \
    --dataset_filter=1 \
    --epochs=50 \
    --bs=64 \
    --num_workers=4
