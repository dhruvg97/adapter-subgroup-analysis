#!/bin/bash
# Trains one Attention Pooling layer configuration (8 pathology tasks only).
# Toggle LAYERS/EXP_NAME below and run once per config — 4 runs total. Attribute
# probes are trained afterwards, on the frozen output — see
# train_attn_attribute_probes.sh. "Even Layers" is also the headline Attention
# Pooling config used in Table 1 / Figure 2.
#
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:4
#SBATCH --output=outputs/logs/train_attn_%j.log
#SBATCH --time=1-12:00:00

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

LAYERS="3 6 9 12"; EXP_NAME="attn_even_layers"       # Even Layers  — headline config, Table 1 / Figure 2
# LAYERS="2 3 4 5"; EXP_NAME="attn_early_layers"     # Early Layers
# LAYERS="9 10 11 12"; EXP_NAME="attn_late_layers"   # Late Layers
# LAYERS="2 3 11 12"; EXP_NAME="attn_split_layers"   # Split Layers

IMAGE_DIR="${ADAPTER_IMAGE_DIR:-/path/to/mimic-cxr-jpg}"
METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
CHECKPOINT_DIR="${ADAPTER_CHECKPOINT_DIR:-./checkpoints}"
PATHOLOGY_TASKS="atelectasis consolidation lung_lesion no_finding pleural_effusion pneumonia pneumothorax support_devices"

mkdir -p "$CHECKPOINT_DIR/$EXP_NAME" outputs/logs

# Multi-GPU launch (set --nproc_per_node to your GPU count; use plain
# `python -m training.train ...` instead for a single GPU).
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m training.train \
        --csv_filepath "$METADATA_CSV" \
        --data_dir "$IMAGE_DIR" \
        --save_dir "$CHECKPOINT_DIR/$EXP_NAME" \
        --img_resolution=512 \
        --img_channels=1 \
        --dataset_filter=1 \
        --parents $PATHOLOGY_TASKS \
        --exp_name="$EXP_NAME" \
        --adapter_type=attn \
        --hidden_dim=768 \
        --layer_ids $LAYERS \
        --seed=8 \
        --epochs=50 \
        --bs=32 \
        --lr=1e-4 \
        --lr_warmup=100 \
        --wd=1e-4 \
        --ema_rate=0.99 \
        --eval_freq=400 \
        --dropout=0.0 \
        --dist \
        --num_workers=4 \
        --prefetch_factor=2 \
        2>&1 | tee "$CHECKPOINT_DIR/$EXP_NAME/log.out"
