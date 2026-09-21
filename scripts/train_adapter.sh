#!/bin/bash
# Trains the "No Adapter" or "MLP" checkpoint (8 pathology tasks only) directly on
# cached CLS embeddings — fast, no backbone forward pass needed since both adapters
# operate on a fixed 768-dim embedding. Toggle ADAPTER_TYPE/EXP_NAME below and run
# once for each of the two adapters. Attribute probes are fit afterwards, on the
# frozen output — see fit_attribute_probes.sh.
#
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/train_adapter_%j.log

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

ADAPTER_TYPE="none"; EXP_NAME="no_adapter"   # "No Adapter" — headline, Table 1
# ADAPTER_TYPE="mlp"; EXP_NAME="mlp_adapter" # "MLP" — headline, Table 1

METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
EMBEDDING_CACHE_DIR="${ADAPTER_EMBEDDING_CACHE_DIR:-./cache/raddino}"
CHECKPOINT_DIR="${ADAPTER_CHECKPOINT_DIR:-./checkpoints}"
PATHOLOGY_TASKS="atelectasis consolidation lung_lesion no_finding pleural_effusion pneumonia pneumothorax support_devices"

mkdir -p "$CHECKPOINT_DIR/$EXP_NAME" outputs/logs

python -m training.train \
    --csv_filepath "$METADATA_CSV" \
    --cache_dir "$EMBEDDING_CACHE_DIR" \
    --save_dir "$CHECKPOINT_DIR/$EXP_NAME" \
    --img_channels=-1 \
    --dataset_filter=1 \
    --parents $PATHOLOGY_TASKS \
    --exp_name="$EXP_NAME" \
    --adapter_type=$ADAPTER_TYPE \
    --hidden_dim=768 \
    --dropout=0.5 \
    --seed=8 \
    --epochs=50 \
    --bs=512 \
    --lr=1e-4 \
    --lr_warmup=100 \
    --wd=1e-4 \
    --ema_rate=0.99 \
    --eval_freq=500 \
    --num_workers=4 \
    2>&1 | tee "$CHECKPOINT_DIR/$EXP_NAME/log.out"
