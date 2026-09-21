#!/bin/bash
# Extracts and caches frozen Rad-DINO CLS embeddings for every image in the
# dataset. Required before training/evaluating no_adapter or mlp_adapter (attention
# pooling always uses raw images and never needs this cache).
#
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/extract_cls_%j.log

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

IMAGE_DIR="${ADAPTER_IMAGE_DIR:-/path/to/mimic-cxr-jpg}"
METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
EMBEDDING_CACHE_DIR="${ADAPTER_EMBEDDING_CACHE_DIR:-./cache/raddino}"

python -m training.extract_cls_embeddings \
    --data_dir "$IMAGE_DIR" \
    --csv_filepath "$METADATA_CSV" \
    --output_dir "$EMBEDDING_CACHE_DIR" \
    --splits train,valid,test \
    --bs 64 \
    --num_workers 8
