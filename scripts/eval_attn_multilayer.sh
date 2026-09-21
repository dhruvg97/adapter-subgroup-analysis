#!/bin/bash
# Point-estimate pathology AUROC for all 4 Attention Pooling layer configs, on the
# resampled test set, in one shared-backbone pass. Feeds the notebook's Table 2 /
# Figure 3 bar charts and disparity heatmap.
#
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/eval_attn_multilayer_%j.log

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

IMAGE_DIR="${ADAPTER_IMAGE_DIR:-/path/to/mimic-cxr-jpg}"
METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
CHECKPOINT_DIR="${ADAPTER_CHECKPOINT_DIR:-./checkpoints}"
OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-./outputs}"

python -m evaluation.eval_attn_multilayer \
    --checkpoints attn_early_layers attn_late_layers attn_split_layers attn_even_layers \
    --checkpoint_root "$CHECKPOINT_DIR" \
    --csv_filepath "$METADATA_CSV" \
    --data_dir "$IMAGE_DIR" \
    --dataset_filter=1 \
    --out "$OUTPUT_DIR/eval/attn_multilayer_eval.csv" \
    --figures_dir "$OUTPUT_DIR/figures"
