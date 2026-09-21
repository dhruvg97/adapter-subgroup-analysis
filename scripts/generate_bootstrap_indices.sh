#!/bin/bash
# Optional: pre-generates patient-resampled bootstrap draws for estimating
# training-run variance (see training/generate_bootstrap_indices.py). NOT required
# to reproduce any figure in the paper — the paper's reported 95% CIs come from
# eval_bootstrap_ci.sh's test-set resampling instead.

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-./outputs}"

python -m training.generate_bootstrap_indices \
    --csv_filepath "$METADATA_CSV" \
    --save_dir "$OUTPUT_DIR/bootstrap_indices" \
    --dataset_filter=1 \
    --n_bootstrap=100 \
    --seed=42
