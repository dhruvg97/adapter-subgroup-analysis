#!/bin/bash
# Produces Table 1, Table 2, Figure 2 and Figure 3's underlying CSVs, with 95%
# bootstrap confidence intervals (100 iterations). Requires all 6 pathology
# checkpoints and all 4 attention-pooling attribute probes to exist first (see
# train_adapter.sh, train_attn.sh, fit_attribute_probes.sh,
# train_attn_attribute_probes.sh). The GPU scoring pass runs once and is cached to
# disk — re-running this script reuses the cache unless --force_rescore is passed.
#
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/eval_bootstrap_ci_%j.log

# ── Edit before running ───────────────────────────────────────────────────────
# Activate your own Python environment first, e.g.: conda activate <your-env>
set -e
cd "$(dirname "$0")/.."

IMAGE_DIR="${ADAPTER_IMAGE_DIR:-/path/to/mimic-cxr-jpg}"
METADATA_CSV="${ADAPTER_METADATA_CSV:-./data/chai_cxr_master.csv}"
EMBEDDING_CACHE_DIR="${ADAPTER_EMBEDDING_CACHE_DIR:-./cache/raddino}"
CHECKPOINT_DIR="${ADAPTER_CHECKPOINT_DIR:-./checkpoints}"
OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-./outputs}"

python -m evaluation.eval_bootstrap_ci \
    --checkpoint_root "$CHECKPOINT_DIR" \
    --csv_filepath "$METADATA_CSV" \
    --data_dir "$IMAGE_DIR" \
    --cache_dir "$EMBEDDING_CACHE_DIR" \
    --dataset_filter=1 \
    --n_bootstrap=100 \
    --out_dir "$OUTPUT_DIR/eval/bootstrap" \
    --score_cache "$OUTPUT_DIR/eval/bootstrap/score_cache.npz" \
    --test_df_cache "$OUTPUT_DIR/eval/bootstrap/test_df_cache.csv"
