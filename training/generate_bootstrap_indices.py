"""Pre-generate patient-resampled bootstrap draws for training-variance estimation.

This is NOT what produces the paper's reported 95% confidence intervals — those
come from resampling the frozen TEST set at evaluation time (see
evaluation/eval_bootstrap_ci.py). This script instead resamples TRAIN-split
patients with replacement, for anyone who wants to estimate how much a checkpoint's
performance varies across independent training runs. It is optional and not
required to reproduce any figure in the paper.

Usage:
    python -m training.generate_bootstrap_indices \
        --csv_filepath ./data/chai_cxr_master.csv \
        --save_dir     ./outputs/bootstrap_indices \
        --dataset_filter 1 \
        --n_bootstrap  100 \
        --seed         42

Each output file is a .npy array of patient IDs for one bootstrap draw, passed to
train.py via --patient_ids_file.
"""
import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv_filepath",    type=str, required=True)
parser.add_argument("--save_dir",        type=str, required=True)
parser.add_argument("--dataset_filter",  type=int, default=1)
parser.add_argument("--n_bootstrap",     type=int, default=100)
parser.add_argument("--seed",            type=int, default=42)
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

df = pd.read_csv(args.csv_filepath, low_memory=False)
df = df[(df["Split"] == "train") & (df["Dataset"] == args.dataset_filter)]

unique_pids = df["PatientID"].unique()
print(f"Unique patients in train split (dataset={args.dataset_filter}): {len(unique_pids):,}")

for b in range(args.n_bootstrap):
    sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
    out = os.path.join(args.save_dir, f"bootstrap_{b:04d}.npy")
    np.save(out, sampled)

# Also save the full (non-bootstrapped) patient set, for a baseline/no-resampling run.
np.save(os.path.join(args.save_dir, "full.npy"), unique_pids)
print(f"Saved {args.n_bootstrap} bootstrap index files to {args.save_dir}")
