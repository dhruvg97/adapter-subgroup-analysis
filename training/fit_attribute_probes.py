"""Post-hoc attribute (race/sex/view) probing on a frozen, pathology-only-trained
`none`/`mlp` checkpoint.

The paper trains pathology heads first (8 tasks only), then fits a fresh linear
probe for race/sex/view on top of the frozen adapter output — this is that probe
step, for the two adapter types whose output is a fixed embedding (`none`: the raw
CLS token; `mlp`: the MLP adapter's output). Attention pooling has no fixed
embedding to probe this way (the pooling itself is input-dependent), so its
attribute probe is a separate training run instead — see train_attn_probe.py.

Saves probe weights directly into best_weights/, using the same (C, in_dim) shape
and naming convention (race.npy / sex.npy / view.npy) as train.py's own head
weights, so evaluation/common.py needs no special case to load them.

IMPORTANT — fit on TRAIN, evaluate on TEST: these probes are fit on the TRAIN
split's cached embeddings and scored separately (by evaluation code) on the TEST
split. Fitting on the test split itself would let the probe memorize the exact
images it's later evaluated on, inflating AUROC with data leakage.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# race/view: 3-class, target = col - 1 (col encodes 0=NaN, 1..3=classes).
# sex: binary, positive = Female (Sex==2) — matches train.py's own pos_weight
# convention for the "sex" task.
DIRECT_TASKS = ["race", "sex", "view"]


def load_adapter(checkpoint_dir, device="cpu"):
    ckpt = torch.load(f"{checkpoint_dir}/best_checkpoint.pt", map_location=device)
    ckpt_args = ckpt["args"]
    adapter_type = ckpt_args.get("adapter_type", "none")

    if adapter_type == "none":
        return None, adapter_type
    if adapter_type != "mlp":
        raise ValueError(
            f"fit_attribute_probes.py only supports adapter_type in ('none', 'mlp'), got "
            f"{adapter_type!r} — attention pooling uses train_attn_probe.py instead."
        )

    backbone_dim = 768
    state = ckpt["model_state_dict"]

    def _strip(key):
        for prefix in ("module.", "_orig_mod."):
            while key.startswith(prefix):
                key = key[len(prefix):]
        return key

    clean_state = {_strip(k): v for k, v in state.items()}
    adapter_keys = {k[len("adapter."):]: v for k, v in clean_state.items() if k.startswith("adapter.")}
    if not adapter_keys:
        raise RuntimeError(f"No adapter.* keys found. Prefixes present: "
                           f"{sorted(set(k.split('.')[0] for k in clean_state))}")

    adapter = nn.Sequential(
        nn.LayerNorm(backbone_dim),
        nn.Linear(backbone_dim, ckpt_args["hidden_dim"]),
        nn.GELU(),
        nn.Dropout(ckpt_args["dropout"]),
    )
    adapter.load_state_dict(adapter_keys)
    adapter.eval()
    return adapter, adapter_type


def load_split(csv_filepath, cache_dir, split, dataset_filter, emb_dim=768):
    df = pd.read_csv(csv_filepath, low_memory=False)
    split_df = df[df["Split"] == split].copy().reset_index(drop=True)
    split_df["_embed_idx"] = np.arange(len(split_df))
    n_split = len(split_df)
    split_df = split_df[split_df["Dataset"] == dataset_filter].copy().reset_index(drop=True)

    mm = np.memmap(f"{cache_dir}/raddino_emb_float32_{split}.dat",
                   dtype="float32", mode="r", shape=(n_split, emb_dim))
    X = np.array(mm[split_df["_embed_idx"].values], dtype=np.float32)
    return split_df, X


def fit_direct_probe(X_fit, train_df, task):
    """race/view -> 3-class multinomial (sklearn auto-selects multinomial for
    lbfgs + >2 classes, giving coef_ shape (3, in_dim)). sex -> binary (coef_
    shape (1, in_dim), same as every other binary head)."""
    if task == "race":
        valid = (train_df["Race"].values != 0)
        y = (train_df.loc[valid, "Race"].values - 1).astype(int)
    elif task == "sex":
        valid = (train_df["Sex"].values != 0)
        y = (train_df.loc[valid, "Sex"].values == 2).astype(int)  # positive = Female
    elif task == "view":
        valid = (train_df["View"].values != 0)
        y = (train_df.loc[valid, "View"].values - 1).astype(int)
    else:
        raise ValueError(f"Unknown direct task: {task}")

    clf = LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs").fit(X_fit[valid], y)
    pos_rate = float(y.mean()) if task == "sex" else float("nan")
    return clf.coef_, int(valid.sum()), pos_rate


def main(args):
    train_df, X_train = load_split(args.csv_filepath, args.cache_dir,
                                    "train", args.dataset_filter)
    adapter, adapter_type = load_adapter(args.checkpoint_dir)
    print(f"adapter_type: {adapter_type}")

    if adapter is not None:
        with torch.no_grad():
            X_fit = adapter(torch.tensor(X_train, dtype=torch.float32)).numpy()
    else:
        X_fit = X_train

    weights_dir = os.path.join(args.checkpoint_dir, "best_weights")
    os.makedirs(weights_dir, exist_ok=True)  # should already exist from training

    print("\nAttribute probes (race / sex / view):")
    for task in DIRECT_TASKS:
        coef, n, pos_rate = fit_direct_probe(X_fit, train_df, task)
        np.save(os.path.join(weights_dir, f"{task}.npy"), coef)
        pos_str = f"pos_rate={pos_rate:.4f}" if not np.isnan(pos_rate) else "(multi-class, no single pos_rate)"
        print(f"  {task:<6} n={n:>7}  shape={coef.shape}  {pos_str}")

    print(f"\nAttribute probes saved into {weights_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--csv_filepath",   type=str, required=True)
    parser.add_argument("--cache_dir",      type=str, required=True)
    parser.add_argument("--dataset_filter", type=int, default=1)
    args = parser.parse_args()
    main(args)
