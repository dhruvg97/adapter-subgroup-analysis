"""Evaluate several Attention Pooling checkpoints (different --layer_ids configs) on
the joint Race x View *resampled* MIMIC test set, in a single shared backbone pass.

Rerunning a single-checkpoint eval once per layer config would repeat the most
expensive step (the Rad-DINO backbone forward pass) N times for no reason: the
backbone is always called with output_hidden_states=True and returns every layer
regardless of which layer_ids a given checkpoint actually pools over (see
MultiHeadPredictor.forward_backbone in training/train.py). So here the backbone runs
ONCE per image batch, and each checkpoint's own attn + heads are applied to their own
layer slice on top.

Reports overall + per-subgroup (Race/View/Sex) pathology AUROC for every checkpoint
passed in. Also writes a prevalence audit figure (positive-case counts before vs.
after resampling) alongside the CSV.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluation.common import (
    PATHOLOGY_COL_MAP,
    PATHOLOGY_TASKS,
    RACE_GROUPS,
    SEX_GROUPS,
    VIEW_GROUPS,
    build_joint_sets,
    load_test_df,
    pathology_rows,
)
from training.datasets import CXR1M
from training.train import MultiHeadPredictor

RACE_COLORS = {"Asian": "#2CA02C", "Black": "#9467BD", "White": "#D62728"}
VIEW_COLORS = {"AP": "#17BECF", "PA": "#BCBD22", "Lateral": "#7F7F7F"}
SEX_COLORS = {"Male": "#1F77B4", "Female": "#FF7F0E"}


def build_attn_model(name, cargs, state, device):
    layer_ids = cargs.get("layer_ids")
    assert layer_ids, f"{name}: checkpoint has no saved layer_ids"

    tasks = {k: 1 for k in cargs["parents"]}
    for k, v in dict(dataset=7, race=3, view=3).items():
        if k in tasks:
            tasks[k] = v
    missing_tasks = [t for t in PATHOLOGY_TASKS if t not in tasks]
    assert not missing_tasks, (
        f"{name}: checkpoint's --parents is missing pathology task(s) {missing_tasks}"
    )

    model = MultiHeadPredictor(
        tasks, adapter_type="attn", hidden_dim=cargs["hidden_dim"], head_dim=cargs.get("head_dim", 64),
        layer_ids=layer_ids,
    )
    missing, _ = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("backbone.")]
    assert not bad, f"{name}: attn adapter weights failed to load: {bad}"
    model.to(device).eval()
    print(f"  loaded {name}: layer_ids={layer_ids}")
    return model, layer_ids


def run_shared_attn_inference(models_info, test_df, joint_sets, args, device):
    needed = np.unique(
        np.concatenate([js["_mimic_pos"].values for js in joint_sets.values() if js is not None])
    )
    print(f"  images needed for eval (union across all {len(PATHOLOGY_TASKS)} pathology tasks): {len(needed)}")

    resolutions = {info["cargs"].get("img_resolution", 512) for info in models_info.values()}
    assert len(resolutions) == 1, f"Checkpoints disagree on img_resolution ({resolutions})"
    res = resolutions.pop()

    eval_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((res, res)), transforms.ToTensor()]
    )
    ds = CXR1M(
        root=args.data_dir, csv_filepath=args.csv_filepath, split="test",
        transform=eval_transform, cache_root=None, dataset_filter=str(args.dataset_filter),
    )
    assert len(ds.df) == len(test_df), "CXR1M test rows misaligned with test_df"
    pos_for_row = needed.copy()
    ds.df = ds.df.iloc[needed].reset_index(drop=True)

    all_scores = {
        name: {t: np.full(len(test_df), np.nan, dtype=np.float32) for t in PATHOLOGY_TASKS}
        for name in models_info
    }
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    # The backbone is frozen and identical across every checkpoint — reuse one.
    any_model = next(iter(models_info.values()))["model"]
    preprocess_fn = any_model.preprocess
    backbone = any_model.backbone

    row = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"shared backbone inference ({len(models_info)} layer configs at once)"):
            x, _ = preprocess_fn(batch["x"], batch["pa"])
            bs_here = x.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden_states = backbone(x, output_hidden_states=True).hidden_states  # computed ONCE
                for name, info in models_info.items():
                    model = info["model"]
                    feats = torch.cat([hidden_states[i] for i in info["layer_ids"]], dim=-1)
                    for task in PATHOLOGY_TASKS:
                        q = model.queries[task].expand(bs_here, -1, -1)
                        out, _ = model.attn(q, feats, feats)
                        s = torch.sigmoid(model.heads[task](out[:, 0])).squeeze(-1).float().cpu().numpy()
                        all_scores[name][task][pos_for_row[row:row + bs_here]] = s
            row += bs_here
    return all_scores


def print_checkpoint_table(name, layer_ids, df_ckpt):
    cols = ["all", "Asian", "Black", "White", "AP", "PA", "Lateral", "Male", "Female"]
    print(f"\n{'='*78}\n  {name}  (layers={layer_ids})\n{'='*78}")
    pivot = df_ckpt.pivot_table(index="task", columns="subgroup", values="auroc", aggfunc="first")
    pivot = pivot.reindex(index=PATHOLOGY_TASKS, columns=[c for c in cols if c in pivot.columns])
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))


def plot_resampling_prevalence(test_df, joint_sets, save_path):
    """Positive-case count audit: before vs. after resampling, per pathology."""
    x_indices = np.arange(len(PATHOLOGY_TASKS))
    x_labels = [PATHOLOGY_COL_MAP[k] for k in PATHOLOGY_TASKS]

    race_before = {name: [] for name in RACE_GROUPS.values()}
    race_after = {name: [] for name in RACE_GROUPS.values()}
    view_before = {name: [] for name in VIEW_GROUPS.values()}
    view_after = {name: [] for name in VIEW_GROUPS.values()}
    sex_before = {name: [] for name in SEX_GROUPS.values()}
    sex_after = {name: [] for name in SEX_GROUPS.values()}

    for path_key in PATHOLOGY_TASKS:
        b_df = joint_sets[path_key]
        path_col = PATHOLOGY_COL_MAP[path_key]
        pos_mask_before = (
            (test_df[path_col] == 2)
            & test_df["Race"].isin(RACE_GROUPS.keys())
            & test_df["View"].isin(VIEW_GROUPS.keys())
        )
        df_before_pos = test_df[pos_mask_before]
        df_after_pos = b_df[b_df["_label"] == 1] if b_df is not None else None

        for grp_val, grp_name in RACE_GROUPS.items():
            race_before[grp_name].append(int((df_before_pos["Race"] == grp_val).sum()))
            race_after[grp_name].append(int((df_after_pos["Race"] == grp_val).sum()) if df_after_pos is not None else 0)
        for grp_val, grp_name in VIEW_GROUPS.items():
            view_before[grp_name].append(int((df_before_pos["View"] == grp_val).sum()))
            view_after[grp_name].append(int((df_after_pos["View"] == grp_val).sum()) if df_after_pos is not None else 0)
        for grp_val, grp_name in SEX_GROUPS.items():
            sex_before[grp_name].append(int((df_before_pos["Sex"] == grp_val).sum()))
            sex_after[grp_name].append(int((df_after_pos["Sex"] == grp_val).sum()) if df_after_pos is not None else 0)

    fig, axes = plt.subplots(3, 2, figsize=(24, 24), sharex=True)
    race_offsets = np.linspace(-0.25, 0.25, len(RACE_GROUPS))
    view_offsets = np.linspace(-0.25, 0.25, len(VIEW_GROUPS))
    sex_offsets = np.linspace(-0.15, 0.15, len(SEX_GROUPS))

    def _bar_row(ax_before, ax_after, before_data, after_data, colors, offsets, title_prefix):
        for (grp_name, color), offset in zip(colors.items(), offsets):
            ax_before.bar(x_indices + offset, before_data[grp_name], width=0.22, color=color,
                          label=grp_name, alpha=0.85, zorder=3)
        ax_before.set_title(f"Original Test Set — {title_prefix}", fontsize=24, fontweight="bold", pad=14)
        ax_before.set_ylabel("No. of Positive Images", fontsize=16, fontweight="bold", labelpad=10)
        ax_before.legend(fontsize=14, loc="upper right", framealpha=0.95)
        ax_before.grid(axis="y", linestyle="--", linewidth=1.2, alpha=0.4, zorder=0)

        for (grp_name, color), offset in zip(colors.items(), offsets):
            ax_after.bar(x_indices + offset, after_data[grp_name], width=0.22, color=color,
                         label=grp_name, alpha=0.85, zorder=3)
        ax_after.set_title(f"Rebalanced Test Set — {title_prefix}", fontsize=24, fontweight="bold", pad=14)
        ax_after.legend(fontsize=14, loc="upper right", framealpha=0.95)
        ax_after.grid(axis="y", linestyle="--", linewidth=1.2, alpha=0.4, zorder=0)

    _bar_row(axes[0, 0], axes[0, 1], race_before, race_after, RACE_COLORS, race_offsets, "Positive Case Race Distribution")
    _bar_row(axes[1, 0], axes[1, 1], view_before, view_after, VIEW_COLORS, view_offsets, "Positive Case View Distribution")
    _bar_row(axes[2, 0], axes[2, 1], sex_before, sex_after, SEX_COLORS, sex_offsets, "Positive Case Sex Distribution")

    for ax in (axes[2, 0], axes[2, 1]):
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=22)
    for row in axes:
        for ax in row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(2.5)
            ax.spines["bottom"].set_linewidth(2.5)
            ax.tick_params(axis="both", which="major", labelsize=14, width=2.5, length=6)

    plt.tight_layout(h_pad=4.0, w_pad=2.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  prevalence audit figure saved to {save_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True, help="Folder names under --checkpoint_root")
    p.add_argument("--checkpoint_root", default="./checkpoints")
    p.add_argument("--csv_filepath", default="./data/chai_cxr_master.csv")
    p.add_argument("--data_dir", default="/path/to/mimic-cxr-jpg")
    p.add_argument("--dataset_filter", default="1")
    p.add_argument("--n_factor", type=int, default=3)
    p.add_argument("--resample_seed", type=int, default=42)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default="./outputs/eval/attn_multilayer_eval.csv")
    p.add_argument("--figures_dir", default="./outputs/figures",
                    help="Directory to save the resampling prevalence audit figure.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_root = Path(args.checkpoint_root)

    print("Loading test split + building joint Race x View resampled cohorts (shared across all checkpoints)...")
    test_df, _ = load_test_df(args.csv_filepath, args.dataset_filter)
    joint_sets = build_joint_sets(test_df, args.n_factor, args.resample_seed)
    print(f"  test rows (Dataset={args.dataset_filter}): {len(test_df)}")

    print("\nAuditing resampling prevalence (before vs after, positives only)...")
    plot_resampling_prevalence(test_df, joint_sets, Path(args.figures_dir) / "resampling_prevalence_race_view_sex.png")

    print(f"\nLoading {len(args.checkpoints)} attn checkpoints...")
    models_info = {}
    for name in args.checkpoints:
        ckpt_path = ckpt_root / name / "best_checkpoint.pt"
        if not ckpt_path.exists():
            print(f"skipping {name}: {ckpt_path} not found")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cargs = ckpt["args"]
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        adapter_type = cargs.get("adapter_type", "none")
        assert adapter_type == "attn", f"{name}: expected adapter_type='attn', got '{adapter_type}'"
        model, layer_ids = build_attn_model(name, cargs, state, device)
        models_info[name] = {"model": model, "layer_ids": layer_ids, "cargs": cargs}

    if not models_info:
        print("No valid attn checkpoints loaded — nothing to evaluate.")
        return

    print(f"\nRunning shared-backbone inference for {len(models_info)} layer configs in one pass...")
    all_scores = run_shared_attn_inference(models_info, test_df, joint_sets, args, device)

    all_rows = []
    for name, info in models_info.items():
        rows = pathology_rows("Attention Pooling", name, all_scores[name], joint_sets)
        all_rows.extend(rows)
        print_checkpoint_table(name, info["layer_ids"], pd.DataFrame(rows))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_path, index=False)
    print(f"\nWrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
