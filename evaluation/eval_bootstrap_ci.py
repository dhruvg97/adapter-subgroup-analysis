"""Bootstrap 95% CIs for the paper's two summary tables and two disparity heatmaps:

  1. Table 1  — global pathology + attribute AUROC, No Adapter vs MLP vs Attention Pooling (Even Layers)
  2. Figure 2 — subgroup disparity heatmap (with significance), same 3 methods
  3. Table 2  — global pathology + attribute AUROC, the 4 attention-pooling layer configs
  4. Figure 3 — subgroup disparity heatmap (with significance), the 4 layer configs

DESIGN: every model here is frozen, so a checkpoint's per-image score never changes
across bootstrap iterations — only which images/rows get resampled does. So the
expensive part (the Rad-DINO backbone forward pass) runs EXACTLY ONCE per checkpoint,
cached to disk; all `--n_bootstrap` iterations are then a fast, GPU-free loop over
those cached scores.

  - Pathology: reuses evaluation.common.build_joint_sets(test_df, n_factor, seed) with
    a fresh seed per iteration — this IS the bootstrap mechanism for the resampled
    test set (each seed draws a different WeightedRandomSampler cohort). Point
    estimate = seed 42, matching every other point-estimate table in this repo.
  - Attributes: classic row-level bootstrap (resample valid rows with replacement) on
    the standard test set, since there's no resampling procedure to reseed there.

Both mechanisms are "paired" across checkpoints/methods: the same resampled cohort
(pathology) or the same bootstrap row-index draw (attributes) is reused for every
checkpoint within a given iteration, so iteration-i comparisons between methods are
apples-to-apples.

All 8 attention-pooling models (4 pathology-only + 4 attribute probes) share the same
frozen backbone and resolution, so they're scored together in ONE shared backbone
pass. No Adapter / MLP are scored from cached CLS embeddings.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import evaluation.common as ec
from evaluation.eval_attn_multilayer import build_attn_model
from evaluation.eval_attn_probe_attributes import build_probe_model
from training.datasets import CXR1M

# ── Checkpoint registry ──────────────────────────────────────────────────────────
CHECKPOINTS_PATHOLOGY = {
    "No Adapter": "no_adapter",
    "MLP": "mlp_adapter",
    "Early Layers": "attn_early_layers",
    "Late Layers": "attn_late_layers",
    "Split Layers": "attn_split_layers",
    "Even Layers": "attn_even_layers",
}
CHECKPOINTS_ATTRIBUTE = {
    "No Adapter": "no_adapter",
    "MLP": "mlp_adapter",
    "Early Layers": "attn_early_layers_attribute_probe",
    "Late Layers": "attn_late_layers_attribute_probe",
    "Split Layers": "attn_split_layers_attribute_probe",
    "Even Layers": "attn_even_layers_attribute_probe",
}
ATTN_PATHOLOGY_LABELS = {v: k for k, v in CHECKPOINTS_PATHOLOGY.items() if k not in ("No Adapter", "MLP")}
ATTN_PROBE_LABELS = {v: k for k, v in CHECKPOINTS_ATTRIBUTE.items() if k not in ("No Adapter", "MLP")}

METHODS_3WAY = ["No Adapter", "MLP", "Even Layers"]
METHODS_4CONFIG = ["Early Layers", "Late Layers", "Split Layers", "Even Layers"]


# ══════════════════════════════════════════════════════════════════════════════════
# STEP 1 — score every image once per checkpoint, cache to disk
# ══════════════════════════════════════════════════════════════════════════════════
def score_attn_all(pathology_ckpts, probe_ckpts, ckpt_root, test_df, args, device):
    """Shared backbone pass for ALL attention-pooling models at once (4 pathology +
    4 probe = 8 models). Returns (pathology_scores, attribute_scores), each keyed by
    checkpoint folder name -> {task: score_array}."""
    path_models = {}
    for name in pathology_ckpts:
        ckpt = torch.load(Path(ckpt_root) / name / "best_checkpoint.pt", map_location="cpu")
        cargs = ckpt["args"]
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model, layer_ids = build_attn_model(name, cargs, state, device)
        path_models[name] = {"model": model, "layer_ids": layer_ids, "cargs": cargs}

    probe_models = {}
    for name in probe_ckpts:
        model, layer_ids, cargs = build_probe_model(name, ckpt_root)
        model.to(device)
        probe_models[name] = {"model": model, "layer_ids": layer_ids, "cargs": cargs}

    all_models = {**path_models, **probe_models}
    resolutions = {info["cargs"].get("img_resolution", 512) for info in all_models.values()}
    assert len(resolutions) == 1, f"attn checkpoints disagree on img_resolution: {resolutions}"
    res = resolutions.pop()

    eval_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((res, res)), transforms.ToTensor()]
    )
    ds = CXR1M(
        root=args.data_dir, csv_filepath=args.csv_filepath, split="test",
        transform=eval_transform, cache_root=None, dataset_filter=args.dataset_filter,
    )
    assert len(ds.df) == len(test_df), "CXR1M test rows misaligned with test_df"

    n = len(test_df)
    pathology_scores = {name: {t: np.full(n, np.nan, dtype=np.float32) for t in ec.PATHOLOGY_TASKS}
                         for name in path_models}
    attribute_scores = {name: {"race": np.full((n, 3), np.nan, dtype=np.float32),
                                "view": np.full((n, 3), np.nan, dtype=np.float32),
                                "sex": np.full(n, np.nan, dtype=np.float32)}
                         for name in probe_models}

    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    any_model = next(iter(all_models.values()))["model"]
    preprocess_fn = any_model.preprocess
    backbone = any_model.backbone  # frozen + identical across all 8 models — reuse

    row = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"shared backbone ({len(path_models)} pathology + {len(probe_models)} probe models)"):
            x, _ = preprocess_fn(batch["x"], batch["pa"])
            bs_here = x.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden_states = backbone(x, output_hidden_states=True).hidden_states  # computed ONCE

                for name, info in path_models.items():
                    model = info["model"]
                    feats = torch.cat([hidden_states[i] for i in info["layer_ids"]], dim=-1)
                    for task in ec.PATHOLOGY_TASKS:
                        q = model.queries[task].expand(bs_here, -1, -1)
                        out, _ = model.attn(q, feats, feats)
                        s = torch.sigmoid(model.heads[task](out[:, 0])).squeeze(-1).float().cpu().numpy()
                        pathology_scores[name][task][row:row + bs_here] = s

                for name, info in probe_models.items():
                    model = info["model"]
                    feats = torch.cat([hidden_states[i] for i in info["layer_ids"]], dim=-1)
                    tasks = ["race", "sex", "view"]
                    queries = torch.cat([model.queries[t].expand(bs_here, -1, -1) for t in tasks], dim=1)
                    attn_out, _ = model.attn(queries, feats, feats)
                    for i, t in enumerate(tasks):
                        logits = model.heads[t](attn_out[:, i]).float().cpu().numpy()
                        attribute_scores[name][t][row:row + bs_here] = expit(logits[:, 0]) if t == "sex" else softmax(logits, axis=1)
            row += bs_here

    return pathology_scores, attribute_scores


def build_score_cache(args, device):
    print("Loading standard test split...")
    test_df, n_full_test = ec.load_test_df(args.csv_filepath, args.dataset_filter)
    print(f"  test rows (Dataset={args.dataset_filter}): {len(test_df)}  |  full test split: {n_full_test}")

    pathology_scores, attribute_scores = {}, {}

    print("\nScoring No Adapter + MLP from cached embeddings...")
    dat_path = Path(args.cache_dir) / "raddino_emb_float32_test.dat"
    dat = np.memmap(dat_path, dtype="float32", mode="r", shape=(n_full_test, 768))
    X_test = np.array(dat[test_df["_embed_idx"].values], dtype=np.float32)
    for name in (CHECKPOINTS_PATHOLOGY["No Adapter"], CHECKPOINTS_PATHOLOGY["MLP"]):
        method, p_scores, a_scores = ec.score_cached_checkpoint(name, args.checkpoint_root, X_test)
        pathology_scores[name] = p_scores
        attribute_scores[name] = a_scores
        print(f"  scored {name} ({method})")

    print("\nScoring all 8 attention-pooling models in one shared backbone pass...")
    attn_path_ckpts = list(ATTN_PATHOLOGY_LABELS.keys())
    attn_probe_ckpts = list(ATTN_PROBE_LABELS.keys())
    p_scores_attn, a_scores_attn = score_attn_all(attn_path_ckpts, attn_probe_ckpts, args.checkpoint_root, test_df, args, device)
    pathology_scores.update(p_scores_attn)
    for name in attn_probe_ckpts:
        # attribute_scores is keyed by the PROBE checkpoint name (its own entry);
        # pathology_scores is keyed by the pathology checkpoint name — kept separate.
        attribute_scores[name] = a_scores_attn[name]

    return test_df, pathology_scores, attribute_scores


def save_cache(cache_path, test_df_path, test_df, pathology_scores, attribute_scores):
    npz_dict = {}
    for ckpt, tasks in pathology_scores.items():
        for task, arr in tasks.items():
            npz_dict[f"path__{ckpt}__{task}"] = arr
    for ckpt, tasks in attribute_scores.items():
        for task, arr in tasks.items():
            npz_dict[f"attr__{ckpt}__{task}"] = arr
    np.savez_compressed(cache_path, **npz_dict)
    test_df.to_csv(test_df_path, index=False)
    print(f"\nCached scores -> {cache_path}\nCached test_df -> {test_df_path}")


def load_cache(cache_path, test_df_path):
    test_df = pd.read_csv(test_df_path, low_memory=False)
    npz = np.load(cache_path)
    pathology_scores, attribute_scores = {}, {}
    for key in npz.files:
        kind, ckpt, task = key.split("__", 2)
        target = pathology_scores if kind == "path" else attribute_scores
        target.setdefault(ckpt, {})[task] = npz[key]
    return test_df, pathology_scores, attribute_scores


# ══════════════════════════════════════════════════════════════════════════════════
# STEP 2 — bootstrap loops (pure CPU, over cached scores)
# ══════════════════════════════════════════════════════════════════════════════════
def score_binary_from_probs(task, y, s):
    if len(np.unique(y)) < 2:
        return [(task, np.nan, len(y), int((y == 1).sum()))]
    return [(task, round(float(roc_auc_score(y, s)), 4), len(y), int((y == 1).sum()))]


def score_multiclass_from_probs(task, y, ss, class_names):
    n = len(y)
    if len(np.unique(y)) < 2:
        rows = [(task, np.nan, n, np.nan)]
        rows += [(f"{task}_{c}", np.nan, n, np.nan) for c in class_names]
        return rows
    overall = round(float(roc_auc_score(y, ss, multi_class="ovo", average="macro")), 4)
    rows = [(task, overall, n, np.nan)]
    y_oh = np.eye(ss.shape[1])[y]
    per_class = roc_auc_score(y_oh, ss, average=None)
    for k, cname in enumerate(class_names):
        rows.append((f"{task}_{cname}", round(float(per_class[k]), 4), n, int((y == k).sum())))
    return rows


def run_pathology_bootstrap(test_df, pathology_scores, args, seeds, point_seed=42):
    """Returns (point_estimate_df, bootstrap_long_df)."""
    print(f"\nPathology point estimate (seed={point_seed})...")
    joint_sets_pe = ec.build_joint_sets(test_df, args.n_factor, point_seed)
    pe_rows = []
    for label, ckpt in CHECKPOINTS_PATHOLOGY.items():
        pe_rows.extend(ec.pathology_rows(label, ckpt, pathology_scores[ckpt], joint_sets_pe))
    point_estimate_df = pd.DataFrame(pe_rows)

    print(f"Pathology bootstrap ({len(seeds)} iterations)...")
    all_rows = []
    for i, seed in enumerate(tqdm(seeds, desc="pathology bootstrap")):
        joint_sets = ec.build_joint_sets(test_df, args.n_factor, seed)  # same draw reused for every method below
        for label, ckpt in CHECKPOINTS_PATHOLOGY.items():
            rows = ec.pathology_rows(label, ckpt, pathology_scores[ckpt], joint_sets)
            for r in rows:
                r["iteration"], r["seed"] = i, seed
            all_rows.extend(rows)
    bootstrap_df = pd.DataFrame(all_rows)
    return point_estimate_df, bootstrap_df


def run_attribute_bootstrap(test_df, attribute_scores, seeds):
    """Classic row-level bootstrap (resample with replacement), since there's no
    resampling procedure to reseed for the standard test set. Point estimate = the
    original, un-resampled scoring."""
    tasks = ["race", "sex", "view"]
    task_labels = {t: ec.get_attribute_labels(t, test_df) for t in tasks}  # {task: (y, valid_mask)}

    print("\nAttribute point estimate (no resampling)...")
    pe_rows = []
    for label, ckpt in CHECKPOINTS_ATTRIBUTE.items():
        for task in tasks:
            y, valid_mask = task_labels[task]
            probs_valid = attribute_scores[ckpt][task][valid_mask]
            rows = (score_binary_from_probs(task, y, probs_valid) if task == "sex"
                    else score_multiclass_from_probs(task, y, probs_valid, ec.MULTICLASS_CLASS_NAMES[task]))
            for subtask, auroc, n, n_pos in rows:
                pe_rows.append(dict(method=label, checkpoint=ckpt, task=subtask, n=n, n_pos=n_pos, auroc=auroc))
    point_estimate_df = pd.DataFrame(pe_rows)

    print(f"Attribute bootstrap ({len(seeds)} iterations)...")
    all_rows = []
    for task in tasks:
        y, valid_mask = task_labels[task]
        n_valid = len(y)
        class_names = ec.MULTICLASS_CLASS_NAMES.get(task)
        for i, seed in enumerate(tqdm(seeds, desc=f"attribute bootstrap ({task})")):
            idx = np.random.RandomState(seed).choice(n_valid, size=n_valid, replace=True)  # same draw for every method
            y_i = y[idx]
            for label, ckpt in CHECKPOINTS_ATTRIBUTE.items():
                probs_valid = attribute_scores[ckpt][task][valid_mask]
                if task == "sex":
                    rows = score_binary_from_probs(task, y_i, probs_valid[idx])
                else:
                    rows = score_multiclass_from_probs(task, y_i, probs_valid[idx, :], class_names)
                for subtask, auroc, n, n_pos in rows:
                    all_rows.append(dict(method=label, checkpoint=ckpt, task=subtask,
                                          iteration=i, seed=seed, n=n, n_pos=n_pos, auroc=auroc))
    bootstrap_df = pd.DataFrame(all_rows)
    return point_estimate_df, bootstrap_df


# ══════════════════════════════════════════════════════════════════════════════════
# STEP 3 — build the 4 deliverables from the shared long-format tables
# ══════════════════════════════════════════════════════════════════════════════════
def summarize_ci(bootstrap_df, group_cols, value_col="auroc"):
    def _pct(x, q):
        x = x.dropna()
        return np.nan if len(x) == 0 else float(np.percentile(x, q))
    g = bootstrap_df.groupby(group_cols)[value_col]
    return pd.DataFrame({
        "ci_lower": g.apply(lambda x: _pct(x, 2.5)),
        "ci_upper": g.apply(lambda x: _pct(x, 97.5)),
    }).reset_index()


def build_global_table(methods, pe_path, boot_path, pe_attr, boot_attr, label):
    """Table 1 / Table 2: overall pathology AUROC + per-class attribute AUROC, each
    with [point_estimate, ci_lower, ci_upper]."""
    pe_p = pe_path[(pe_path["method"].isin(methods)) & (pe_path["subgroup_type"] == "overall") & (pe_path["subgroup"] == "all")]
    boot_p = boot_path[(boot_path["method"].isin(methods)) & (boot_path["subgroup_type"] == "overall") & (boot_path["subgroup"] == "all")]
    ci_p = summarize_ci(boot_p, ["method", "task"])
    path_table = pe_p[["method", "task", "auroc"]].rename(columns={"auroc": "point_estimate"}).merge(ci_p, on=["method", "task"])
    path_table["category"] = "pathology"

    pe_a = pe_attr[(pe_attr["method"].isin(methods)) & (pe_attr["task"].isin(ec.ATTRIBUTE_ROWS))]
    boot_a = boot_attr[(boot_attr["method"].isin(methods)) & (boot_attr["task"].isin(ec.ATTRIBUTE_ROWS))]
    ci_a = summarize_ci(boot_a, ["method", "task"])
    attr_table = pe_a[["method", "task", "auroc"]].rename(columns={"auroc": "point_estimate"}).merge(ci_a, on=["method", "task"])
    attr_table["category"] = "attribute"

    full = pd.concat([path_table, attr_table], ignore_index=True)
    full = full[["category", "task", "method", "point_estimate", "ci_lower", "ci_upper"]]
    print(f"\n[{label}] global table:")
    print(full.to_string(index=False))
    return full


def build_disparity_table(methods, pe_path, boot_path, label):
    """Figure 2 / Figure 3: per-subgroup disparity — AUROC minus the OVERALL AUROC
    (the subgroup=="all" row: one AUROC computed on the full resampled cohort,
    pooling every subgroup together) for that (method, task) — with a bootstrap CI
    and a significance flag (CI excludes 0)."""
    def _disparity(df):
        merge_cols = ["method", "task"] + (["iteration"] if "iteration" in df.columns else [])
        overall = df[df["subgroup"] == "all"][merge_cols + ["auroc"]].rename(columns={"auroc": "overall_auroc"})
        d = df[df["subgroup"].isin(ec.DEMO_ORDER)].copy()
        d = d.merge(overall, on=merge_cols, how="left")
        missing = d[d["overall_auroc"].isna()][["method", "task"]].drop_duplicates()
        if len(missing):
            print(f"  no overall ('all') AUROC found for {len(missing)} (method, task) pairs — "
                  f"their disparity will be NaN:")
            print(missing.to_string(index=False))
        d["disparity"] = (d["auroc"] - d["overall_auroc"]) * 100
        return d

    pe_d = _disparity(pe_path[pe_path["method"].isin(methods)])
    boot_d = _disparity(boot_path[boot_path["method"].isin(methods)])

    ci = summarize_ci(boot_d, ["method", "task", "subgroup"], value_col="disparity")
    table = pe_d[["method", "task", "subgroup", "disparity"]].rename(columns={"disparity": "point_estimate"})
    table = table.merge(ci, on=["method", "task", "subgroup"])
    table["significant"] = ~((table["ci_lower"] <= 0) & (table["ci_upper"] >= 0))

    print(f"\n[{label}] disparity table (significant cells: {int(table['significant'].sum())} / {len(table)}):")
    print(table.to_string(index=False))
    return table, boot_d


# ══════════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_root", default="./checkpoints")
    p.add_argument("--csv_filepath", default="./data/chai_cxr_master.csv")
    p.add_argument("--data_dir", default="/path/to/mimic-cxr-jpg")
    p.add_argument("--cache_dir", default="./cache/raddino")
    p.add_argument("--dataset_filter", default="1")
    p.add_argument("--n_factor", type=int, default=3)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_bootstrap", type=int, default=100)
    p.add_argument("--point_estimate_seed", type=int, default=42)
    p.add_argument("--out_dir", default="./outputs/eval/bootstrap")
    p.add_argument("--score_cache", default="./outputs/eval/bootstrap/score_cache.npz")
    p.add_argument("--test_df_cache", default="./outputs/eval/bootstrap/test_df_cache.csv")
    p.add_argument("--force_rescore", action="store_true", default=False,
                    help="Ignore any existing score cache and redo the GPU scoring pass.")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.score_cache).parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    cache_exists = Path(args.score_cache).exists() and Path(args.test_df_cache).exists()
    if cache_exists and not args.force_rescore:
        print(f"Found existing score cache at {args.score_cache} — loading (pass --force_rescore to redo it).")
        test_df, pathology_scores, attribute_scores = load_cache(args.score_cache, args.test_df_cache)
    else:
        test_df, pathology_scores, attribute_scores = build_score_cache(args, device)
        save_cache(args.score_cache, args.test_df_cache, test_df, pathology_scores, attribute_scores)
    print(f"\n[GPU scoring stage: {time.time() - t0:.1f}s]")

    seeds = list(range(1, args.n_bootstrap + 1))  # distinct from point_estimate_seed=42

    t1 = time.time()
    pe_path, boot_path = run_pathology_bootstrap(test_df, pathology_scores, args, seeds, args.point_estimate_seed)
    pe_attr, boot_attr = run_attribute_bootstrap(test_df, attribute_scores, seeds)
    print(f"\n[Bootstrap stage: {time.time() - t1:.1f}s]")

    pe_path.to_csv(f"{args.out_dir}/pathology_point_estimate.csv", index=False)
    boot_path.to_csv(f"{args.out_dir}/pathology_bootstrap_full.csv", index=False)
    pe_attr.to_csv(f"{args.out_dir}/attribute_point_estimate.csv", index=False)
    boot_attr.to_csv(f"{args.out_dir}/attribute_bootstrap_full.csv", index=False)

    # ── Table 1: 3-way global table ─────────────────────────────────────────────
    t1_df = build_global_table(METHODS_3WAY, pe_path, boot_path, pe_attr, boot_attr, "Table 1: No Adapter vs MLP vs Attention Pooling")
    t1_df.to_csv(f"{args.out_dir}/1_global_table_3way_summary.csv", index=False)

    # ── Figure 2: 3-way disparity heatmap + significance ────────────────────────
    t2_summary, t2_full = build_disparity_table(METHODS_3WAY, pe_path, boot_path, "Figure 2: No Adapter vs MLP vs Attention Pooling disparity")
    t2_summary.to_csv(f"{args.out_dir}/2_disparity_3way_summary.csv", index=False)
    t2_full.to_csv(f"{args.out_dir}/2_disparity_3way_full.csv", index=False)

    # ── Table 2: 4-layer-config global table ────────────────────────────────────
    t3_df = build_global_table(METHODS_4CONFIG, pe_path, boot_path, pe_attr, boot_attr, "Table 2: 4 attention-pooling layer configs")
    t3_df.to_csv(f"{args.out_dir}/3_global_table_4config_summary.csv", index=False)

    # ── Figure 3: 4-layer-config disparity heatmap + significance ──────────────
    t4_summary, t4_full = build_disparity_table(METHODS_4CONFIG, pe_path, boot_path, "Figure 3: 4 attention-pooling layer configs disparity")
    t4_summary.to_csv(f"{args.out_dir}/4_disparity_4config_summary.csv", index=False)
    t4_full.to_csv(f"{args.out_dir}/4_disparity_4config_full.csv", index=False)

    print(f"\nAll outputs written to {args.out_dir}/")
    print(f"[Total wall time: {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
