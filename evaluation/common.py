"""Shared evaluation library: label conventions, the prevalence-preserving test-set
resampling procedure, and cached-embedding scoring for the `no_adapter`/`mlp_adapter`
checkpoints.

Used directly by the results notebook (to score `no_adapter`/`mlp_adapter` inline,
without a separate CLI script — scoring from cached embeddings is cheap) and by
evaluation/eval_bootstrap_ci.py (for the same scoring, repeated across 100 bootstrap
iterations). evaluation/eval_attn_multilayer.py and
evaluation/eval_attn_probe_attributes.py import the constants and resampling
functions from here too, so there is exactly one copy of this logic in the repo.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score

# ── Pathology tasks (the paper's 8) ─────────────────────────────────────────────────
PATHOLOGY_COL_MAP = {
    "atelectasis": "Atelectasis",
    "consolidation": "Consolidation",
    "lung_lesion": "Lung Lesion",
    "no_finding": "No Finding",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "support_devices": "Support Devices",
}
PATHOLOGY_TASKS = list(PATHOLOGY_COL_MAP)
PATH_SHORT = {
    "atelectasis": "Atel.", "consolidation": "Consol.", "lung_lesion": "LungLes.",
    "no_finding": "NoFind.", "pleural_effusion": "PlEff.", "pneumonia": "Pneumo.",
    "pneumothorax": "Pneumotx.", "support_devices": "SuppDev.",
}

# ── Subgroups ────────────────────────────────────────────────────────────────────────
RACE_GROUPS = {1: "Asian", 2: "Black", 3: "White"}
VIEW_GROUPS = {1: "AP", 2: "PA", 3: "Lateral"}
SEX_GROUPS = {1: "Male", 2: "Female"}
DEMO_ORDER = ["White", "Asian", "Black", "Female", "Male", "AP", "PA", "Lateral"]

# ── Attributes ───────────────────────────────────────────────────────────────────────
BASE_TASKS = ["race", "sex", "view"]  # 3-class / 3-class / binary
MULTICLASS_CLASS_NAMES = {"race": ["Asian", "Black", "White"], "view": ["AP", "PA", "Lateral"]}
ATTRIBUTE_ROWS = ["race_Asian", "race_Black", "race_White", "sex", "view_AP", "view_PA", "view_Lateral"]

METHOD_LABELS = {"none": "No Adapter", "mlp": "MLP", "attn": "Attention Pooling"}


# ══════════════════════════════════════════════════════════════════════════════════
# Test split + prevalence-preserving Race x View resampling
# ══════════════════════════════════════════════════════════════════════════════════
def load_test_df(csv_filepath, dataset_filter):
    """Returns (test_df, n_full_test). n_full_test is the split size BEFORE dataset
    filtering — that's what the CLS embedding memmap is shaped against, and
    test_df["_embed_idx"] indexes into it."""
    df = pd.read_csv(csv_filepath, low_memory=False)
    test_df = df[df["Split"] == "test"].copy().reset_index(drop=True)
    n_full_test = len(test_df)
    test_df["_embed_idx"] = np.arange(n_full_test)
    test_df = test_df[test_df["Dataset"] == int(dataset_filter)].copy().reset_index(drop=True)
    test_df["_mimic_pos"] = np.arange(len(test_df))
    return test_df, n_full_test


def resample_for_pathology_joint(df, pathology_col, n_factor, seed):
    """Glocker-style two-stage resampling: (1) equalise Race x View intersectional
    cell ratios, (2) recalibrate disease prevalence within each cell back to the
    dataset's overall rate. Label convention: positive iff the pathology column is
    explicitly 2; every other value (unmentioned, explicit negative, uncertain)
    counts as negative — this is the "NaN-as-negative" convention every checkpoint
    was trained under (see training/train.py's NAN_AS_NEGATIVE)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    valid_mask = df["Race"].isin(RACE_GROUPS) & df["View"].isin(VIEW_GROUPS)
    df_v = df[valid_mask].copy().reset_index(drop=True)
    df_v["_label"] = (df_v[pathology_col] == 2).astype(int)
    if len(df_v) == 0:
        return None

    df_v["_group"] = list(zip(df_v["Race"], df_v["View"]))

    # Step 1: equalise intersectional Race x View cell ratios.
    n_samples = int(len(df_v) * n_factor)
    w_group = (1 / df_v["_group"].value_counts(normalize=True)).to_dict()
    group_w = df_v["_group"].apply(lambda g: w_group.get(g, 0.0)).values.astype(float)
    ids = list(torch.utils.data.WeightedRandomSampler(group_w, n_samples, replacement=True))
    df_joint = df_v.iloc[ids].copy().reset_index(drop=True)

    # Step 2: recalibrate disease prevalence within each cell.
    overall_prev = df_joint["_label"].value_counts(normalize=True).to_dict()
    balanced_groups = []
    for g in df_joint["_group"].unique():
        grp = df_joint[df_joint["_group"] == g].copy().reset_index(drop=True)
        obs_prev = grp["_label"].value_counts(normalize=True).to_dict()
        w_disease = {k: overall_prev[k] / obs_prev[k] for k in obs_prev}
        disease_w = grp["_label"].apply(lambda x: w_disease.get(x, 0.0)).values.astype(float)
        ids = list(torch.utils.data.WeightedRandomSampler(disease_w, len(grp), replacement=True))
        balanced_groups.append(grp.iloc[ids])

    return pd.concat(balanced_groups).reset_index(drop=True)


def build_joint_sets(test_df, n_factor, seed):
    return {
        task: resample_for_pathology_joint(test_df, PATHOLOGY_COL_MAP[task], n_factor, seed)
        for task in PATHOLOGY_TASKS
    }


# ══════════════════════════════════════════════════════════════════════════════════
# Attribute ground-truth labels (standard, un-resampled test set)
# ══════════════════════════════════════════════════════════════════════════════════
def get_attribute_labels(task, df):
    """Encoding: Race 0=NaN,1=Asian,2=Black,3=White | Sex 0=NaN,1=Male,2=Female |
    View 0=NaN,1=AP,2=PA,3=Lateral. Multiclass target = col - 1. Sex positive class
    = Female (Sex==2), matching train.py's own pos_weight convention."""
    if task == "race":
        valid = (df["Race"] != 0).values
        return (df.loc[valid, "Race"] - 1).values.astype(int), valid
    if task == "sex":
        valid = (df["Sex"] != 0).values
        return (df.loc[valid, "Sex"] == 2).astype(int).values, valid
    if task == "view":
        valid = (df["View"] != 0).values
        return (df.loc[valid, "View"] - 1).values.astype(int), valid
    raise ValueError(f"Unknown attribute task: {task}")


# ══════════════════════════════════════════════════════════════════════════════════
# Cached-embedding scoring: no_adapter / mlp_adapter
# ══════════════════════════════════════════════════════════════════════════════════
def build_cached_adapter(adapter_type, cargs, state):
    """Reconstruct the trained adapter (identity for 'none') from checkpoint keys."""
    if adapter_type == "none":
        return nn.Identity()
    if adapter_type != "mlp":
        raise ValueError(f"{adapter_type!r} is not a cached-embedding adapter type — use the attn eval scripts instead.")
    hidden = cargs.get("hidden_dim", 768)
    adapter = nn.Sequential(
        nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU(), nn.Dropout(cargs.get("dropout", 0.0))
    )
    keys = {k[len("adapter."):]: v for k, v in state.items() if k.startswith("adapter.")}
    adapter.load_state_dict(keys)
    return adapter.eval()


def score_cached_checkpoint(checkpoint_name, checkpoint_root, X_test):
    """Score a `no_adapter`/`mlp_adapter` checkpoint's pathology + attribute heads
    from cached CLS embeddings (X_test, aligned to test_df row order).

    Returns (method_label, pathology_scores, attribute_scores):
      pathology_scores[task]  -> (n,) sigmoid probability array
      attribute_scores["sex"] -> (n,) sigmoid probability array
      attribute_scores[race/view] -> (n, 3) softmax probability array
    """
    ckpt_path = Path(checkpoint_root) / checkpoint_name / "best_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cargs = ckpt["args"]
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    adapter_type = cargs.get("adapter_type", "none")
    method = METHOD_LABELS.get(adapter_type, adapter_type)

    adapter = build_cached_adapter(adapter_type, cargs, state)
    with torch.no_grad():
        Xp = adapter(torch.tensor(X_test, dtype=torch.float32)).numpy()
    weights_dir = Path(checkpoint_root) / checkpoint_name / "best_weights"

    pathology_scores = {}
    for task in PATHOLOGY_TASKS:
        w = np.load(weights_dir / f"{task}.npy")
        pathology_scores[task] = expit((Xp @ w.T)[:, 0])

    attribute_scores = {}
    for task in BASE_TASKS:
        w = np.load(weights_dir / f"{task}.npy")
        logits = Xp @ w.T
        attribute_scores[task] = expit(logits[:, 0]) if w.shape[0] == 1 else softmax(logits, axis=1)

    return method, pathology_scores, attribute_scores


# ══════════════════════════════════════════════════════════════════════════════════
# Score arrays -> tidy per-subgroup AUROC rows
# ══════════════════════════════════════════════════════════════════════════════════
def _auroc(y, s):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return round(float(roc_auc_score(y, s)), 4)


def pathology_rows(method, checkpoint, pathology_scores, joint_sets):
    """One row per (task, subgroup): overall + each Race/View/Sex subgroup."""
    rows = []
    for task in PATHOLOGY_TASKS:
        b_df, task_scores = joint_sets[task], pathology_scores.get(task)
        if b_df is None or task_scores is None:
            continue
        pos = b_df["_mimic_pos"].values
        y = b_df["_label"].values
        s = task_scores[pos]

        def add(sub_type, sub_name, mask):
            yy, ss = y[mask], s[mask]
            rows.append(dict(
                method=method, checkpoint=checkpoint, task=task,
                subgroup_type=sub_type, subgroup=sub_name,
                n=int(mask.sum()), n_pos=int(yy.sum()), auroc=_auroc(yy, ss),
            ))

        add("overall", "all", np.ones(len(y), dtype=bool))
        for k, name in RACE_GROUPS.items():
            add("race", name, (b_df["Race"] == k).values)
        for k, name in VIEW_GROUPS.items():
            add("view", name, (b_df["View"] == k).values)
        for k, name in SEX_GROUPS.items():
            add("sex", name, (b_df["Sex"] == k).values)
    return rows


def attribute_rows(method, checkpoint, attribute_scores, test_df):
    """One row per attribute task: overall macro-OvO for race/view, plus a
    per-class one-vs-rest breakdown (race_Asian/Black/White, view_AP/PA/Lateral),
    and a single row for sex. Matches the breakdown train.py's own eval_epoch
    reports, so it's directly comparable across adapter types."""
    rows = []
    for task in BASE_TASKS:
        y, valid = get_attribute_labels(task, test_df)
        probs = attribute_scores[task][valid]

        if task == "sex":
            auroc = _auroc(y, probs)
            rows.append(dict(method=method, checkpoint=checkpoint, task=task,
                              n=len(y), n_pos=int((y == 1).sum()), auroc=auroc))
            continue

        class_names = MULTICLASS_CLASS_NAMES[task]
        n = len(y)
        if len(np.unique(y)) < 2:
            rows.append(dict(method=method, checkpoint=checkpoint, task=task, n=n, n_pos=np.nan, auroc=np.nan))
            for cname in class_names:
                rows.append(dict(method=method, checkpoint=checkpoint, task=f"{task}_{cname}", n=n, n_pos=np.nan, auroc=np.nan))
            continue

        overall = round(float(roc_auc_score(y, probs, multi_class="ovo", average="macro")), 4)
        rows.append(dict(method=method, checkpoint=checkpoint, task=task, n=n, n_pos=np.nan, auroc=overall))

        y_oh = np.eye(probs.shape[1])[y]
        per_class = roc_auc_score(y_oh, probs, average=None)
        for k, cname in enumerate(class_names):
            rows.append(dict(method=method, checkpoint=checkpoint, task=f"{task}_{cname}",
                              n=n, n_pos=int((y == k).sum()), auroc=round(float(per_class[k]), 4)))
    return rows
