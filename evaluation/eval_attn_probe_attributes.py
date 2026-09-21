"""Evaluate the frozen-attention race/sex/view probe checkpoints (trained by
training/train_attn_probe.py) on the STANDARD, un-resampled MIMIC test set.

Resampling doesn't apply here: it exists to remove a disease-prevalence confound
when evaluating pathology heads across subgroups. Here the label of interest IS the
demographic attribute itself, so there's no analogous confound — this just wants
overall AUROC on a representative sample, i.e. the natural test distribution.

Like eval_attn_multilayer.py, all N probe checkpoints are evaluated in one shared
pass: the backbone forward (the expensive part) runs once per image batch, and each
checkpoint's own frozen attention module is applied to its own layer slice. Within
each checkpoint, race/sex/view queries are also batched into a single attention
call, since K/V only depend on the (frozen, task-independent) input.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluation.common import MULTICLASS_CLASS_NAMES, get_attribute_labels
from training.datasets import CXR1M
from training.train import MultiHeadPredictor

ATTRIBUTE_TASKS = ["race", "sex", "view"]  # exactly the probe's own heads


def load_test_df(csv_filepath, dataset_filter):
    df = pd.read_csv(csv_filepath, low_memory=False)
    test_df = df[df["Split"] == "test"].copy().reset_index(drop=True)
    test_df = test_df[test_df["Dataset"] == int(dataset_filter)].copy().reset_index(drop=True)
    return test_df


def build_probe_model(name, checkpoint_root):
    ckpt_path = Path(checkpoint_root) / name / "best_checkpoint.pt"
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cargs = ckpt["args"]

    adapter_type = cargs.get("adapter_type", "none")
    assert adapter_type == "attn", f"{name}: expected adapter_type='attn', got '{adapter_type}'"
    layer_ids = cargs.get("layer_ids")
    assert layer_ids, f"{name}: checkpoint has no saved layer_ids"

    probe_tasks = {"race": 3, "sex": 1, "view": 3}
    model = MultiHeadPredictor(
        probe_tasks, adapter_type="attn", hidden_dim=cargs["hidden_dim"],
        head_dim=cargs.get("head_dim", 64), layer_ids=layer_ids,
    )
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)  # saved by exactly this class — should load cleanly
    model.eval()

    print(f"  loaded {name}: layer_ids={layer_ids}")
    return model, layer_ids, cargs


def run_shared_inference(models_info, test_df, args, device):
    resolutions = {info["cargs"].get("img_resolution", 512) for info in models_info.values()}
    assert len(resolutions) == 1, f"checkpoints disagree on img_resolution ({resolutions})"
    res = resolutions.pop()

    eval_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((res, res)), transforms.ToTensor()]
    )
    ds = CXR1M(
        root=args.data_dir, csv_filepath=args.csv_filepath, split="test",
        transform=eval_transform, cache_root=None, dataset_filter=args.dataset_filter,
    )
    assert len(ds.df) == len(test_df), "CXR1M test rows misaligned with test_df"

    all_logits = {
        name: {"race": np.full((len(test_df), 3), np.nan, dtype=np.float32),
               "view": np.full((len(test_df), 3), np.nan, dtype=np.float32),
               "sex": np.full(len(test_df), np.nan, dtype=np.float32)}
        for name in models_info
    }

    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    any_model = next(iter(models_info.values()))["model"]
    preprocess_fn = any_model.preprocess
    backbone = any_model.backbone  # frozen + identical across all checkpoints — reuse

    row = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"shared backbone inference ({len(models_info)} probe checkpoints)"):
            x, _ = preprocess_fn(batch["x"], batch["pa"])
            bs_here = x.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden_states = backbone(x, output_hidden_states=True).hidden_states  # computed ONCE
                for name, info in models_info.items():
                    model = info["model"]
                    feats = torch.cat([hidden_states[i] for i in info["layer_ids"]], dim=-1)
                    tasks = ["race", "sex", "view"]
                    queries = torch.cat([model.queries[t].expand(bs_here, -1, -1) for t in tasks], dim=1)
                    attn_out, _ = model.attn(queries, feats, feats)
                    for i, t in enumerate(tasks):
                        logits = model.heads[t](attn_out[:, i]).float().cpu().numpy()
                        if t == "sex":
                            all_logits[name][t][row:row + bs_here] = logits[:, 0]
                        else:
                            all_logits[name][t][row:row + bs_here] = logits
            row += bs_here
    return all_logits


def score_task(task, logits, test_df):
    y, valid = get_attribute_labels(task, test_df)
    l = logits[valid]
    if task == "sex":
        s = expit(l)
        if len(np.unique(y)) < 2:
            return [(task, np.nan, len(y), int((y == 1).sum()))]
        return [(task, round(float(roc_auc_score(y, s)), 4), len(y), int((y == 1).sum()))]

    n = len(y)
    class_names = MULTICLASS_CLASS_NAMES[task]
    if len(np.unique(y)) < 2:
        rows = [(task, np.nan, n, np.nan)]
        rows += [(f"{task}_{c}", np.nan, n, np.nan) for c in class_names]
        return rows

    ss = softmax(l, axis=1)
    overall = round(float(roc_auc_score(y, ss, multi_class="ovo", average="macro")), 4)
    rows = [(task, overall, n, np.nan)]
    y_oh = np.eye(ss.shape[1])[y]
    per_class = roc_auc_score(y_oh, ss, average=None)
    for k, cname in enumerate(class_names):
        rows.append((f"{task}_{cname}", round(float(per_class[k]), 4), n, int((y == k).sum())))
    return rows


def collect_rows(name, layer_ids, logits_by_task, test_df):
    rows = []
    base_ckpt = name.replace("_attribute_probe", "")
    for task in ATTRIBUTE_TASKS:
        for subtask, auroc, n, n_pos in score_task(task, logits_by_task[task], test_df):
            rows.append(dict(
                method="Attention Pooling", checkpoint=name, base_checkpoint=base_ckpt,
                layer_ids=",".join(str(i) for i in layer_ids), task=subtask,
                n=n, n_pos=n_pos, auroc=auroc,
            ))
    return rows


def print_checkpoint_table(name, rows):
    order = ["race", "race_Asian", "race_Black", "race_White", "sex", "view", "view_AP", "view_PA", "view_Lateral"]
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    s = pd.DataFrame(rows).set_index("task")["auroc"].reindex(order)
    print(s.to_string(float_format=lambda v: f"{v:.4f}"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True,
                    help="Folder names under --checkpoint_root, each a *_attribute_probe checkpoint.")
    p.add_argument("--checkpoint_root", default="./checkpoints")
    p.add_argument("--csv_filepath", default="./data/chai_cxr_master.csv")
    p.add_argument("--data_dir", default="/path/to/mimic-cxr-jpg")
    p.add_argument("--dataset_filter", default="1")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", default="./outputs/eval/attn_probe_attributes_eval.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading standard (un-resampled) test split...")
    test_df = load_test_df(args.csv_filepath, args.dataset_filter)
    print(f"  test rows (Dataset={args.dataset_filter}): {len(test_df)}")

    print(f"\nLoading {len(args.checkpoints)} probe checkpoints...")
    models_info = {}
    for name in args.checkpoints:
        try:
            model, layer_ids, cargs = build_probe_model(name, args.checkpoint_root)
        except AssertionError as e:
            print(f"skipping {name}: {e}")
            continue
        model.to(device)
        models_info[name] = {"model": model, "layer_ids": layer_ids, "cargs": cargs}

    if not models_info:
        print("No valid probe checkpoints loaded — nothing to evaluate.")
        return

    print(f"\nRunning shared-backbone inference for {len(models_info)} probe checkpoints in one pass...")
    all_logits = run_shared_inference(models_info, test_df, args, device)

    all_rows = []
    for name, info in models_info.items():
        rows = collect_rows(name, info["layer_ids"], all_logits[name], test_df)
        all_rows.extend(rows)
        print_checkpoint_table(name, rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
