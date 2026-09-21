"""Extract and cache frozen Rad-DINO CLS-token embeddings for every image in a split.

The `no_adapter` and `mlp_adapter` training/evaluation scripts read pre-extracted
768-dim CLS embeddings from a memmap instead of running the backbone on raw images
every time (the backbone is frozen and its output never changes, so this is a pure
speed-up). Attention pooling never uses this cache — it always needs the full set of
patch tokens across multiple layers, so it runs the live backbone on raw images.

Run this once per split before training/evaluating `no_adapter` or `mlp_adapter`.

IMPORTANT: this extracts an embedding for every row of the split (every dataset in
your metadata CSV, not just MIMIC-CXR), because downstream code indexes into the
resulting memmap by row position *before* filtering to a single dataset. Filtering
to one dataset here would silently misalign every other script's lookups.

Usage:
  python -m training.extract_cls_embeddings \
      --data_dir <path-to-mimic-cxr-jpg> \
      --csv_filepath ./data/chai_cxr_master.csv \
      --output_dir ./cache/raddino
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from training.datasets import CXR1M
from training.utils import seed_all, seed_worker

EMB_DIM = 768


class CLSExtractor(nn.Module):
    """Frozen Rad-DINO wrapper that returns only the CLS token (patch tokens
    discarded)."""

    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/rad-dino", dtype=torch.bfloat16
        ).eval()
        self.hf_processor = AutoImageProcessor.from_pretrained(
            "microsoft/rad-dino", use_fast=True
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def preprocess(self, x: Tensor) -> Tensor:
        device = next(self.backbone.parameters()).device
        if (x.ndim == 4) and (x.shape[1] == 1):  # greyscale -> Rad-DINO's expected RGB input
            x = self.hf_processor(x.repeat(1, 3, 1, 1), return_tensors="pt", do_rescale=False)["pixel_values"]
        return x.to(device, non_blocking=True)

    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess(x)
        return self.backbone(x).last_hidden_state[:, 0]  # (batch, 768) CLS token


@torch.inference_mode()
def extract_split(split: str, dataloader: torch.utils.data.DataLoader, model: CLSExtractor, output_dir: Path):
    n_images = len(dataloader.dataset)
    emb_path = output_dir / f"raddino_emb_float32_{split}.dat"
    print(f"\n── {split}: {n_images} images -> {emb_path}")

    emb_mm = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n_images, EMB_DIM))

    ptr, t0 = 0, time.time()
    for batch in tqdm(dataloader, desc=f"Extracting {split}"):
        x = batch["x"]
        bs = x.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = model(x)  # (bs, 768)
        emb_mm[ptr:ptr + bs] = cls.float().cpu().numpy()
        ptr += bs
        if ptr % 10000 < bs:  # periodic flush so a crash doesn't lose everything
            emb_mm.flush()
    emb_mm.flush()

    elapsed = time.time() - t0
    size_gb = (n_images * EMB_DIM * 4) / (1024 ** 3)
    print(f"   Done in {elapsed/60:.1f} min  |  {size_gb:.2f} GB  |  shape: {emb_mm.shape}")


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading Rad-DINO...")
    model = CLSExtractor().to(device).eval()
    print("Rad-DINO loaded and frozen.")

    for split in args.splits.split(","):
        dataset = CXR1M(
            root=args.data_dir,
            csv_filepath=args.csv_filepath,
            split=split,
            cache_root=None,       # raw images, not the cache we're building
            dataset_filter=None,   # every dataset in the split — see module docstring
        )
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.bs,
            shuffle=False,      # order must match the memmap's row layout
            drop_last=False,    # every image must be captured
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )
        extract_split(split, loader, model, output_dir)

    print("\nExtraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--csv_filepath", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", type=str, default="train,valid,test",
                        help="Comma-separated splits to extract.")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--img_resolution", type=int, default=512)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
