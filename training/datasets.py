import argparse
import os
import random
from collections.abc import Callable
from typing import TypedDict, get_type_hints

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from training.utils import seed_worker


class Metadata(TypedDict):
    dataset: int
    age: float | None
    race: int
    race_asian: int   # 1 if Asian, else 0
    race_black: int   # 1 if Black, else 0
    race_white: int   # 1 if White, else 0
    sex: int
    sex_male: int     # 1 if Male, else 0
    view: int
    view_ap: int      # 1 if AP, else 0
    view_pa: int      # 1 if PA, else 0
    view_lateral: int # 1 if LATERAL, else 0
    atelectasis: int
    cardiomegaly: int
    consolidation: int
    edema: int
    enlarged_cardiomediastinum: int
    fracture: int
    lung_lesion: int
    lung_opacity: int
    no_finding: int
    pleural_effusion: int
    pleural_other: int
    pneumonia: int
    pneumothorax: int
    support_devices: int


# Number of classes per task's decision head — 1 = binary (BCE), >1 = categorical (CE).
CLASS_SCHEMA: Metadata = {
    "dataset": 8,
    "age": None,
    "race": 4,
    "race_asian": 3,  # one-vs-rest class
    "race_black": 3,
    "race_white": 3,
    "sex": 3,
    "sex_male": 3,
    "view": 4,
    "view_ap": 3,
    "view_pa": 3,
    "view_lateral": 3,
    "atelectasis": 3,
    "cardiomegaly": 3,
    "consolidation": 3,
    "edema": 3,
    "enlarged_cardiomediastinum": 3,
    "fracture": 3,
    "lung_lesion": 3,
    "lung_opacity": 3,
    "no_finding": 3,
    "pleural_effusion": 3,
    "pleural_other": 3,
    "pneumonia": 3,
    "pneumothorax": 3,
    "support_devices": 3,
}


_DERIVED_KEYS = {"race_asian", "race_black", "race_white", "sex_male", "view_ap", "view_pa", "view_lateral"}


def key_to_col(key: str) -> str:
    return " ".join(word.capitalize() for word in key.split("_"))


def get_sample(
    root: str, row: pd.Series, return_image: bool = True
) -> tuple[np.ndarray, Metadata] | Metadata:
    metadata: Metadata = {k: row[key_to_col(k)] for k in get_type_hints(Metadata) if k not in _DERIVED_KEYS}
    metadata["age"] = metadata["age"] / 100

    # One-vs-rest binary labels: NaN(0)->0, not-class(other value)->0, is-class->1.
    # Integer encoding: race NaN=0,Asian=1,Black=2,White=3 | sex NaN=0,Male=1,Female=2
    # | view NaN=0,AP=1,PA=2,LATERAL=3. These OvR heads are not used by any figure in
    # this repo (they were a legacy probe format) — kept only for anyone extending
    # datasets.py with additional demographic probes.
    r, s, v = row["Race"], row["Sex"], row["View"]
    metadata["race_asian"]   = 0 if r == 0 else (2 if r == 1 else 1)
    metadata["race_black"]   = 0 if r == 0 else (2 if r == 2 else 1)
    metadata["race_white"]   = 0 if r == 0 else (2 if r == 3 else 1)
    metadata["sex_male"]     = 0 if s == 0 else (2 if s == 1 else 1)
    metadata["view_ap"]      = 0 if v == 0 else (2 if v == 1 else 1)
    metadata["view_pa"]      = 0 if v == 0 else (2 if v == 2 else 1)
    metadata["view_lateral"] = 0 if v == 0 else (2 if v == 3 else 1)

    # "No Finding" positive iff the column is explicitly 2 — everything else (never
    # mentioned, or another pathology present) counts as negative under the
    # NaN-as-negative label convention used throughout training and evaluation.
    metadata["no_finding"] = 2 if row["No Finding"] == 2 else 0

    if return_image:
        image = np.array(Image.open(os.path.join(root, row["ImagePath"])))
        if image.dtype == np.uint16:
            image = (image >> 8).astype("uint8")
        return image, metadata
    else:
        return metadata


class CXR1M(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        csv_filepath: str,
        split: str,
        transform: Callable | None = None,
        parents: list[str] | None = None,
        cache_root: str | None = None,
        dataset_filter: str | None = None,
        patient_ids: np.ndarray | None = None,  # for patient-level bootstrapping
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.cache_root = cache_root
        if cache_root is not None:
            print(f"Using {split} memmap...")
            file = f"raddino_emb_float32_{split}.dat"
            self.CHW, self.mm_dtype = (768,), np.float32
            self.cache_root = os.path.join(cache_root, file)
            assert os.path.exists(self.cache_root), (
                f"{self.cache_root} not found — run training/extract_cls_embeddings.py first."
            )
        self.cache = None  # lazy
        self.parents = None if parents is None else set(parents)  # set for O(1)
        self.df = pd.read_csv(csv_filepath, low_memory=False)
        self.df = self.df.loc[self.df["Split"] == split].copy()
        # embed_idx must be assigned before dataset filtering so positions match the .dat file
        self.df["_embed_idx"] = np.arange(len(self.df))
        self.n_split = len(self.df)  # full split size used for memmap shape
        if dataset_filter is not None and dataset_filter != "all":
            self.df = self.df.loc[self.df["Dataset"] == int(dataset_filter)].copy()
        # ── Bootstrap patient filter with correct replacement weighting ───────────
        if patient_ids is not None:
            pid_counts = pd.Series(patient_ids).value_counts()
            self.df    = self.df.loc[self.df["PatientID"].isin(pid_counts.index)].copy()
            repeat_n   = self.df["PatientID"].map(pid_counts).values
            self.df    = self.df.loc[self.df.index.repeat(repeat_n)]
        # ────────────────────────────────────────────────────────────────────────
        self.df = self.df.reset_index(drop=True)

    def _maybe_get_cache(self) -> None:
        if (self.cache is None) and (self.cache_root is not None):
            self.cache = np.memmap(
                self.cache_root,
                mode="r",
                dtype=self.mm_dtype,
                shape=(self.n_split, *self.CHW),
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        self._maybe_get_cache()  # lazy
        while True:
            try:
                if self.cache is not None:
                    embed_idx = int(self.df.iloc[idx]["_embed_idx"])
                    image = np.array(self.cache[embed_idx], copy=True)
                    metadata = get_sample(self.root, self.df.iloc[idx], False)
                else:
                    image, metadata = get_sample(self.root, self.df.iloc[idx])
                break
            except (OSError, ValueError, RuntimeError):
                # A small number of MIMIC-CXR JPEGs are corrupted; resample a
                # different index rather than crashing a multi-hour training run.
                idx = random.randrange(len(self))

        if image.ndim == 2:
            image = image[None, ...]
        image = torch.from_numpy(image)
        if self.transform is not None:
            image = self.transform(image)

        if self.parents is not None:
            metadata = {k: v for k, v in metadata.items() if k in self.parents}

        return dict(x=image, pa=metadata)


def get_cxr1m(args: argparse.Namespace) -> dict[str, CXR1M]:
    if args.img_channels == -1:  # use cached raddino embeddings, no image transform
        transform = {"train": None, "eval": None}
    else:
        transform = {
            "train": transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(args.img_resolution, args.img_resolution)),
                    transforms.ToTensor(),
                ]
            ),
            "eval": transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(args.img_resolution, args.img_resolution)),
                    transforms.ToTensor(),
                ]
            ),
        }

    patient_ids_file = getattr(args, "patient_ids_file", None)
    patient_ids = np.load(patient_ids_file, allow_pickle=True) if patient_ids_file else None

    datasets = {
        k: CXR1M(
            root=args.data_dir,
            csv_filepath=args.csv_filepath,
            split=k,
            transform=transform[(k if k == "train" else "eval")],
            cache_root=getattr(args, "cache_dir", None),
            dataset_filter=getattr(args, "dataset_filter", None),
            patient_ids=(patient_ids if k == "train" else None),  # bootstrap draws train only
        )
        for k in ["train", "valid", "test"]
    }
    return datasets


def get_dataloaders(
    args: argparse.Namespace, datasets: dict[str, CXR1M]
) -> dict[str, torch.utils.data.DataLoader]:
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_dist:
        from torch.utils.data.distributed import DistributedSampler

    s = int(getattr(args, "resume_step", 0))
    rank = torch.distributed.get_rank() if is_dist else 0

    dataloaders = {}
    for k in ["train", "valid"]:  # test is scored separately by the evaluation scripts
        is_train = k == "train"
        seed, sampler = int(args.seed + (7654321 * s if is_train else 0)), None
        if is_dist:
            sampler = DistributedSampler(datasets[k], shuffle=is_train, seed=seed)
        g = torch.Generator()
        g.manual_seed(seed + rank)
        kwargs = dict(
            dataset=datasets[k],
            batch_size=args.bs,
            shuffle=(sampler is None) and is_train,
            drop_last=is_train,
            sampler=sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if args.num_workers > 0:
            kwargs["prefetch_factor"] = args.prefetch_factor
            kwargs["persistent_workers"] = True

        dataloaders[k] = torch.utils.data.DataLoader(**kwargs)
    return dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--csv_filepath", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--img_resolution", type=int, default=512)
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()
    datasets = get_cxr1m(args)
    dataloaders = get_dataloaders(args, datasets)
    batch = next(iter(dataloaders["train"]))
    img_shape = (1, args.img_resolution, args.img_resolution)
    assert batch["x"].shape == (args.bs, *img_shape)
    print("OK — dataloaders build correctly.")
