import builtins
import copy
import datetime
import os
import random
from collections.abc import Iterable

import numpy as np
import torch


class ModelEMA:
    def __init__(self, params: Iterable[torch.nn.Parameter], rate: float = 0.999):
        self.rate = rate
        self.params = list(params)  # reference
        self.ema_params = [
            copy.deepcopy(p).detach().requires_grad_(False) for p in self.params
        ]

    @torch.no_grad()
    def update(self):
        for ema_p, p in zip(self.ema_params, self.params, strict=True):
            ema_p.mul_(self.rate).add_(p, alpha=1 - self.rate)

    @torch.no_grad()
    def apply(self):
        self.stored_params = [p.clone() for p in self.params]
        for p, ema_p in zip(self.params, self.ema_params, strict=True):
            p.copy_(ema_p)

    @torch.no_grad()
    def restore(self):
        assert self.stored_params is not None
        for p, stored_p in zip(self.params, self.stored_params, strict=True):
            p.copy_(stored_p)
        del self.stored_params


def seed_all(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:  # single GPU
        local_rank = rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=local_rank,
        timeout=datetime.timedelta(minutes=30),
    )
    print(
        f"WORLD_SIZE: {world_size}, RANK: {rank}, LOCAL_RANK: {local_rank}, "
        + f"MASTER: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )
    if rank != 0:
        # Only rank 0 prints, to keep multi-GPU logs readable.
        def print_pass(*args):
            pass

        builtins.print = print_pass
    return local_rank, rank, world_size


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    while True:
        if hasattr(model, "_orig_mod"):  # compiled wrapper
            model = model._orig_mod
            continue
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            continue
        return model
