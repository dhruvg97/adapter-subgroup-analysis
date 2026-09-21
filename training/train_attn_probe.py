"""Train race/sex/view attribute probes on top of a FROZEN, pathology-trained
attention-pooling checkpoint.

This deliberately does not modify train.py or datasets.py. It reuses train.py's
Trainer, get_dataloaders, ModelEMA, etc. unmodified, and only adds what's new:

  1. A frozen probe model: loads a pathology attn checkpoint's backbone + attention
     module (both frozen, zero gradient), and attaches brand-new, randomly
     initialised queries + heads for race (3-class), sex (binary), view (3-class).
     Nothing about the pathology run's weights or its own metrics is touched.

  2. A batched-query attention forward pass: MultiHeadPredictor.forward loops over
     tasks and calls self.attn(query, x, x) once per task, recomputing the K/V
     projection of x from scratch every time even though x (the frozen multi-layer
     backbone output) is identical across race/sex/view. Since attn is frozen, x is
     also fixed regardless of query — so all 3 tasks' queries are stacked into a
     single nn.MultiheadAttention call, computing K/V once and reusing it.

No augmentation is used for the train split by default (recommended for probing a
frozen representation) — pass --augment to use train.py's augmented pipeline instead.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel.distributed import DistributedDataParallel
from torchvision import transforms

from training.datasets import CXR1M, get_cxr1m, get_dataloaders
from training.train import MultiHeadPredictor, Trainer
from training.utils import ModelEMA, seed_all, setup_distributed


# ── Frozen-attention probe model ────────────────────────────────────────────────────
class FrozenAttnProbe(MultiHeadPredictor):
    """Attention-pooling adapter with the pooling module frozen, and every active
    task's query batched into a single nn.MultiheadAttention call (see module
    docstring point 2)."""

    def forward(self, x, targets, pos_weights=None):
        if x.ndim == 4:
            x = self.forward_backbone(x)

        active_tasks = [t for t in self.heads if t in targets]  # deterministic order
        if not active_tasks:
            raise ValueError("Missing task keys.")

        b = x.shape[0]
        queries = torch.cat([self.queries[t].expand(b, -1, -1) for t in active_tasks], dim=1)  # (b, n_tasks, d)
        attn_out, _ = self.attn(queries, x, x)  # K/V computed ONCE, shared by every task
        logits = {t: self.heads[t](attn_out[:, i]) for i, t in enumerate(active_tasks)}

        loss = self.auto_multi_loss(logits, targets, pos_weights=pos_weights)
        return loss, logits


def build_frozen_probe_model(args):
    ckpt_path = Path(args.checkpoint_root) / args.pretrained_checkpoint_name / "best_checkpoint.pt"
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cargs = ckpt["args"]

    adapter_type = cargs.get("adapter_type", "none")
    assert adapter_type == "attn", (
        f"{args.pretrained_checkpoint_name}: expected adapter_type='attn', got '{adapter_type}'"
    )
    layer_ids = cargs.get("layer_ids")
    assert layer_ids, f"{args.pretrained_checkpoint_name}: checkpoint has no saved layer_ids"

    probe_tasks = {"race": 3, "sex": 1, "view": 3}  # matches train.py's dataset/race/view class-count overrides

    model = FrozenAttnProbe(
        probe_tasks, adapter_type="attn", hidden_dim=cargs["hidden_dim"],
        head_dim=cargs.get("head_dim", 64), layer_ids=layer_ids,
    )

    full_state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    attn_state = {k[len("attn."):]: v for k, v in full_state.items() if k.startswith("attn.")}
    assert attn_state, f"{args.pretrained_checkpoint_name}: no 'attn.*' weights found in checkpoint"
    model.attn.load_state_dict(attn_state, strict=True)  # strict: fail now, not after 30 epochs

    for p in model.attn.parameters():
        p.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"loaded frozen attn from {args.pretrained_checkpoint_name} (layer_ids={layer_ids})")
    print(f"  trainable params: {n_trainable:,}  (new race/sex/view queries + heads)")
    print(f"  frozen params:    {n_frozen:,}  (backbone + attention pooling)")
    return model, layer_ids, cargs


# ── Dataset construction (augmentation on/off) ──────────────────────────────────────
def build_datasets(args):
    if args.augment:
        print("Using train.py's augmentation pipeline (--augment was set).")
        return get_cxr1m(args)

    print("No augmentation for train split (recommended for probing a frozen "
          "representation) — plain resize + tensor, same as the eval transform.")
    eval_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(args.img_resolution, args.img_resolution)),
        transforms.ToTensor(),
    ])
    return {
        k: CXR1M(
            root=args.data_dir, csv_filepath=args.csv_filepath, split=k,
            transform=eval_tf, cache_root=None, dataset_filter=args.dataset_filter,
        )
        for k in ["train", "valid"]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained_checkpoint_name", required=True,
                    help="Folder name under --checkpoint_root of the frozen pathology attn checkpoint.")
    p.add_argument("--checkpoint_root", default="./checkpoints")
    p.add_argument("--save_dir", default=None,
                    help="Default: <checkpoint_root>/<pretrained_checkpoint_name>_attribute_probe")
    p.add_argument("--exp_name", default=None, help="Default: same as save_dir's folder name.")

    p.add_argument("--csv_filepath", default="./data/chai_cxr_master.csv")
    p.add_argument("--data_dir", default="/path/to/mimic-cxr-jpg")
    p.add_argument("--dataset_filter", type=str, default="1")
    p.add_argument("--img_resolution", type=int, default=512)
    p.add_argument("--img_channels", type=int, default=1)
    p.add_argument("--augment", action="store_true", default=False)

    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=64,
                    help="Higher than the pathology run's default — the backward pass is now tiny "
                         "(frozen attn + backbone), so throughput can usually absorb a bigger batch.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_warmup", type=int, default=100)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.999])
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--ema_rate", type=float, default=0.99)
    p.add_argument("--eval_freq", type=int, default=400)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--determ", action="store_true", default=False)
    p.add_argument("--dist", action="store_true", default=False)
    p.add_argument("--wandb_mode", default="disabled", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    if args.exp_name is None:
        args.exp_name = f"{args.pretrained_checkpoint_name}_attribute_probe"
    if args.save_dir is None:
        args.save_dir = str(Path(args.checkpoint_root) / args.exp_name)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    device, rank, world_size = (
        setup_distributed() if args.dist
        else (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 0, 1)
    )
    is_dist = args.dist and dist.is_available() and dist.is_initialized()
    seed_all(args.seed, args.determ)

    model, layer_ids, base_cargs = build_frozen_probe_model(args)
    model.to(device)

    # Trainer.save_checkpoint saves vars(args) — our own argparse Namespace, which
    # never defines --adapter_type/--hidden_dim/--head_dim/--layer_ids as CLI args.
    # Stamp them on here so the saved probe checkpoint is self-describing, same as
    # every pathology checkpoint already is.
    args.adapter_type = "attn"
    args.hidden_dim = base_cargs["hidden_dim"]
    args.head_dim = base_cargs.get("head_dim", 64)
    args.layer_ids = layer_ids

    datasets_dict = build_datasets(args)
    if rank == 0:
        print(f"  train rows: {len(datasets_dict['train'])}  |  valid rows: {len(datasets_dict['valid'])}")
    dataloaders = get_dataloaders(args, datasets_dict)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.wd, betas=tuple(args.betas), eps=args.eps,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / args.lr_warmup, total_iters=args.lr_warmup
    )
    # Restricted to trainable_params to avoid deep-copying the frozen backbone +
    # attention weights as an EMA shadow that would never change.
    ema = ModelEMA(trainable_params, rate=args.ema_rate)

    model_for_train = model
    if is_dist:
        model_for_train = DistributedDataParallel(model, device_ids=[device])

    if rank == 0:
        wandb.init(
            project="adapter-subgroup-analysis", name=args.exp_name, config=vars(args),
            mode=args.wandb_mode,
        )
        for k, v in vars(args).items():
            print(f"--{k}={v}")
        print(f"#trainable params: {sum(p.numel() for p in trainable_params):,}")

    trainer = Trainer(model_for_train, args, optimizer=optimizer, scheduler=scheduler, ema=ema)

    train_loss = 0.0
    for i in range(args.epochs):
        if is_dist:
            dataloaders["train"].sampler.set_epoch(i)
        now = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        print(f"\n{now}, Epoch {i+1}:")
        train_loss = trainer.train_epoch(dataloaders)
        if rank == 0:
            wandb.log({"train_loss": train_loss}, trainer.step)

    if rank == 0:
        trainer.save_checkpoint(-1.0)
        print(f"Final checkpoint saved at step {trainer.step}, train_loss={train_loss:.5f}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
