import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from training.datasets import CLASS_SCHEMA, get_cxr1m, get_dataloaders
from training.utils import ModelEMA, seed_all, setup_distributed, unwrap

# Label convention: an "unmentioned" pathology/attribute (raw value 0) is treated
# as a negative, same as an explicit negative — no row is ever dropped for a
# binary task. This matches every checkpoint trained for the paper. Categorical
# tasks (race/view) still mask out raw 0 (there is no "negative" class for those).
NAN_AS_NEGATIVE = True


class MultiHeadPredictor(nn.Module):
    def __init__(
        self,
        tasks: dict[str, int],  # e.g. {"pleural_effusion": 1, "cardiomegaly": 1, "race": 3}
        adapter_type: str = "none",  # "none" | "mlp" | "attn"
        hidden_dim: int = 768,
        dropout: float = 0.0,
        head_dim: int = 64,
        layer_ids: list[int] = None,  # only used when adapter_type == "attn"
    ):
        super().__init__()
        self.tasks = tasks
        self.adapter_type = adapter_type
        self.backbone = AutoModel.from_pretrained(
            "microsoft/rad-dino", dtype=torch.bfloat16
        ).eval()
        self.hf_processor = AutoImageProcessor.from_pretrained(
            "microsoft/rad-dino", use_fast=True
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        backbone_dim = self.backbone.config.hidden_size

        if adapter_type == "attn":
            self.layer_ids = layer_ids
            hidden_dim = len(self.layer_ids) * backbone_dim
            assert hidden_dim % head_dim == 0
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=hidden_dim // head_dim,
                batch_first=True,
            )
            # One learnable query vector per task — attention decides which of the
            # concatenated layers/tokens matter most for that task.
            self.queries = nn.ParameterDict(
                {k: nn.Parameter(torch.zeros(1, 1, hidden_dim)) for k in tasks.keys()}
            )
            for q in self.queries.values():
                nn.init.trunc_normal_(q, std=0.02)
            head_in_dim = hidden_dim

        elif adapter_type == "mlp":
            self.adapter = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_in_dim = hidden_dim

        else:  # "none" — heads act directly on the raw backbone CLS embedding
            head_in_dim = backbone_dim

        self.heads = nn.ModuleDict(
            {k: nn.Linear(head_in_dim, out_dim) for k, out_dim in tasks.items()}
        )

    def forward(
        self, x: Tensor, targets: dict[str, Tensor], pos_weights=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if x.ndim == 4:  # raw image -> backbone features
            x = self.forward_backbone(x)
        if self.adapter_type == "attn":
            logits = {}
            active_tasks = targets.keys() & self.heads.keys()  # skip tasks with no labels in this batch
            for task in active_tasks:
                query = self.queries[task].expand(x.shape[0], -1, -1)  # (b, 1, len(layer_ids)*d)
                attn_out, _ = self.attn(query, x, x)
                logits[task] = self.heads[task](attn_out[:, 0])
        else:
            if self.adapter_type == "mlp":
                x = self.adapter(x)
            logits = {task: head(x) for task, head in self.heads.items()}

        loss = self.auto_multi_loss(logits, targets, pos_weights=pos_weights)
        return loss, logits

    @torch.no_grad()
    def forward_backbone(self, x: Tensor) -> Tensor:
        if self.adapter_type == "attn":
            hs = self.backbone(x, output_hidden_states=True).hidden_states
            return torch.cat([hs[i] for i in self.layer_ids], dim=-1)  # (b, t, len(layer_ids)*d)
        else:
            return self.backbone(x).last_hidden_state[:, 0]  # (b, d) CLS token

    def auto_multi_loss(self, logits: dict[str, Tensor], targets: dict[str, Tensor], pos_weights=None):
        active_tasks = targets.keys() & logits.keys()
        if not active_tasks:
            raise ValueError("Missing task keys.")

        losses, active = [], 0
        for task in active_tasks:
            z, y = logits[task], targets[task]
            is_categorical = (z.ndim == 2) and (z.shape[1] > 1)
            if not is_categorical:
                # Unmentioned (raw 0 -> preprocess()'s -1 sentinel) is always a negative.
                y = torch.where(y == -1, torch.zeros_like(y), y)
            mask = (y != -1) if is_categorical else (~torch.isnan(y))
            if not mask.any():  # no supervision for this task in this batch
                losses.append(z.sum() * 0.0)  # keeps the task in the graph with zero contribution
                continue
            if is_categorical:
                losses.append(nn.functional.cross_entropy(z, y.long(), ignore_index=-1))
            else:
                if NAN_AS_NEGATIVE:
                    y = torch.where(torch.isnan(y), torch.zeros_like(y), y)  # truly missing -> negative too
                    mask = torch.ones_like(y, dtype=torch.bool)
                z, y = z[mask].squeeze(-1), y[mask].float()
                if ((y == 0) | (y == 1)).all():  # binary
                    pw = pos_weights.get(task) if pos_weights else None
                    losses.append(nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=pw))
                else:  # continuous (age)
                    losses.append((z - y).abs().mean())
            active += 1
        return torch.stack(losses).sum() / max(active, 1)

    def preprocess(
        self, x: Tensor, pa: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = next(self.backbone.parameters()).device
        if (x.ndim == 4) and (x.shape[1] == 1):  # greyscale -> Rad-DINO's expected RGB input
            x = self.hf_processor(
                x.repeat(1, 3, 1, 1), return_tensors="pt", do_rescale=False
            )["pixel_values"]
        x = x.to(device, non_blocking=True)
        for k, v in pa.items():
            pa[k] = v.to(device, non_blocking=True)
            if k != "age":
                pa[k] = pa[k] - 1  # shifts raw-0/"unmentioned" to -1, the loss's ignore/negative sentinel
        return x, pa


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: argparse.Namespace,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        ema: ModelEMA | None = None,
    ):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.device = next(model.parameters()).device
        self.is_dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.step, self.epoch = 0, 0
        self.best_auroc = -1.0  # higher is better
        interval = float(os.environ.get("TQDM_MININTERVAL", 1))
        self.tqdm_kwargs = dict(disable=(self.rank != 0), mininterval=interval)

    def train_epoch(self, dataloaders: dict[str, torch.utils.data.DataLoader]) -> float:
        missing = {"train", "valid"} - dataloaders.keys()
        assert not missing, f"Missing dataloader(s): {sorted(missing)}"
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        dataloader = dataloaders["train"]
        loader = tqdm(enumerate(dataloader), total=len(dataloader), **self.tqdm_kwargs)
        total_loss = torch.tensor(0.0, device=self.device)
        n = torch.tensor(0, device=self.device)

        for _, batch in loader:
            x, pa = batch["x"], batch["pa"]
            bs = x.shape[0]
            x, pa = getattr(self.model, "module", self.model).preprocess(x, pa)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = self.model(x, pa)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            n += bs
            total_loss += loss.detach() * bs
            if self.rank == 0:
                wandb.log({"gnorm": gnorm, "lr": lr}, self.step)
                loader.update(1)
                loader.set_description(
                    f"train loss: {total_loss / n:.5f}", refresh=False
                )
            if (self.step % self.args.eval_freq) == 0:
                self.model.eval()
                if self.ema is not None:
                    self.ema.apply()
                metrics = self.eval_epoch(dataloaders["valid"])
                if self.rank == 0:
                    auroc_keys = [k for k in metrics.keys() if k.endswith("_rocauc") or k.endswith("_rocauc_macroavg_ovo")]
                    mean_auroc = sum(metrics[k] for k in auroc_keys) / len(auroc_keys) if auroc_keys else 0.0
                    metrics["mean_auroc"] = mean_auroc
                    wandb.log({f"valid_{k}": v for k, v in metrics.items()}, self.step)
                    self.save_checkpoint(mean_auroc)
                    print("\n".join(f"{k}: {v:7f}" for k, v in metrics.items()))
                if self.ema is not None:
                    self.ema.restore()
                self.model.train()

        self.epoch += 1
        if self.is_dist:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
        return (total_loss / n).item()

    @torch.inference_mode()
    def eval_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()
        loader = tqdm(enumerate(dataloader), total=len(dataloader), **self.tqdm_kwargs)
        total_loss = torch.tensor(0.0, device=self.device)
        n = torch.tensor(0, device=self.device)
        tasks = list(getattr(self.model, "module", self.model).tasks)
        preds, targets = {t: [] for t in tasks}, {t: [] for t in tasks}

        for _, batch in loader:
            x, pa = batch["x"], batch["pa"]
            bs = x.shape[0]
            x, pa = getattr(self.model, "module", self.model).preprocess(x, pa)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logits = self.model(x, pa)
            n += bs
            total_loss += loss.detach() * bs

            for task in tasks:
                z, y = logits[task], pa[task]
                is_categorical = (z.ndim == 2) and (z.shape[1] > 1)
                if is_categorical:
                    mask = (y != -1)
                else:
                    y = torch.where(y == -1, torch.zeros_like(y), y)  # mirrors auto_multi_loss
                    if NAN_AS_NEGATIVE:
                        y = torch.where(torch.isnan(y), torch.zeros_like(y), y)
                        mask = torch.ones_like(y, dtype=torch.bool)
                    else:
                        mask = ~torch.isnan(y)
                if not mask.any():
                    continue
                preds[task].append(z[mask].detach())
                targets[task].append(y[mask].detach())

            if self.rank == 0:
                loader.set_description(f"eval loss: {total_loss/n:.5f}", refresh=False)

        if self.is_dist:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)

            for task in tasks:
                local_z = torch.cat(preds[task], dim=0).cpu()
                if len(targets[task]) == 0:
                    local_y = torch.empty((0,), dtype=torch.long)
                else:
                    local_y = torch.cat(targets[task], dim=0).cpu()

                gathered_z = gathered_y = None
                if self.rank == 0:
                    ws = dist.get_world_size()
                    gathered_z, gathered_y = [None] * ws, [None] * ws

                dist.gather_object(local_z, gathered_z, dst=0)
                dist.gather_object(local_y, gathered_y, dst=0)

                preds[task], targets[task] = [], []
                if self.rank == 0:
                    preds[task] = [torch.cat(gathered_z, dim=0)]
                    targets[task] = [torch.cat(gathered_y, dim=0)]

            if self.rank != 0:
                return None

        metrics = {"loss": (total_loss / n).item()}
        for task in tasks:
            if (len(preds[task]) == 0) or (len(targets[task]) == 0):
                continue
            z = torch.cat(preds[task], dim=0).float().cpu().squeeze(-1)
            y = torch.cat(targets[task], dim=0).cpu()

            if (z.ndim == 2) and (z.shape[1] > 1):  # multiclass (e.g. race, view)
                y_oh   = nn.functional.one_hot(y.long(), num_classes=z.shape[1]).numpy()
                scores = z.softmax(dim=-1).numpy()

                metrics[f"{task}_auprc_macroavg"] = average_precision_score(y_oh, scores, average="macro")
                per_class_ap = average_precision_score(y_oh, scores, average=None)
                for k, ap in enumerate(per_class_ap):
                    metrics[f"{task}_auprc_class{k}"] = float(ap)

                metrics[f"{task}_rocauc_macroavg_ovo"] = roc_auc_score(
                    y_oh, scores, average="macro", multi_class="ovo"
                )  # OvO macro-average — insensitive to class imbalance, unlike OvR macro
                per_class_roc = roc_auc_score(y_oh, scores, average=None)
                for k, roc in enumerate(per_class_roc):
                    metrics[f"{task}_rocauc_class{k}"] = float(roc)

            else:
                if ((y == 0) | (y == 1)).all():
                    y_true  = y.numpy()
                    y_score = torch.sigmoid(z).numpy()
                    metrics[f"{task}_auprc"]  = average_precision_score(y_true, y_score)
                    metrics[f"{task}_rocauc"] = roc_auc_score(y_true, y_score)
                else:
                    metrics[f"{task}_mae"] = torch.mean((z - y.float()).abs()).item()
        return metrics

    def save_checkpoint(self, current_auroc: float) -> None:
        prefix = "last"
        if current_auroc > self.best_auroc:
            self.best_auroc = current_auroc
            prefix = "best"
        ckpt_path = os.path.join(self.args.save_dir, f"{prefix}_checkpoint.pt")
        torch.save(
            {
                "model_state_dict": unwrap(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "args": vars(self.args),
                "step": self.step,
                "epoch": self.epoch,
            },
            ckpt_path,
        )

        weights_dir = os.path.join(self.args.save_dir, f"{prefix}_weights")
        os.makedirs(weights_dir, exist_ok=True)

        # Per-task head weights — same shape/meaning across every adapter type, so
        # evaluation scripts never need to know which adapter produced a checkpoint.
        for task, head in unwrap(self.model).heads.items():
            np.save(os.path.join(weights_dir, f"{task}.npy"),
                    head.weight.detach().cpu().float().numpy())

        adapter_type = unwrap(self.model).adapter_type
        if adapter_type == "mlp":
            torch.save(unwrap(self.model).adapter.state_dict(),
                      os.path.join(weights_dir, "adapter.pt"))
        elif adapter_type == "attn":
            torch.save(unwrap(self.model).attn.state_dict(),
                      os.path.join(weights_dir, "attn.pt"))

        print(f"=> step: {self.step}, {prefix} model saved: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_filter", type=str, default=None, help="Filter to a single dataset, e.g. '1' for MIMIC-CXR only. None = all datasets in the CSV.")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--csv_filepath", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory with cached CLS embeddings (see extract_cls_embeddings.py). None = train from raw images.")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--parents", nargs="+", type=str, default=list(CLASS_SCHEMA), help="Which tasks to train heads for.")
    parser.add_argument("--img_resolution", type=int, default=512)
    parser.add_argument("--img_channels", type=int, default=1, help="Set to -1 to train from --cache_dir instead of raw images.")
    parser.add_argument("--patient_ids_file", type=str, default=None,
                    help="Optional .npy file of patient IDs for a training-variance bootstrap draw (see generate_bootstrap_indices.py). None = use all patients.")
    # MODEL
    parser.add_argument("--adapter_type", type=str, default="none", choices=["none", "mlp", "attn"])
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--layer_ids", type=int, nargs="+", default=[3, 6, 9, 12],
                        help="Transformer block layers to pool over (only used when --adapter_type=attn).")
    # TRAIN
    parser.add_argument("--exp_name", type=str, default="smoke")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup", type=int, default=2000)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--ema_rate", type=float, default=0.9999)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--determ", action="store_true", default=False)
    parser.add_argument("--dist", action="store_true", default=False)
    parser.add_argument("--wandb_mode", default="disabled", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    device, rank, world_size = (
        setup_distributed()
        if args.dist
        else (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 0, 1)
    )
    is_dist = args.dist and dist.is_available() and dist.is_initialized()
    seed_all(args.seed, args.determ)

    datasets = get_cxr1m(args)
    dataloaders = get_dataloaders(args, datasets)

    tasks = {k: 1 for k in args.parents}  # binary by default
    for k, v in dict(dataset=7, race=3, view=3).items():  # override for categorical tasks
        if k in tasks:
            tasks[k] = v

    model = MultiHeadPredictor(
        tasks,
        adapter_type=args.adapter_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        head_dim=args.head_dim,
        layer_ids=args.layer_ids,
    )
    model.to(device)
    ema = ModelEMA(model.parameters(), rate=args.ema_rate)
    if is_dist:
        model = DistributedDataParallel(model, device_ids=[device])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=tuple(args.betas),
        eps=args.eps,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / args.lr_warmup, total_iters=args.lr_warmup
    )

    if rank == 0:
        wandb.init(project="adapter-subgroup-analysis", name=args.exp_name, config=vars(args), mode=args.wandb_mode)
        for k, v in vars(args).items():
            print(f"--{k}={v}")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"#params: {num_params:,}")

    trainer = Trainer(model, args, optimizer=optimizer, scheduler=scheduler, ema=ema)

    for i in range(args.epochs):
        if is_dist:
            dataloaders["train"].sampler.set_epoch(i)
        now = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime())
        print(f"\n{now}, Epoch {i+1}:")
        train_loss = trainer.train_epoch(dataloaders)
        if rank == 0:
            wandb.log({"train_loss": train_loss}, trainer.step)

    if rank == 0:
        trainer.save_checkpoint(-1.0)  # always save the final-epoch state, regardless of best_auroc
        print(f"Final checkpoint saved at step {trainer.step}, train_loss={train_loss:.5f}")

    if is_dist:
        dist.destroy_process_group()
