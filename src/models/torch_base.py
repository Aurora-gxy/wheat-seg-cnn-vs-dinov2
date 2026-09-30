"""Shared training / inference code for every PyTorch model (DeepLabV3+ now, DINOv2 in step 3).

A subclass only has to:
  * set `name` and `pad_multiple`
  * implement `build_net()` returning an nn.Module that maps
        normalised images (B, 3, H, W)  ->  logits (B, 2, H, W)
Everything else (training loop, validation, checkpointing, prediction) lives here,
so both models are trained and evaluated in exactly the same way.
"""
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import EWSDataset, Sample, basic_augment, normalize, pad_to_multiple
from ..metrics import scores_from_confusion
from ..utils import get_device, set_seed
from .base import Segmenter

def distort_augment(image, mask, rng):
    """Basic flips/rotations, plus random noise / blur / low light / occlusion on the image only."""
    from ..distortions import random_distortion

    image, mask = basic_augment(image, mask, rng)
    return random_distortion(image, rng), mask


AUGMENTATIONS = {"none": None, "basic": basic_augment, "distort": distort_augment}


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    loss: str = "ce"             # "ce", "dice" or "ce_dice"
    augment: str = "basic"       # key of AUGMENTATIONS
    val_every: int = 1           # validate every N epochs
    patience: Optional[int] = None   # stop if val mIoU does not improve for N validations
    seed: int = 42
    eval_batch_size: int = 8


# ---------------------------------------------------------------- losses
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss averaged over the two classes. Less sensitive to class imbalance than CE."""
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    card = probs.sum(dims) + onehot.sum(dims)
    return 1 - ((2 * inter + eps) / (card + eps)).mean()


def build_loss(name: str):
    if name == "ce":
        return lambda logits, y: F.cross_entropy(logits, y)
    if name == "dice":
        return dice_loss
    if name == "ce_dice":
        return lambda logits, y: F.cross_entropy(logits, y) + dice_loss(logits, y)
    raise ValueError(f"Unknown loss {name!r}")


def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------- base class
class TorchSegmenter(Segmenter):
    name = "torch_base"
    pad_multiple = 1

    def __init__(self, train_cfg: Optional[TrainConfig] = None, device=None):
        self.cfg = train_cfg or TrainConfig()
        self.device = device or get_device()
        self.net: nn.Module = self.build_net().to(self.device)
        self.history: List[Dict] = []
        self.best_val_miou: Optional[float] = None
        self.train_time_s: Optional[float] = None

    # ---- to implement in subclasses
    def build_net(self) -> nn.Module:
        raise NotImplementedError

    def model_kwargs(self) -> Dict:
        """Constructor arguments needed to rebuild this model from a checkpoint."""
        return {}

    def set_train_mode(self) -> None:
        """Override if part of the network must stay in eval mode (e.g. a frozen backbone)."""
        self.net.train()

    # ---- parameters
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    # ---- forward with automatic padding
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: normalised (B, 3, H, W) -> logits (B, 2, H, W), padding handled internally."""
        x, (h, w) = pad_to_multiple(x, self.pad_multiple)
        return self.net(x)[..., :h, :w]

    # ---- training
    def fit(self, train_samples: List[Sample], val_samples: Optional[List[Sample]] = None) -> None:
        cfg = self.cfg
        set_seed(cfg.seed)
        train_ds = EWSDataset(train_samples, augment=AUGMENTATIONS[cfg.augment], seed=cfg.seed)
        loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
            drop_last=len(train_ds) > 1 and len(train_ds) % cfg.batch_size == 1,   # no batch of 1 (BatchNorm)
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        val_ds = EWSDataset(val_samples) if val_samples else None

        params = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_steps = cfg.epochs * len(loader)
        scheduler = _cosine_with_warmup(optimizer, cfg.warmup_epochs * len(loader), total_steps)
        loss_fn = build_loss(cfg.loss)

        print(f"Training {self.name} on {self.device} | {len(train_ds)} train images, "
              f"{len(val_ds) if val_ds else 0} val images | {len(loader)} steps/epoch x {cfg.epochs} epochs")
        print(f"Parameters: {self.num_parameters() / 1e6:.1f} M total, "
              f"{self.num_trainable_parameters() / 1e6:.2f} M trainable")

        best_miou, best_state, bad_rounds = -1.0, None, 0
        start = time.perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            self.set_train_mode()
            losses = []
            for step, batch in enumerate(loader, 1):
                print(f"\repoch {epoch:3d}/{cfg.epochs} | step {step}/{len(loader)} ...", end="", flush=True)
                x = batch["image"].to(self.device)
                y = batch["mask"].to(self.device)
                loss = loss_fn(self.logits(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())

            row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   "lr": optimizer.param_groups[0]["lr"]}
            if val_ds is not None and (epoch % cfg.val_every == 0 or epoch == cfg.epochs):
                print(f"\repoch {epoch:3d}/{cfg.epochs} | validating ...          ", end="", flush=True)
                val = self.validate(val_ds)
                self._empty_cache()
                row.update({"val_miou": val["miou"], "val_plant_iou": val["plant_iou"]})
                if val["miou"] > best_miou:
                    best_miou, bad_rounds = val["miou"], 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    row["best"] = True
                else:
                    bad_rounds += 1
            row["epoch_time_s"] = time.perf_counter() - t0
            self.history.append(row)

            msg = f"epoch {epoch:3d}/{cfg.epochs} | loss {row['train_loss']:.4f}"
            if "val_miou" in row:
                msg += f" | val mIoU {row['val_miou'] * 100:5.2f}%" + (" *" if row.get("best") else "")
            msg += f" | {row['epoch_time_s']:.1f}s"
            print("\r" + msg + " " * 10, flush=True)

            if cfg.patience is not None and bad_rounds >= cfg.patience:
                print(f"Early stopping: no improvement for {cfg.patience} validations.")
                break

        self.train_time_s = time.perf_counter() - start
        if best_state is not None:                 # keep the weights with the best val mIoU
            self.net.load_state_dict(best_state)
            self.best_val_miou = best_miou
            print(f"Best val mIoU {best_miou * 100:.2f}% (restored those weights)")
        print(f"Training time: {self.train_time_s / 60:.1f} min")

    def _empty_cache(self) -> None:
        """Release cached GPU memory (helps on Macs with little unified memory)."""
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def validate(self, dataset: EWSDataset) -> Dict[str, float]:
        self.net.eval()
        cm = np.zeros((2, 2), dtype=np.int64)
        loader = DataLoader(dataset, batch_size=self.cfg.eval_batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            pred = self.logits(batch["image"].to(self.device)).argmax(1).cpu().numpy()
            gt = batch["mask"].numpy()
            idx = gt.ravel() * 2 + pred.ravel()
            cm += np.bincount(idx, minlength=4).reshape(2, 2)
        return scores_from_confusion(cm)

    # ---- inference
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = torch.from_numpy(normalize(image)).unsqueeze(0).to(self.device)
        return self.logits(x).argmax(1)[0].cpu().numpy().astype(np.uint8)

    # ---- checkpoints
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.name,
            "model_kwargs": self.model_kwargs(),
            "train_cfg": asdict(self.cfg),
            "state_dict": self.net.state_dict(),
            "history": self.history,
            "best_val_miou": self.best_val_miou,
            "train_time_s": self.train_time_s,
        }, path)

    def load_state(self, ckpt: Dict) -> None:
        self.net.load_state_dict(ckpt["state_dict"])
        self.history = ckpt.get("history", [])
        self.best_val_miou = ckpt.get("best_val_miou")
        self.train_time_s = ckpt.get("train_time_s")
