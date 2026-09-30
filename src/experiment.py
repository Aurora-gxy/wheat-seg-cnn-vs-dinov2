"""Shared helpers for naming and configuring training runs (used by train.py and run_experiments.py)."""
import math

# Batch size per model. The CNN fine-tunes 26.7 M parameters, so a smaller batch keeps memory
# low on laptops; DINOv2 only trains its head, so it can use a larger batch.
DEFAULT_BATCH = {"deeplabv3plus": 4, "dinov2_linear": 8, "dinov2_decoder": 8}


def run_name(model: str, loss: str = "ce", augment: str = "basic", fraction: float = 1.0,
             seed: int = 42, backbone: str = "vits14", equal_steps: bool = False,
             quick: bool = False) -> str:
    name = model
    if model.startswith("dinov2") and backbone != "vits14":
        name += f"_{backbone}"
    name += f"_{loss}_{augment}_frac{round(fraction * 100)}_seed{seed}"
    if equal_steps and fraction < 1:
        name += "_eq"
    if quick:
        name += "_quick"
    return name


def equal_steps_schedule(epochs: int, fraction: float, warmup_epochs: int = 2):
    """With less data an epoch has fewer steps. To give every run the same number of weight
    updates (a fair label-efficiency comparison), scale the epochs by 1 / fraction and validate
    proportionally less often, so early-stopping patience means the same thing."""
    scale = 1.0 / fraction
    return (int(math.ceil(epochs * scale)),          # epochs
            max(1, int(round(scale))),               # val_every
            int(round(warmup_epochs * scale)))       # warmup epochs
