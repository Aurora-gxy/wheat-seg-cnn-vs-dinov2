"""Model registry: build any model by name, e.g. get_model("exg_otsu")."""
from pathlib import Path

from .base import Segmenter

AVAILABLE_MODELS = ["exg_otsu", "deeplabv3plus", "dinov2_linear", "dinov2_decoder"]
TRAINABLE_MODELS = ["deeplabv3plus", "dinov2_linear", "dinov2_decoder"]

# Default learning rates: the whole CNN is fine-tuned (small LR);
# for DINOv2 only a small head is trained from scratch (larger LR).
DEFAULT_LR = {"deeplabv3plus": 2e-4, "dinov2_linear": 1e-3, "dinov2_decoder": 1e-3}


def get_model(name: str, **kwargs) -> Segmenter:
    name = name.lower()
    if name == "exg_otsu":
        from .classical import ExGOtsu
        return ExGOtsu(**kwargs)
    if name == "deeplabv3plus":
        from .cnn import DeepLabV3Plus
        return DeepLabV3Plus(**kwargs)
    if name in ("dinov2_linear", "dinov2_decoder"):
        from .dinov2 import DINOv2Segmenter
        kwargs.setdefault("head", name.split("_", 1)[1])
        return DINOv2Segmenter(**kwargs)
    raise ValueError(f"Unknown model: {name!r}")


def load_checkpoint(path) -> Segmenter:
    """Rebuild a trained PyTorch model from a .pth file saved by TorchSegmenter.save()."""
    import torch

    from .torch_base import TrainConfig

    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = get_model(ckpt["model"], train_cfg=TrainConfig(**ckpt["train_cfg"]), **ckpt["model_kwargs"])
    model.load_state(ckpt)
    return model
