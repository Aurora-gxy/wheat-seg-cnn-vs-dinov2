"""Model registry: build any model by name, e.g. get_model("exg_otsu")."""
from pathlib import Path

from .base import Segmenter

AVAILABLE_MODELS = ["exg_otsu", "deeplabv3plus"]
TRAINABLE_MODELS = ["deeplabv3plus"]


def get_model(name: str, **kwargs) -> Segmenter:
    name = name.lower()
    if name == "exg_otsu":
        from .classical import ExGOtsu
        return ExGOtsu(**kwargs)
    if name == "deeplabv3plus":
        from .cnn import DeepLabV3Plus
        return DeepLabV3Plus(**kwargs)
    # Added in step 3: "dinov2_linear", "dinov2_decoder"
    raise ValueError(f"Unknown model: {name!r}")


def load_checkpoint(path) -> Segmenter:
    """Rebuild a trained PyTorch model from a .pth file saved by TorchSegmenter.save()."""
    import torch

    from .torch_base import TrainConfig

    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = get_model(ckpt["model"], train_cfg=TrainConfig(**ckpt["train_cfg"]), **ckpt["model_kwargs"])
    model.load_state(ckpt)
    return model
