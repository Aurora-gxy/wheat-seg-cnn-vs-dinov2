"""Model registry: build any model by name, e.g. get_model("exg_otsu")."""
from .base import Segmenter


def get_model(name: str, **kwargs) -> Segmenter:
    name = name.lower()
    if name == "exg_otsu":
        from .classical import ExGOtsu
        return ExGOtsu(**kwargs)
    # Added in later steps:
    #   "deeplabv3plus"  -> CNN baseline (step 2)
    #   "dinov2_linear", "dinov2_decoder" -> foundation model (step 3)
    raise ValueError(f"Unknown model: {name!r}")


AVAILABLE_MODELS = ["exg_otsu"]
