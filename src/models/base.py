"""Common interface that every segmentation model in this project implements."""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..data import Sample


class Segmenter(ABC):
    """Base class for all models (classical, CNN, DINOv2).

    fit(train, val)   -> train the model (no-op for methods that need no training)
    predict(image)    -> binary mask
    """

    name: str = "base"

    def fit(self, train_samples: List[Sample], val_samples: Optional[List[Sample]] = None) -> None:
        """Default: nothing to train."""
        return None

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: uint8 RGB array (H, W, 3)  ->  uint8 mask (H, W) with 1 = plant, 0 = soil."""

    def num_parameters(self) -> int:
        """Number of model parameters (0 for non-neural methods)."""
        return 0

    def num_trainable_parameters(self) -> int:
        return 0
