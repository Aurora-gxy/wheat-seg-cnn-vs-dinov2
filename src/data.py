"""EWS dataset loading.

Mask format (checked on the raw files):
    each *_mask.png is a greyscale+alpha PNG ("LA" mode).
    In the grey channel, plant pixels are black (0) and soil pixels are white (255),
    with a few anti-aliased values in between along leaf edges.
    We convert it to:  PLANT = 1 where grey < 128,  SOIL = 0 elsewhere.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import config


@dataclass(frozen=True)
class Sample:
    name: str
    image_path: Path
    mask_path: Path


# ---------------------------------------------------------------- file listing
def list_samples(split: str, fraction: float = 1.0, seed: int = config.DEFAULT_SEED,
                 data_dir: Path = config.DATA_DIR) -> List[Sample]:
    """Return the (image, mask) pairs of one split: 'train', 'val' or 'test'.

    fraction < 1.0 returns a random subset (used for the label-efficiency experiment).
    Subsets are nested: with the same seed, the 10% subset is contained in the 25% subset, etc.
    """
    if split not in config.SPLITS:
        raise ValueError(f"split must be one of {list(config.SPLITS)}, got {split!r}")
    folder = Path(data_dir) / config.SPLITS[split]
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    samples = []
    for img_path in sorted(folder.glob("*.png")):
        if img_path.stem.endswith(config.MASK_SUFFIX):
            continue
        mask_path = img_path.with_name(img_path.stem + config.MASK_SUFFIX + ".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path.name}")
        samples.append(Sample(img_path.stem, img_path, mask_path))

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction < 1.0:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(samples))
        k = max(1, int(round(fraction * len(samples))))
        samples = [samples[i] for i in sorted(order[:k])]
    return samples


# ---------------------------------------------------------------- loading
def load_image(path: Path) -> np.ndarray:
    """RGB image as uint8 array of shape (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    """Binary mask as uint8 array of shape (H, W): 1 = plant, 0 = soil."""
    grey = np.array(Image.open(path).convert("L"))
    return (grey < 128).astype(np.uint8)


def load_sample(sample: Sample) -> Tuple[np.ndarray, np.ndarray]:
    return load_image(sample.image_path), load_mask(sample.mask_path)


# ---------------------------------------------------------------- augmentation
def basic_augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Random horizontal/vertical flip and 90-degree rotation (applied to image and mask)."""
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    k = int(rng.integers(0, 4))
    if k:
        image, mask = np.rot90(image, k), np.rot90(mask, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


# ---------------------------------------------------------------- tensors
def normalize(image: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW, ImageNet-normalised."""
    x = image.astype(np.float32) / 255.0
    x = (x - np.array(config.IMAGENET_MEAN, np.float32)) / np.array(config.IMAGENET_STD, np.float32)
    return x.transpose(2, 0, 1)


def pad_to_multiple(x, multiple: int, value: float = 0.0):
    """Pad a torch tensor (..., H, W) on the bottom/right so H and W are multiples of `multiple`.

    Returns (padded_tensor, (H, W)) so the prediction can be cropped back afterwards.
    DeepLabV3+ needs multiples of 16, DINOv2 needs multiples of 14.
    """
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=value)
    return x, (h, w)


class EWSDataset:
    """PyTorch dataset. Each item is a dict with 'image' (3,H,W float), 'mask' (H,W long), 'name'."""

    def __init__(self, samples: List[Sample], augment: Optional[Callable] = None,
                 seed: int = config.DEFAULT_SEED, cache: bool = True):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        # 190 images of 350x350 is tiny (~70 MB), so keep them in memory
        self._cache = [load_sample(s) for s in samples] if cache else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch

        image, mask = self._cache[idx] if self._cache else load_sample(self.samples[idx])
        if self.augment is not None:
            image, mask = self.augment(image, mask, self.rng)
        return {
            "image": torch.from_numpy(normalize(image)),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "name": self.samples[idx].name,
        }
