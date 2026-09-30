"""Image distortions that simulate difficult field conditions.

Two uses:
  1. Robustness test (step 4): every test image is corrupted in 6 ways. The corruption is
     deterministic (seeded by scenario + image name), so every model sees exactly the same
     corrupted images and the comparison is fair.
  2. Optional training augmentation ("distort"): the same corruptions applied at random during
     training, to test whether training on corrupted images makes a model more robust.

The 6 scenarios follow the COMP9517 group project, so results are comparable in spirit.
"""
import zlib
from typing import Callable, Dict

import cv2
import numpy as np


# ---------------------------------------------------------------- basic corruptions
def gaussian_noise(img: np.ndarray, rng: np.random.Generator, sigma: float = 20.0) -> np.ndarray:
    """Sensor noise: add N(0, sigma^2) to every pixel (0-255 scale)."""
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, rng: np.random.Generator, kernel: int = 5) -> np.ndarray:
    """Out-of-focus or motion: Gaussian blur with a k x k kernel."""
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def low_brightness_contrast(img: np.ndarray, rng: np.random.Generator, factor: float = 0.6) -> np.ndarray:
    """Overcast / dusk: reduce contrast around the mean, then reduce brightness, both by `factor`."""
    x = img.astype(np.float32)
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = ((x - mean) * factor + mean) * factor
    return np.clip(x, 0, 255).astype(np.uint8)


def occlusion(img: np.ndarray, rng: np.random.Generator, ratio: float = 0.15) -> np.ndarray:
    """Shadows, debris, equipment in view: a black rectangle covering `ratio` of the image."""
    h, w = img.shape[:2]
    area = ratio * h * w
    aspect = rng.uniform(0.5, 2.0)
    bh = int(min(h, round(np.sqrt(area / aspect))))
    bw = int(min(w, round(area / max(bh, 1))))
    y0 = int(rng.integers(0, h - bh + 1))
    x0 = int(rng.integers(0, w - bw + 1))
    out = img.copy()
    out[y0:y0 + bh, x0:x0 + bw] = 0
    return out


# ---------------------------------------------------------------- test scenarios
SCENARIOS: Dict[str, Callable] = {
    "gaussian_noise": lambda im, r: gaussian_noise(im, r),
    "gaussian_blur": lambda im, r: gaussian_blur(im, r),
    "low_light": lambda im, r: low_brightness_contrast(im, r),
    "occlusion": lambda im, r: occlusion(im, r),
    "noise_blur": lambda im, r: gaussian_blur(gaussian_noise(im, r), r),
    "blur_low_light": lambda im, r: low_brightness_contrast(gaussian_blur(im, r), r),
}

SCENARIO_LABELS = {
    "clean": "Clean",
    "gaussian_noise": "Gaussian noise (σ=20)",
    "gaussian_blur": "Gaussian blur (5×5)",
    "low_light": "Low brightness & contrast (×0.6)",
    "occlusion": "Occlusion (15%)",
    "noise_blur": "Noise + blur",
    "blur_low_light": "Blur + low light",
}


class Distortion:
    """Deterministic corruption: the same (scenario, image) always gives the same result."""

    def __init__(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
        self.scenario = scenario

    def __call__(self, image: np.ndarray, key: str = "") -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(f"{self.scenario}/{key}".encode()))
        return SCENARIOS[self.scenario](image, rng)


# ---------------------------------------------------------------- training augmentation
def random_distortion(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply each corruption independently with some probability (mild settings, image only)."""
    if rng.random() < 0.3:
        image = gaussian_noise(image, rng, sigma=float(rng.uniform(5, 25)))
    if rng.random() < 0.2:
        image = gaussian_blur(image, rng, kernel=int(rng.choice([3, 5])))
    if rng.random() < 0.3:
        image = low_brightness_contrast(image, rng, factor=float(rng.uniform(0.55, 0.95)))
    if rng.random() < 0.2:
        image = occlusion(image, rng, ratio=float(rng.uniform(0.05, 0.2)))
    return image
