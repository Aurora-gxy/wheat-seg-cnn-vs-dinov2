"""Excess-Green + Otsu threshold: a training-free baseline.

ExG = 2g - r - b on chromatic coordinates (r, g, b each divided by r + g + b).
Green plants have high ExG, soil has low ExG; Otsu picks the threshold automatically per image.
It is mainly used to test the pipeline end to end, and as the "no learning" reference point.
"""
import cv2
import numpy as np

from .base import Segmenter


class ExGOtsu(Segmenter):
    name = "exg_otsu"

    def __init__(self, blur_kernel: int = 5, min_component: int = 32):
        self.blur_kernel = blur_kernel
        self.min_component = min_component

    def predict(self, image: np.ndarray) -> np.ndarray:
        rgb = image.astype(np.float32)
        total = rgb.sum(axis=2) + 1e-6
        r, g, b = (rgb[..., i] / total for i in range(3))
        exg = 2 * g - r - b                                   # roughly in [-1, 2]

        exg8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.blur_kernel > 1:
            exg8 = cv2.GaussianBlur(exg8, (self.blur_kernel, self.blur_kernel), 0)
        _, mask = cv2.threshold(exg8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.min_component > 0:                            # drop tiny speckles
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            small = np.where(stats[:, cv2.CC_STAT_AREA] < self.min_component)[0]
            small = small[small != 0]
            mask[np.isin(labels, small)] = 0
        return mask.astype(np.uint8)
