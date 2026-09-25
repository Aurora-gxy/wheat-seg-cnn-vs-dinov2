"""Segmentation metrics for the 2-class (soil / plant) problem.

Two ways of averaging are reported:
  * dataset-level ("micro"): sum the confusion matrix over every pixel of every image,
    then compute the scores once. This is the standard way to report mIoU and is our main number.
  * image-level mean: compute the scores for each image, then average over images.
    Useful for seeing the spread (best / worst images).
"""
from typing import Dict, List

import numpy as np

from .config import CLASS_NAMES, PLANT, SOIL


def confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """2x2 matrix: rows = ground truth class, columns = predicted class."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    idx = gt.astype(np.int64).ravel() * 2 + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=4).reshape(2, 2)


def _div(a: float, b: float, empty: float) -> float:
    return float(a) / float(b) if b > 0 else empty


def scores_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    """Precision / recall / F1 / IoU for each class, plus mIoU, mean F1 and pixel accuracy."""
    out = {}
    for c in (PLANT, SOIL):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp          # predicted c, truth is the other class
        fn = cm[c, :].sum() - tp          # truth c, predicted the other class
        absent = (tp + fn) == 0           # class not present in ground truth
        # If a class is absent and never predicted, count it as perfect instead of 0/0.
        precision = _div(tp, tp + fp, 1.0 if absent else 0.0)
        recall = _div(tp, tp + fn, 1.0)
        f1 = _div(2 * tp, 2 * tp + fp + fn, 1.0)
        iou = _div(tp, tp + fp + fn, 1.0)
        name = CLASS_NAMES[c]
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
        out[f"{name}_iou"] = iou
    out["miou"] = (out["plant_iou"] + out["soil_iou"]) / 2
    out["mf1"] = (out["plant_f1"] + out["soil_f1"]) / 2
    out["pixel_acc"] = _div(np.trace(cm), cm.sum(), 1.0)
    return out


class SegMetrics:
    """Accumulates predictions image by image.

        m = SegMetrics()
        for ...: m.update(name, pred_mask, gt_mask)
        m.summary()      -> dict of dataset-level scores (+ image-mean mIoU)
        m.per_image      -> list of dicts, one per image
    """

    def __init__(self):
        self.cm = np.zeros((2, 2), dtype=np.int64)
        self.per_image: List[Dict] = []

    def update(self, name: str, pred: np.ndarray, gt: np.ndarray, **extra) -> Dict:
        cm = confusion_matrix(pred, gt)
        self.cm += cm
        row = {"image": name, **scores_from_confusion(cm), **extra}
        self.per_image.append(row)
        return row

    def summary(self) -> Dict[str, float]:
        out = scores_from_confusion(self.cm)
        if self.per_image:
            out["image_mean_miou"] = float(np.mean([r["miou"] for r in self.per_image]))
            out["image_std_miou"] = float(np.std([r["miou"] for r in self.per_image]))
        out["n_images"] = len(self.per_image)
        return out
