"""Run a model over a list of samples and record accuracy + inference time.

The same function is reused for every experiment:
  * clean test set               -> distortion=None
  * robustness test (step 4)     -> distortion=<function that corrupts the image>
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .data import Sample, load_sample
from .metrics import SegMetrics
from .models.base import Segmenter
from .utils import Timer


def evaluate(model: Segmenter, samples: List[Sample],
             distortion: Optional[Callable[[np.ndarray], np.ndarray]] = None,
             save_pred_dir: Optional[Path] = None,
             verbose: bool = True) -> Tuple[Dict, List[Dict]]:
    """Returns (summary, per_image_rows)."""
    metrics = SegMetrics()
    times = []
    if save_pred_dir is not None:
        Path(save_pred_dir).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        image, gt = load_sample(sample)
        if distortion is not None:
            image = distortion(image)

        with Timer() as t:
            pred = model.predict(image)
        times.append(t.seconds)

        pred = np.asarray(pred, dtype=np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f"{model.name} returned shape {pred.shape}, expected {gt.shape}")
        metrics.update(sample.name, pred, gt, time_ms=t.seconds * 1000)

        if save_pred_dir is not None:
            Image.fromarray(pred * 255).save(Path(save_pred_dir) / f"{sample.name}_pred.png")
        if verbose:
            print(f"\r  [{i + 1}/{len(samples)}] {sample.name}", end="", flush=True)
    if verbose:
        print()

    summary = metrics.summary()
    # First call often includes one-off setup (weights to GPU, kernel compilation) -> exclude it.
    steady = times[1:] if len(times) > 1 else times
    summary["time_ms_per_image"] = float(np.mean(steady) * 1000)
    summary["params_total"] = model.num_parameters()
    summary["params_trainable"] = model.num_trainable_parameters()
    return summary, metrics.per_image


def save_results(summary: Dict, per_image: List[Dict], model_name: str, split: str,
                 scenario: str = "clean", tag: str = "",
                 metrics_dir: Path = config.METRICS_DIR) -> Path:
    """Write per-image CSV and add/replace one row in results/metrics/summary.csv.

    A row is identified by (model, tag, split, scenario), so re-running an experiment
    overwrites its old row instead of creating duplicates.
    """
    metrics_dir = Path(metrics_dir)
    (metrics_dir / "per_image").mkdir(parents=True, exist_ok=True)

    run_id = "_".join(x for x in [model_name, tag, split, scenario] if x)
    pd.DataFrame(per_image).to_csv(metrics_dir / "per_image" / f"{run_id}.csv", index=False)

    key = {"model": model_name, "tag": tag, "split": split, "scenario": scenario}
    row = pd.DataFrame([{**key, **summary}])
    summary_path = metrics_dir / "summary.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path, keep_default_na=False)
        same = np.ones(len(old), dtype=bool)
        for k, v in key.items():
            same &= old[k].astype(str) == str(v)
        row = pd.concat([old[~same], row], ignore_index=True)
    row.to_csv(summary_path, index=False, float_format="%.6f")
    return summary_path


def print_summary(summary: Dict, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    for k in ["miou", "plant_iou", "soil_iou", "mf1", "plant_f1", "plant_precision",
              "plant_recall", "pixel_acc", "image_mean_miou", "image_std_miou"]:
        if k in summary:
            print(f"  {k:<18} {summary[k] * 100:6.2f} %")
    print(f"  {'time_ms_per_image':<18} {summary['time_ms_per_image']:8.2f} ms")
    print(f"  {'n_images':<18} {summary['n_images']}")
