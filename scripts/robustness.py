"""Experiment B: robustness. Evaluate trained models on the test set under 6 distortions.

No training happens here: each model is loaded from its checkpoint and tested on
clean images plus 6 corrupted versions of the same images (identical for every model).

Run from the project root:
    python scripts/robustness.py                      # all *_frac100_seed42 checkpoints + ExG
    python scripts/robustness.py --checkpoints checkpoints/a.pth checkpoints/b.pth

Outputs:
    results/metrics/summary.csv                one row per (model, scenario)
    results/robustness/robustness_miou.csv     table: model x scenario (mIoU %)
    results/robustness/robustness_drop.csv     table: drop vs clean (percentage points)
    results/figures/distortion_examples.png    what the 6 distortions look like
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config
from src.data import list_samples, load_image
from src.distortions import SCENARIO_LABELS, SCENARIOS, Distortion
from src.evaluate import evaluate, save_results
from src.models import get_model, load_checkpoint


def run_meta(tag: str) -> dict:
    """Recover augment / fraction / seed from a run name like 'deeplabv3plus_ce_basic_frac100_seed42'."""
    m = re.search(r"_([a-z]+)_frac(\d+)_seed(\d+)", tag)
    if not m:                                    # ExG + Otsu (no training)
        return {"augment": "basic", "fraction": 1.0, "seed": 42}
    return {"augment": m.group(1), "fraction": int(m.group(2)) / 100, "seed": int(m.group(3))}


def clean_row_exists(model_name: str, tag: str) -> bool:
    path = config.METRICS_DIR / "summary.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path, keep_default_na=False)
    hit = ((df["model"] == model_name) & (df["tag"].astype(str) == tag)
           & (df["split"] == "test") & (df["scenario"] == "clean"))
    return bool(hit.any())


def save_examples(sample, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = load_image(sample.image_path)
    names = ["clean"] + list(SCENARIOS)
    fig, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 2.6), dpi=150)
    for ax, name in zip(axes, names):
        shown = image if name == "clean" else Distortion(name)(image, sample.name)
        ax.imshow(shown)
        ax.set_title(SCENARIO_LABELS[name], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", type=Path, default=None,
                   help="default: every checkpoints/*_frac100_seed42.pth")
    p.add_argument("--no-exg", action="store_true", help="skip the ExG + Otsu baseline")
    args = p.parse_args()

    ckpts = args.checkpoints
    if ckpts is None:
        ckpts = sorted(config.CHECKPOINT_DIR.glob("*_frac100_seed42.pth"))
    if not ckpts and args.no_exg:
        sys.exit("No checkpoints found. Train models first with scripts/train.py")

    test = list_samples("test")
    models = [] if args.no_exg else [("exg_otsu", "", lambda: get_model("exg_otsu"))]
    models += [(None, c.stem, (lambda c=c: load_checkpoint(c))) for c in ckpts]

    rows = []
    for _, tag, build in models:
        model = build()
        label = tag or model.name
        print(f"\n=== {label} ===")
        for scenario in ["clean"] + list(SCENARIOS):
            distortion = None if scenario == "clean" else Distortion(scenario)
            summary, per_image = evaluate(model, test, distortion=distortion, verbose=False)
            summary.update(run_meta(tag))
            # keep the clean-test row written by train.py (it also stores training time etc.)
            if not (scenario == "clean" and clean_row_exists(model.name, tag)):
                save_results(summary, per_image, model.name, "test", scenario=scenario, tag=tag)
            rows.append({"run": label, "model": model.name, "scenario": scenario,
                         "miou": summary["miou"] * 100, "plant_iou": summary["plant_iou"] * 100})
            print(f"  {SCENARIO_LABELS[scenario]:<34} mIoU {summary['miou'] * 100:6.2f} %")

    df = pd.DataFrame(rows)
    order = ["clean"] + list(SCENARIOS)
    miou = df.pivot(index="run", columns="scenario", values="miou")[order]
    miou["mean_distorted"] = miou[list(SCENARIOS)].mean(axis=1)
    drop = miou[list(SCENARIOS) + ["mean_distorted"]].sub(miou["clean"], axis=0)

    out = config.RESULTS_DIR / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    miou.round(2).to_csv(out / "robustness_miou.csv")
    drop.round(2).to_csv(out / "robustness_drop.csv")
    save_examples(test[0], config.FIGURES_DIR / "distortion_examples.png")

    print("\nmIoU (%) per scenario:")
    print(miou.round(1).to_string())
    print("\nChange vs clean (percentage points):")
    print(drop.round(1).to_string())
    print(f"\nSaved tables to {out.relative_to(config.ROOT)}/ and "
          f"example images to results/figures/distortion_examples.png")


if __name__ == "__main__":
    main()
