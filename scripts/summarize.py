"""Collect everything in results/metrics/summary.csv into the tables used in the report.

Run from the project root:
    python scripts/summarize.py

Outputs (results/tables/):
    main_comparison.csv    experiment A + D: accuracy (mean ± std over seeds), speed, size
    label_efficiency.csv   experiment C: test mIoU vs share of training images
    robustness.csv         experiment B: test mIoU per distortion (seed 42 models)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config

NUMERIC = ["miou", "plant_iou", "mf1", "plant_precision", "plant_recall", "time_ms_per_image",
           "params_total", "params_trainable", "train_time_min", "fraction", "seed", "best_val_miou"]
MODEL_ORDER = ["exg_otsu", "dinov2_linear", "deeplabv3plus", "dinov2_decoder"]


def load() -> pd.DataFrame:
    path = config.METRICS_DIR / "summary.csv"
    if not path.exists():
        sys.exit("results/metrics/summary.csv not found - run some experiments first.")
    df = pd.read_csv(path, keep_default_na=False)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("augment", "equal_steps"):
        if c not in df:
            df[c] = ""
    df = df[~df["tag"].astype(str).str.contains("quick|sanity")]
    df = df[df["split"] == "test"]
    # ExG needs no training: treat it as a full-data, basic run
    exg = df["model"] == "exg_otsu"
    df.loc[exg, "fraction"] = 1.0
    df.loc[exg, "augment"] = "basic"
    df.loc[exg, "seed"] = 42
    return df


def order(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return df.sort_values(col, key=lambda s: s.map(lambda m: rank.get(m, 99)))


def pct(mean, std, n):
    return f"{mean * 100:.1f} ± {std * 100:.1f}" if n > 1 else f"{mean * 100:.1f}"


def main():
    df = load()
    out = config.RESULTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    clean = df[df["scenario"] == "clean"]

    # ---------- A + D: main comparison (full data, basic augmentation, all seeds)
    full = clean[(clean["fraction"] == 1.0) & (clean["augment"] == "basic")]
    rows = []
    for model, g in full.groupby("model"):
        n = len(g)
        rows.append({
            "model": model, "n_seeds": n,
            "test_mIoU_%": pct(g["miou"].mean(), g["miou"].std(ddof=1) if n > 1 else 0, n),
            "plant_IoU_%": pct(g["plant_iou"].mean(), g["plant_iou"].std(ddof=1) if n > 1 else 0, n),
            "mF1_%": pct(g["mf1"].mean(), g["mf1"].std(ddof=1) if n > 1 else 0, n),
            "trainable_params_M": round(g["params_trainable"].mean() / 1e6, 3),
            "total_params_M": round(g["params_total"].mean() / 1e6, 1),
            "train_time_min": round(g["train_time_min"].mean(), 1) if g["train_time_min"].notna().any() else 0,
            "ms_per_image": round(g["time_ms_per_image"].mean(), 1),
        })
    main_tab = order(pd.DataFrame(rows))
    main_tab.to_csv(out / "main_comparison.csv", index=False)
    print("\n=== A + D. Main comparison (test set, full training data) ===")
    print(main_tab.to_string(index=False))

    # ---------- C: label efficiency (seed 42; equal-steps runs for fractions < 1)
    lab = clean[(clean["augment"] == "basic") & (clean["seed"] == 42) & (clean["model"] != "exg_otsu")]
    lab = lab[(lab["fraction"] == 1.0) | lab["tag"].astype(str).str.endswith("_eq")]
    if lab["fraction"].nunique() > 1:
        tab = lab.pivot_table(index="model", columns="fraction", values="miou", aggfunc="mean") * 100
        tab.columns = [f"{int(round(c * 100))}%" for c in tab.columns]
        tab = tab.loc[[m for m in MODEL_ORDER if m in tab.index]].round(1)
        tab.to_csv(out / "label_efficiency.csv")
        print("\n=== C. Label efficiency: test mIoU (%) vs share of training images ===")
        print(tab.to_string())
    else:
        print("\n(C. Label efficiency: no reduced-data runs yet - run_experiments.py --plan label)")

    # ---------- B: robustness (seed 42, full data; basic and distortion-trained models)
    rob = df[(df["seed"] == 42) & (df["fraction"] == 1.0)]
    if rob["scenario"].nunique() > 1:
        rob = rob.assign(run=rob["model"] + rob["augment"].map(lambda a: " (+distort train)" if a == "distort" else ""))
        tab = rob.pivot_table(index="run", columns="scenario", values="miou", aggfunc="mean") * 100
        cols = ["clean"] + [c for c in ["gaussian_noise", "gaussian_blur", "low_light", "occlusion",
                                        "noise_blur", "blur_low_light"] if c in tab.columns]
        tab = tab[cols]
        tab["mean_distorted"] = tab[cols[1:]].mean(axis=1)
        tab["drop_vs_clean"] = tab["mean_distorted"] - tab["clean"]
        tab = tab.round(1)
        tab.to_csv(out / "robustness.csv")
        print("\n=== B. Robustness: test mIoU (%) under distortions ===")
        print(tab.to_string())
    else:
        print("\n(B. Robustness: not run yet - python scripts/robustness.py)")

    print(f"\nTables saved to {out.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
