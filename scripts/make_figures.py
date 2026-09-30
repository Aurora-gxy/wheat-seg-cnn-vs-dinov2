"""Step 5: turn the results into report / README / slide figures.

Run from the project root (after run_experiments.py, robustness.py):
    python scripts/make_figures.py                 # all figures
    python scripts/make_figures.py --no-predictions  # skip the one that needs the models

Outputs (results/figures/):
    fig1_accuracy.png           test mIoU per model (mean ± std over seeds)
    fig2_robustness.png         mIoU under 6 distortions, standard vs distortion training
    fig3_label_efficiency.png   mIoU vs share of labelled training images
    fig4_accuracy_vs_cost.png   mIoU vs trainable parameters and training time
    fig5_predictions.png        example predictions with errors highlighted (needs checkpoints)
    fig6_training_curves.png    validation mIoU during training
and results/tables/per_image_comparison.csv (per-image test mIoU of every model).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src import config

FIG = config.FIGURES_DIR
TABLES = config.RESULTS_DIR / "tables"

# ---------------------------------------------------------------- style
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e1e0d9", "#ffffff"
# fixed colour per model (colour follows the model in every figure)
MODELS = ["exg_otsu", "dinov2_linear", "deeplabv3plus", "dinov2_decoder"]
COLOR = {"exg_otsu": "#8f8d86", "dinov2_linear": "#1baf7a",
         "deeplabv3plus": "#2a78d6", "dinov2_decoder": "#eb6834"}
LABEL = {"exg_otsu": "ExG + Otsu (no training)", "dinov2_linear": "DINOv2 + linear head",
         "deeplabv3plus": "DeepLabV3+ (CNN)", "dinov2_decoder": "DINOv2 + light decoder"}
SHORT = {"exg_otsu": "ExG + Otsu", "dinov2_linear": "DINOv2 linear",
         "deeplabv3plus": "DeepLabV3+", "dinov2_decoder": "DINOv2 decoder"}
SCENARIOS = ["clean", "gaussian_noise", "gaussian_blur", "low_light", "occlusion",
             "noise_blur", "blur_low_light"]
SCEN_LABEL = {"clean": "Clean", "gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur",
              "low_light": "Low light", "occlusion": "Occlusion 15%", "noise_blur": "Noise + blur",
              "blur_low_light": "Blur + low light"}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titlepad": 24,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def clean_axes(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  saved results/figures/{name}")


# ---------------------------------------------------------------- data
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(config.METRICS_DIR / "summary.csv", keep_default_na=False)
    for c in ["miou", "plant_iou", "params_trainable", "params_total", "train_time_min",
              "time_ms_per_image", "fraction", "seed"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["split"] == "test") & ~df["tag"].astype(str).str.contains("quick|sanity")].copy()
    exg = df["model"] == "exg_otsu"
    df.loc[exg, ["fraction", "seed"]] = [1.0, 42]
    df.loc[exg, "augment"] = "basic"
    df.loc[exg, ["params_trainable", "train_time_min"]] = 0.0
    df["miou"] *= 100
    return df


def main_table(df):
    full = df[(df["scenario"] == "clean") & (df["fraction"] == 1.0) & (df["augment"] == "basic")]
    g = full.groupby("model")
    out = pd.DataFrame({"mean": g["miou"].mean(), "std": g["miou"].std(ddof=1).fillna(0),
                        "n": g.size(), "params": g["params_trainable"].mean(),
                        "train_min": g["train_time_min"].mean(), "ms": g["time_ms_per_image"].mean()})
    return out.reindex([m for m in MODELS if m in out.index])


# ---------------------------------------------------------------- figures
def fig_accuracy(df):
    t = main_table(df)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(t))
    for i, (m, r) in enumerate(t.iterrows()):
        ax.barh(i, r["mean"], height=0.5, color=COLOR[m], zorder=2)
        if r["n"] > 1:
            ax.errorbar(r["mean"], i, xerr=r["std"], color=INK, capsize=3, lw=1, zorder=3)
        txt = f"{r['mean']:.1f} ± {r['std']:.1f}" if r["n"] > 1 else f"{r['mean']:.1f}"
        ax.text(r["mean"] + max(r["std"], 0) + 1, i, txt, va="center", fontsize=9.5, color=INK)
    params = [f"{p / 1e6:.2f} M trainable params" if p >= 1e5
              else (f"{p / 1e3:.0f} K trainable params" if p > 0 else "no training")
              for p in t["params"]]
    names = {**LABEL, "exg_otsu": "ExG + Otsu"}
    ax.set_yticks(y, [f"{names[m]}\n{p}" for m, p in zip(t.index, params)])
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test mIoU (%)")
    clean_axes(ax)
    ax.set_title("Segmentation accuracy on the test set")
    subtitle(ax, "Mean ± std over 3 random seeds (ExG + Otsu is deterministic)")
    save(fig, "fig1_accuracy.png")


def fig_robustness(df):
    rob = df[(df["seed"] == 42) & (df["fraction"] == 1.0)]
    have = set(rob["scenario"])
    if len(have & set(SCENARIOS)) < 2:
        print("  (skip fig2: run scripts/robustness.py first)")
        return
    piv = rob.pivot_table(index=["model", "augment"], columns="scenario", values="miou")
    has_distort = any(a == "distort" for _, a in piv.index)
    ncols = 2 if has_distort else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2), sharey=True, squeeze=False)
    axes = axes[0]
    ys = np.arange(len(SCENARIOS))[::-1]

    ax = axes[0]
    for m in MODELS:
        if (m, "basic") not in piv.index:
            continue
        vals = [piv.loc[(m, "basic")].get(s, np.nan) for s in SCENARIOS]
        ax.scatter(vals, ys, s=60, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3, label=SHORT[m])
    ax.set_title("A. Standard training")
    subtitle(ax, "Test mIoU (%) on the same corrupted images for every model")

    if has_distort:
        ax = axes[1]
        for m, off in [("deeplabv3plus", 0.17), ("dinov2_decoder", -0.17)]:
            if (m, "distort") not in piv.index or (m, "basic") not in piv.index:
                continue
            before = [piv.loc[(m, "basic")].get(s, np.nan) for s in SCENARIOS]
            after = [piv.loc[(m, "distort")].get(s, np.nan) for s in SCENARIOS]
            yo = ys + off                      # small vertical offset so the two models don't overlap
            for b, a, yv in zip(before, after, yo):
                ax.plot([b, a], [yv, yv], color=COLOR[m], lw=1.2, alpha=0.5, zorder=2)
            ax.scatter(before, yo, s=60, facecolor=SURFACE, edgecolor=COLOR[m], linewidth=1.6, zorder=3)
            ax.scatter(after, yo, s=60, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.set_title("B. Trained with distortion augmentation")
        subtitle(ax, "Hollow = standard training, filled = + distortion augmentation")

    for ax in axes:
        clean_axes(ax)
        ax.set_xlim(45, 92)
        ax.set_xlabel("Test mIoU (%)")
        ax.axhspan(ys[0] - 0.5, ys[0] + 0.5, color="#f3f2ee", zorder=0)   # highlight the clean row
    axes[0].set_yticks(ys, [SCEN_LABEL[s] for s in SCENARIOS])
    handles = [Line2D([], [], marker="o", ls="", markersize=8, color=COLOR[m], label=SHORT[m])
               for m in MODELS if (m, "basic") in piv.index]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_robustness.png")


def fig_label_efficiency(df):
    lab = df[(df["scenario"] == "clean") & (df["augment"] == "basic") & (df["seed"] == 42)
             & (df["model"] != "exg_otsu")]
    lab = lab[(lab["fraction"] == 1.0) | lab["tag"].astype(str).str.endswith("_eq")]
    if lab["fraction"].nunique() < 2:
        print("  (skip fig3: run run_experiments.py --plan label first)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fracs = sorted(lab["fraction"].unique())
    for m in [m for m in MODELS if m in set(lab["model"])]:
        s = lab[lab["model"] == m].groupby("fraction")["miou"].mean().reindex(fracs)
        ax.plot(fracs, s.values, color=COLOR[m], lw=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round", zorder=3)
        ax.text(fracs[-1] * 1.08, s.values[-1], f"{SHORT[m]}  {s.values[-1]:.1f}", va="center",
                fontsize=9, color=INK)
    exg = df[(df["model"] == "exg_otsu") & (df["scenario"] == "clean")]["miou"]
    if len(exg):
        ax.axhline(exg.iloc[0], color=COLOR["exg_otsu"], lw=1.2, zorder=1)
        ax.text(fracs[0], exg.iloc[0] + 1, "ExG + Otsu (needs no labels)", fontsize=8.5, color=INK2)
    n_full = 142
    ax.set_xscale("log")
    ax.set_xticks(fracs, [f"{f * 100:.0f}%\n({round(f * n_full)} images)" for f in fracs])
    ax.minorticks_off()
    ax.set_xlim(fracs[0] * 0.85, fracs[-1] * 1.9)
    ax.set_ylim(50, 92)
    ax.set_ylabel("Test mIoU (%)")
    ax.set_xlabel("Labelled training images used")
    clean_axes(ax, grid_axis="y")
    ax.set_title("Label efficiency: how much labelled data is needed?")
    subtitle(ax, "Same number of weight updates for every run (seed 42)")
    save(fig, "fig3_label_efficiency.png")


def fig_accuracy_vs_cost(df):
    t = main_table(df)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, col, xlabel, logx in [(axes[0], "params", "Trainable parameters (log scale)", True),
                                  (axes[1], "train_min", "Training time on a MacBook Air (min)", False)]:
        for m, r in t.iterrows():
            if logx and r[col] <= 0:
                continue
            ax.errorbar(r[col], r["mean"], yerr=r["std"] if r["n"] > 1 else None, fmt="none",
                        ecolor=INK3, lw=1, capsize=2, zorder=2)
            ax.scatter(r[col], r["mean"], s=90, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
            dx, dy, ha = {"dinov2_decoder": (0, 10, "center"), "deeplabv3plus": (6, -15, "center")}.get(m, (8, -3, "left"))
            ax.annotate(SHORT[m], (r[col], r["mean"]), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=9, color=INK)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(1e3, 2e8)
            ax.minorticks_off()
        else:
            ax.set_xlim(-1, max(t[col]) * 1.35)
        ax.set_xlabel(xlabel)
        clean_axes(ax, grid_axis="both")
    axes[0].set_ylabel("Test mIoU (%)")
    axes[0].set_ylim(50, 92)
    axes[0].set_title("Accuracy vs cost")
    subtitle(axes[0], "Up and to the left is better (ExG has no trainable parameters)")
    fig.tight_layout()
    save(fig, "fig4_accuracy_vs_cost.png")


def per_image_table():
    rows = {}
    files = {"exg_otsu": "exg_otsu_test_clean.csv"}
    for m in MODELS[1:]:
        files[m] = f"{m}_{m}_ce_basic_frac100_seed42_test_clean.csv"
    for m, f in files.items():
        p = config.METRICS_DIR / "per_image" / f
        if p.exists():
            rows[m] = pd.read_csv(p).set_index("image")["miou"] * 100
    if not rows:
        return None
    t = pd.DataFrame(rows)
    if {"dinov2_decoder", "deeplabv3plus"} <= set(t.columns):
        t["decoder_minus_cnn"] = t["dinov2_decoder"] - t["deeplabv3plus"]
    TABLES.mkdir(parents=True, exist_ok=True)
    t.round(2).to_csv(TABLES / "per_image_comparison.csv")
    print("  saved results/tables/per_image_comparison.csv")
    return t


def fig_predictions(per_image):
    import torch  # noqa: F401  (only needed here)
    from src.data import list_samples, load_sample
    from src.models import get_model, load_checkpoint

    ckpts = {m: config.CHECKPOINT_DIR / f"{m}_ce_basic_frac100_seed42.pth" for m in MODELS[1:]}
    missing = [m for m, p in ckpts.items() if not p.exists()]
    if missing or per_image is None or "decoder_minus_cnn" not in per_image:
        print(f"  (skip fig5: missing checkpoints or per-image results {missing})")
        return
    diff = per_image["decoder_minus_cnn"]
    picks = [("Typical image", (per_image["deeplabv3plus"] - per_image["deeplabv3plus"].median()).abs().idxmin()),
             ("DINOv2 decoder much better", diff.idxmax()),
             ("DeepLabV3+ much better", diff.idxmin())]
    models = {"exg_otsu": get_model("exg_otsu")}
    models.update({m: load_checkpoint(p) for m, p in ckpts.items()})
    samples = {s.name: s for s in list_samples("test")}

    fp_c, fn_c = np.array([0.89, 0.29, 0.28]), np.array([0.93, 0.63, 0.0])   # red / amber
    cols = ["RGB image", "Ground truth"] + [SHORT[m] for m in MODELS]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.0 * len(cols), 2.25 * len(picks)))
    for r, (why, name) in enumerate(picks):
        image, gt = load_sample(samples[name])
        axes[r, 0].imshow(image)
        axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(why, fontsize=9, color=INK)
        for c, m in enumerate(MODELS, start=2):
            pred = models[m].predict(image)
            vis = np.repeat(pred[..., None].astype(float), 3, axis=2)
            vis[(pred == 1) & (gt == 0)] = fp_c
            vis[(pred == 0) & (gt == 1)] = fn_c
            axes[r, c].imshow(vis)
            axes[r, c].set_xlabel(f"mIoU {per_image.loc[name, m]:.1f}", fontsize=8.5, color=INK2)
        for c in range(len(cols)):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9.5, loc="center", fontweight="bold")
    fig.legend(handles=[Patch(color="white", ec=GRID, label="Plant (correct)"),
                        Patch(color="black", label="Soil (correct)"),
                        Patch(color=fp_c, label="False positive (soil predicted as plant)"),
                        Patch(color=fn_c, label="False negative (plant missed)")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig5_predictions.png")


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    found = False
    for m in MODELS[1:]:
        p = config.RESULTS_DIR / "training" / f"{m}_ce_basic_frac100_seed42.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p).dropna(subset=["val_miou"])
        found = True
        ax.plot(h["epoch"], h["val_miou"] * 100, color=COLOR[m], lw=2, solid_capstyle="round")
        if m == "dinov2_linear":        # its line ends mid-chart; label under the start to avoid the CNN curve
            ax.text(2.3, 68.5, SHORT[m], va="top", fontsize=9, color=INK)
        else:
            ax.text(h["epoch"].iloc[-1] + 0.4, h["val_miou"].iloc[-1] * 100, SHORT[m], va="center",
                    fontsize=9, color=INK)
    if not found:
        plt.close(fig)
        print("  (skip fig6: no training histories)")
        return
    ax.set_xlim(0.5, 30)
    ax.set_ylim(40, 92)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mIoU (%)")
    clean_axes(ax, grid_axis="y")
    ax.set_title("Training stability")
    subtitle(ax, "Validation mIoU per epoch (seed 42). Frozen DINOv2 features give a much smoother curve")
    save(fig, "fig6_training_curves.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-predictions", action="store_true", help="skip fig5 (needs torch + checkpoints)")
    args = p.parse_args()
    setup_style()
    df = load_summary()
    print("Making figures ...")
    fig_accuracy(df)
    fig_robustness(df)
    fig_label_efficiency(df)
    fig_accuracy_vs_cost(df)
    per_image = per_image_table()
    if not args.no_predictions:
        fig_predictions(per_image)
    fig_training_curves()
    print("Done.")


if __name__ == "__main__":
    main()
