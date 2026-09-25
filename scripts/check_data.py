"""Sanity-check the EWS dataset before training anything.

Run from the project root:
    python scripts/check_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

from src import config
from src.data import list_samples, load_sample


def main():
    rows, problems = [], []
    for split in config.SPLITS:
        for s in list_samples(split):
            image, mask = load_sample(s)
            raw = np.array(Image.open(s.mask_path).convert("L"))
            edge_frac = float(((raw > 0) & (raw < 255)).mean())   # anti-aliased pixels
            if image.shape != (config.IMAGE_SIZE, config.IMAGE_SIZE, 3):
                problems.append(f"{s.name}: image shape {image.shape}")
            if mask.shape != image.shape[:2]:
                problems.append(f"{s.name}: mask shape {mask.shape} != image {image.shape[:2]}")
            rows.append({"split": split, "image": s.name,
                         "plant_ratio": float(mask.mean()), "edge_pixel_ratio": edge_frac})

    df = pd.DataFrame(rows)
    print("\nImages per split:")
    print(df.groupby("split").size().reindex(list(config.SPLITS)).to_string())

    print("\nPlant pixel ratio per split (share of pixels labelled plant):")
    stats = df.groupby("split")["plant_ratio"].agg(["mean", "min", "max"]).reindex(list(config.SPLITS))
    print((stats * 100).round(1).astype(str).add(" %").to_string())

    print(f"\nAnti-aliased mask pixels (neither 0 nor 255): {df['edge_pixel_ratio'].mean() * 100:.2f} % on average")

    overlap = set(df[df.split == "train"].image) & set(df[df.split == "test"].image)
    print(f"Train/test filename overlap: {len(overlap)}")

    out = config.RESULTS_DIR / "data_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nPer-image stats saved to {out.relative_to(config.ROOT)}")

    if problems:
        print("\nPROBLEMS FOUND:")
        print("\n".join("  " + p for p in problems))
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
