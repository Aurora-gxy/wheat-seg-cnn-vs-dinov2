"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "EWS-Dataset"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / "cache"

# Folder names inside the EWS dataset
SPLITS = {"train": "train", "val": "validation", "test": "test"}

IMAGE_SIZE = 350          # EWS images are 350 x 350 RGB
MASK_SUFFIX = "_mask"     # image: X.png  -> mask: X_mask.png

# Class ids used everywhere in this project
SOIL = 0
PLANT = 1
CLASS_NAMES = {SOIL: "soil", PLANT: "plant"}

# ImageNet statistics (used by both ResNet-50 and DINOv2 backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_SEED = 42
