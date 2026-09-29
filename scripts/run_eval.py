"""Evaluate one model on one split and save the metrics.

Run from the project root, e.g.:
    python scripts/run_eval.py --model exg_otsu --split test
    python scripts/run_eval.py --checkpoint checkpoints/<run>.pth --split test --save-preds
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import list_samples
from src.evaluate import evaluate, print_summary, save_results
from src.models import AVAILABLE_MODELS, get_model, load_checkpoint
from src.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=AVAILABLE_MODELS, help="a model that needs no training")
    g.add_argument("--checkpoint", type=Path, help="a trained model saved by scripts/train.py")
    p.add_argument("--split", default="test", choices=list(config.SPLITS))
    p.add_argument("--tag", default=None, help="label for this run (default: checkpoint name)")
    p.add_argument("--save-preds", action="store_true", help="save predicted masks as PNG")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = p.parse_args()

    set_seed(args.seed)
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
        tag = args.tag if args.tag is not None else args.checkpoint.stem
    else:
        model = get_model(args.model)
        model.fit(list_samples("train"), list_samples("val"))   # no-op for training-free models
        tag = args.tag or ""

    samples = list_samples(args.split)
    pred_dir = None
    if args.save_preds:
        pred_dir = config.RESULTS_DIR / "predictions" / f"{tag or model.name}_{args.split}"
    summary, per_image = evaluate(model, samples, save_pred_dir=pred_dir)

    print_summary(summary, f"{tag or model.name} on {args.split}")
    path = save_results(summary, per_image, model.name, args.split, tag=tag)
    print(f"\nSaved to {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
