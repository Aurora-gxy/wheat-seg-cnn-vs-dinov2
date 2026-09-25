"""Evaluate one model on one split and save the metrics.

Run from the project root, e.g.:
    python scripts/run_eval.py --model exg_otsu --split val
    python scripts/run_eval.py --model exg_otsu --split test --save-preds
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import list_samples
from src.evaluate import evaluate, print_summary, save_results
from src.models import AVAILABLE_MODELS, get_model
from src.utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
    p.add_argument("--split", default="test", choices=list(config.SPLITS))
    p.add_argument("--tag", default="", help="optional label, e.g. 'frac25' or 'seed1'")
    p.add_argument("--save-preds", action="store_true", help="save predicted masks as PNG")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = p.parse_args()

    set_seed(args.seed)
    model = get_model(args.model)
    model.fit(list_samples("train"), list_samples("val"))

    samples = list_samples(args.split)
    pred_dir = None
    if args.save_preds:
        pred_dir = config.RESULTS_DIR / "predictions" / f"{args.model}_{args.split}"
    summary, per_image = evaluate(model, samples, save_pred_dir=pred_dir)

    print_summary(summary, f"{args.model} on {args.split}")
    path = save_results(summary, per_image, args.model, args.split, tag=args.tag)
    print(f"\nSaved to {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
