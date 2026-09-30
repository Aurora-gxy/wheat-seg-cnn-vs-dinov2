"""Train a PyTorch model, keep the best checkpoint on the validation set, then evaluate on test.

Run from the project root, e.g.:
    # quick check that everything works (~2-3 min)
    python scripts/train.py --model deeplabv3plus --quick

    # CNN baseline
    python scripts/train.py --model deeplabv3plus --epochs 25 --patience 6 --batch-size 4

    # DINOv2 (frozen backbone, only the head is trained)
    python scripts/train.py --model dinov2_linear --epochs 25 --patience 6
    python scripts/train.py --model dinov2_decoder --epochs 25 --patience 6

Outputs:
    checkpoints/<run>.pth                 trained weights (not committed to git)
    results/training/<run>.csv            loss / val mIoU per epoch
    results/metrics/summary.csv           one row with the test-set scores
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data import list_samples
from src.evaluate import evaluate, print_summary, save_results
from src.models import DEFAULT_LR, TRAINABLE_MODELS, get_model
from src.experiment import DEFAULT_BATCH, equal_steps_schedule, run_name
from src.models.torch_base import AUGMENTATIONS, TrainConfig
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=TRAINABLE_MODELS)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=None, help="default: 4 for the CNN, 8 for DINOv2")
    p.add_argument("--lr", type=float, default=None, help="default depends on the model")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", default="ce", choices=["ce", "dice", "ce_dice"])
    p.add_argument("--augment", default="basic", choices=list(AUGMENTATIONS))
    p.add_argument("--fraction", type=float, default=1.0, help="share of training images to use")
    p.add_argument("--patience", type=int, default=None, help="early stopping patience (validations)")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--no-pretrained", action="store_true", help="do not load pre-trained weights")
    p.add_argument("--backbone", default="vits14", choices=["vits14", "vitb14"], help="DINOv2 size")
    p.add_argument("--n-layers", type=int, default=4, help="DINOv2: use features of the last N blocks")
    p.add_argument("--run-name", default=None)
    p.add_argument("--equal-steps", action="store_true",
                   help="with --fraction < 1: scale epochs so every run gets the same number of updates")
    p.add_argument("--quick", action="store_true", help="2 epochs on 10%% of the data, for testing")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs, args.fraction = 2, 0.1
    if args.lr is None:
        args.lr = DEFAULT_LR[args.model]
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH[args.model]
    is_dino = args.model.startswith("dinov2")
    run = args.run_name or run_name(args.model, args.loss, args.augment, args.fraction, args.seed,
                                    args.backbone, args.equal_steps, args.quick)

    epochs, val_every, warmup = args.epochs, 1, min(2, max(0, args.epochs - 1))
    if args.equal_steps and args.fraction < 1:
        epochs, val_every, warmup = equal_steps_schedule(args.epochs, args.fraction)
        print(f"Equal-steps schedule: {epochs} epochs, validating every {val_every} epochs")
    set_seed(args.seed)

    cfg = TrainConfig(epochs=epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, loss=args.loss, augment=args.augment,
                      patience=args.patience, seed=args.seed, val_every=val_every,
                      warmup_epochs=warmup)
    model_kwargs = {"pretrained": False} if args.no_pretrained else {}
    if is_dino:
        model_kwargs.update(backbone=args.backbone, n_layers=args.n_layers)
    model = get_model(args.model, train_cfg=cfg, **model_kwargs)

    train_samples = list_samples("train", fraction=args.fraction, seed=args.seed)
    val_samples = list_samples("val")
    print(f"Run: {run}")
    model.fit(train_samples, val_samples)

    ckpt_path = config.CHECKPOINT_DIR / f"{run}.pth"
    model.save(ckpt_path)
    hist_path = config.RESULTS_DIR / "training" / f"{run}.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model.history).to_csv(hist_path, index=False)

    summary, per_image = evaluate(model, list_samples("test"))
    summary.update({
        "best_val_miou": model.best_val_miou,
        "train_time_min": model.train_time_s / 60,
        "epochs_run": len(model.history),
        "n_train_images": len(train_samples),
        "fraction": args.fraction,
        "loss": args.loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "equal_steps": bool(args.equal_steps and args.fraction < 1),
        "augment": args.augment,
        "seed": args.seed,
    })
    print_summary(summary, f"{run} on test")
    path = save_results(summary, per_image, args.model, "test", tag=run)
    print(f"\nCheckpoint: {ckpt_path.relative_to(config.ROOT)}")
    print(f"History:    {hist_path.relative_to(config.ROOT)}")
    print(f"Metrics:    {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
