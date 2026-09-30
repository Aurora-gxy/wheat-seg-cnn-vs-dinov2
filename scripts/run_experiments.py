"""Run many training runs in one go (resumable: runs that already have a checkpoint are skipped).

Plans:
    seeds    repeat the full-data runs with extra random seeds -> mean ± std   (experiment A)
    label    train on 10% / 25% / 50% of the training images                  (experiment C)
    distort  train with distortion augmentation, for the robustness test        (experiment B)

Run from the project root, e.g.:
    python scripts/run_experiments.py --plan seeds --dry-run     # show what would run + time
    caffeinate -i python scripts/run_experiments.py --plan seeds
    caffeinate -i python scripts/run_experiments.py --plan seeds label distort

You can stop at any time with Ctrl+C and run the same command again later:
finished runs are skipped, the interrupted run starts again from the beginning.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.experiment import run_name

MODELS = ["deeplabv3plus", "dinov2_linear", "dinov2_decoder"]
# Rough minutes per run on an Apple-silicon MacBook (measured on the full-data runs).
EST_MIN = {"deeplabv3plus": 16, "dinov2_linear": 3.5, "dinov2_decoder": 10}


def build_plan(plan: str, seeds, fractions):
    runs = []
    if plan == "seeds":
        runs = [dict(model=m, seed=s) for m in MODELS for s in seeds]
    elif plan == "label":
        runs = [dict(model=m, fraction=f, equal_steps=True) for m in MODELS for f in fractions]
    elif plan == "distort":
        runs = [dict(model=m, augment="distort") for m in ["deeplabv3plus", "dinov2_decoder"]]
    for r in runs:
        r.setdefault("seed", 42)
        r.setdefault("fraction", 1.0)
        r.setdefault("augment", "basic")
        r.setdefault("equal_steps", False)
    return runs


def command(r):
    cmd = [sys.executable, "scripts/train.py", "--model", r["model"], "--epochs", "25",
           "--patience", "6", "--seed", str(r["seed"]), "--fraction", str(r["fraction"]),
           "--augment", r["augment"]]
    if r["equal_steps"]:
        cmd.append("--equal-steps")
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", nargs="+", required=True, choices=["seeds", "label", "distort"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2], help="extra seeds (42 is already done)")
    p.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    p.add_argument("--dry-run", action="store_true", help="only print the plan")
    args = p.parse_args()

    runs = [r for plan in args.plan for r in build_plan(plan, args.seeds, args.fractions)]
    todo = []
    print(f"{'run':<55} status")
    for r in runs:
        name = run_name(r["model"], "ce", r["augment"], r["fraction"], r["seed"],
                        equal_steps=r["equal_steps"])
        done = (config.CHECKPOINT_DIR / f"{name}.pth").exists()
        print(f"{name:<55} {'done, skip' if done else 'to run'}")
        if not done:
            todo.append((name, r))
    total = sum(EST_MIN[r["model"]] for _, r in todo)
    print(f"\n{len(todo)} run(s) to do, roughly {total:.0f} min ({total / 60:.1f} h) in total.")
    if args.dry_run or not todo:
        return

    failed = []
    start = time.time()
    for i, (name, r) in enumerate(todo, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {name}\n{'=' * 70}", flush=True)
        result = subprocess.run(command(r), cwd=ROOT)
        if result.returncode != 0:
            print(f"!! {name} failed (exit code {result.returncode}); continuing with the next run")
            failed.append(name)
    print(f"\nFinished {len(todo) - len(failed)}/{len(todo)} runs in {(time.time() - start) / 60:.0f} min.")
    if failed:
        print("Failed runs:", *failed, sep="\n  ")
    print("Next: python scripts/summarize.py")


if __name__ == "__main__":
    main()
