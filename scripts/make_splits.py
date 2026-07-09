"""
Build disjoint train/val/test splits from a single deduplicated pool.

Generating each split with a different seed does NOT guarantee disjoint problems:
the injectors draw from a finite parameter space, so at high volume the same
`problem` string appears across splits (measured ~69% train/test overlap at 6k/1k).
That contaminates the held-out eval.

This script generates ONE balanced, deduplicated pool and partitions it per-label
into train/val/test, so the three files share zero problems while keeping the label
distribution identical across splits.

Example:
  python make_splits.py --train-n 6000 --val-n 800 --test-n 1000 --seed 0 --out-dir ../data
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from generate_dataset import build_dataset, write_jsonl


def partition(pool, train_frac, val_frac, seed=0):
    """Split a pool into (train, val, test) per-label, preserving balance.

    Splits are shuffled across labels so that any `--max-examples` subsample
    (e.g. in run_baseline on the holdout) stays label-representative.
    """
    by_label = defaultdict(list)
    for row in pool:
        by_label[row["label"]].append(row)

    train, val, test = [], [], []
    for label, rows in by_label.items():
        n = len(rows)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train.extend(rows[:n_train])
        val.extend(rows[n_train : n_train + n_val])
        test.extend(rows[n_train + n_val :])

    rng = random.Random(seed)
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def reassign_ids(rows, prefix):
    for i, row in enumerate(rows):
        row["id"] = f"{prefix}{i:05d}"
    return rows


def summarize(name, rows):
    counts = dict(Counter(r["label"] for r in rows))
    print(f"{name}: {len(rows)} rows | {counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=6000)
    ap.add_argument("--val-n", type=int, default=800)
    ap.add_argument("--test-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--abstain-frac", type=float, default=0.35)
    ap.add_argument("--out-dir", default="../data")
    args = ap.parse_args()

    total = args.train_n + args.val_n + args.test_n
    # Generate a slightly larger pool to absorb per-label rounding.
    pool_n = int(total * 1.03)
    print(f"Building pool of ~{pool_n} deduplicated examples (seed {args.seed})...")
    pool = build_dataset(pool_n, args.seed, args.abstain_frac)
    print(f"Pool: {len(pool)} unique examples")

    train_frac = args.train_n / total
    val_frac = args.val_n / total
    train, val, test = partition(pool, train_frac, val_frac, seed=args.seed)

    reassign_ids(train, "tr")
    reassign_ids(val, "va")
    reassign_ids(test, "te")

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train, keep_internal=False)
    write_jsonl(out_dir / "val.jsonl", val, keep_internal=False)
    write_jsonl(out_dir / "test_holdout.jsonl", test, keep_internal=False)

    print()
    summarize("train", train)
    summarize("val", val)
    summarize("test_holdout", test)

    tp = {r["problem"] for r in train}
    vp = {r["problem"] for r in val}
    ep = {r["problem"] for r in test}
    print()
    print(f"train-test overlap = {len(tp & ep)}")
    print(f"val-test overlap   = {len(vp & ep)}")
    print(f"train-val overlap  = {len(tp & vp)}")
    if (tp & ep) or (vp & ep) or (tp & vp):
        raise SystemExit("ERROR: splits are not disjoint")
    print("\nSplits are disjoint. Wrote train.jsonl, val.jsonl, test_holdout.jsonl")


if __name__ == "__main__":
    main()
