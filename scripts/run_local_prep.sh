#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Generate disjoint train/val/test_holdout splits (CPU, local)"
python scripts/make_splits.py --train-n 6000 --val-n 800 --test-n 1000 --seed 0 --out-dir data

echo "==> Prepare SFT chat JSONL"
python scripts/prepare_sft.py --data data/train.jsonl --out data/train_sft.jsonl
python scripts/prepare_sft.py --data data/val.jsonl --out data/val_sft.jsonl

echo
echo "Local prep done."
echo "Next: open notebooks/train_and_eval.ipynb in Colab for GPU training/eval."
