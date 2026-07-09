"""
Hard-negative mining: find the examples the BASE model handles badly, so training
can concentrate on the cases LLMs are actually poor at (the differentiation thesis).

An example is a "hard negative" if the base model:
  1. predicts the wrong label (top_label != gold), OR
  2. is overconfident on an abstain case: gold == "abstain" but the model commits to a
     substantive label with confidence >= --overconfident-threshold, OR
  3. is a low-margin near-miss: correct label, but top confidence < --lowconf-threshold
     (kept only when --keep-lowconf is set).

Outputs a hard_negatives.jsonl (original schema + base_pred / base_conf / hard_reason
diagnostics). Optionally emits an augmented training file that repeats the mined hard
negatives --weight times on top of an existing training set, so you can upweight them
during SFT (run prepare_sft.py on the augmented file afterwards).

Examples (GPU, in Colab):
  python mine_hard_negatives.py --model Qwen/Qwen3-1.7B --pool ../data/train.jsonl \
      --out ../data/hard_negatives.jsonl
  python mine_hard_negatives.py --model Qwen/Qwen3-1.7B --pool ../data/train.jsonl \
      --out ../data/hard_negatives.jsonl \
      --augment-train ../data/train.jsonl --weight 3 --out-augmented ../data/train_hardaug.jsonl

Local smoke test (no GPU):
  python mine_hard_negatives.py --selftest --pool ../data/train.jsonl --out /tmp/hn.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SYSTEM_PROMPT, build_user_prompt, load_jsonl, write_jsonl
from model_utils import DummyModel, HFClassifier


def classify_hard(gold, pred, conf, overconf_threshold, lowconf_threshold, keep_lowconf):
    """Return a hard-negative reason string, or None if the example is not hard."""
    if pred != gold:
        if gold == "abstain" and pred != "abstain" and conf >= overconf_threshold:
            return "abstain_overconfident"
        return "wrong_label"
    if keep_lowconf and conf < lowconf_threshold:
        return "low_margin_correct"
    return None


def mine(model, pool, temperature, overconf_threshold, lowconf_threshold, keep_lowconf):
    hard = []
    total = len(pool)
    print(f"Scoring {total} pool examples with base model...", flush=True)
    for idx, example in enumerate(pool, start=1):
        user = build_user_prompt(example)
        _, top_label, conf = model.score_labels(SYSTEM_PROMPT, user, temperature=temperature)
        gold = example["label"]
        reason = classify_hard(
            gold, top_label, conf, overconf_threshold, lowconf_threshold, keep_lowconf
        )
        if reason is not None:
            row = dict(example)
            row["base_pred"] = top_label
            row["base_conf"] = round(conf, 4)
            row["hard_reason"] = reason
            hard.append(row)
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"  {idx}/{total} scored | {len(hard)} hard so far", flush=True)
    return hard


def summarize(hard, pool_size):
    print("\n=== HARD-NEGATIVE SUMMARY ===")
    print(f"Pool: {pool_size} | hard negatives: {len(hard)} ({len(hard)/max(pool_size,1):.1%})")
    print("By reason:", dict(Counter(r["hard_reason"] for r in hard)))
    print("By gold label:", dict(Counter(r["label"] for r in hard)))
    forced = [r for r in hard if r["label"] == "abstain" and r["base_pred"] != "abstain"]
    print(f"Abstain cases the base model forced to a label: {len(forced)}")


def build_augmented(train_path, hard, weight, out_path):
    """train + hard-negatives repeated `weight` times (strip diagnostic fields)."""
    base = load_jsonl(train_path)
    diagnostic_keys = {"base_pred", "base_conf", "hard_reason"}
    clean_hard = [{k: v for k, v in row.items() if k not in diagnostic_keys} for row in hard]

    augmented = list(base)
    for _ in range(weight):
        augmented.extend(clean_hard)
    # Re-id so downstream tooling sees unique ids.
    for i, row in enumerate(augmented):
        row["id"] = f"aug{i:05d}"
    write_jsonl(out_path, augmented)
    print(
        f"\nWrote augmented training set: {len(base)} base + "
        f"{weight}x{len(clean_hard)} hard = {len(augmented)} rows -> {out_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model id (base model to probe)")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter to probe instead")
    parser.add_argument("--pool", required=True, help="JSONL pool to score (e.g. train.jsonl)")
    parser.add_argument("--out", required=True, help="where to write hard_negatives.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0, help="scoring temperature")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--overconfident-threshold", type=float, default=0.8)
    parser.add_argument("--lowconf-threshold", type=float, default=0.4)
    parser.add_argument("--keep-lowconf", action="store_true", help="also keep low-margin correct")
    parser.add_argument("--augment-train", default=None, help="base train JSONL to augment")
    parser.add_argument("--weight", type=int, default=3, help="times to repeat hard negatives")
    parser.add_argument("--out-augmented", default=None, help="output path for augmented train")
    parser.add_argument("--selftest", action="store_true", help="use DummyModel (no GPU)")
    args = parser.parse_args()

    pool = load_jsonl(args.pool)
    if args.max_examples is not None:
        pool = pool[: args.max_examples]
        print(f"Using first {len(pool)} pool examples")

    if args.selftest:
        print("SELF-TEST with dummy model.")
        model = DummyModel(pool)
    else:
        if not args.model and not args.adapter:
            raise SystemExit("Provide --model (and optional --adapter) or use --selftest")
        model = HFClassifier(args.model, temperature=args.temperature, adapter_path=args.adapter)

    hard = mine(
        model,
        pool,
        temperature=args.temperature,
        overconf_threshold=args.overconfident_threshold,
        lowconf_threshold=args.lowconf_threshold,
        keep_lowconf=args.keep_lowconf,
    )
    write_jsonl(args.out, hard)
    print(f"Wrote {len(hard)} hard negatives to {args.out}")
    summarize(hard, len(pool))

    if args.augment_train:
        out_aug = args.out_augmented or str(Path(args.out).with_name("train_hardaug.jsonl"))
        build_augmented(args.augment_train, hard, args.weight, out_aug)


if __name__ == "__main__":
    main()
