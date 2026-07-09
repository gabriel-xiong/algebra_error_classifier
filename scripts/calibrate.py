"""
Fit temperature scaling on a held-out validation set.

Uses label log-prob scoring (Option A from the spec): the model scores each of the
seven labels; temperature T scales logits before softmax; abstain threshold is
chosen on the same val set.

Example:
  python calibrate.py --model Qwen/Qwen3-1.7B --adapter ../outputs/lora --data ../data/val.jsonl --out ../outputs/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LABELS, SYSTEM_PROMPT, build_user_prompt, load_jsonl
from model_utils import HFClassifier


def probs_from_log_scores(log_scores: dict[str, float], temperature: float) -> dict[str, float]:
    """Approximate temperature scaling on cached label log-scores."""
    temp = max(temperature, 1e-5)
    scaled = {label: score / temp for label, score in log_scores.items()}
    max_log = max(scaled.values())
    probs = {label: math.exp(score - max_log) for label, score in scaled.items()}
    norm = sum(probs.values())
    return {label: value / norm for label, value in probs.items()}


def score_dataset(model, data):
    cache = []
    total = len(data)
    print(f"Scoring {total} validation examples (7 labels each)...", flush=True)
    for idx, example in enumerate(data, start=1):
        user = build_user_prompt(example)
        log_scores, _, _ = model.score_labels(SYSTEM_PROMPT, user, temperature=1.0)
        cache.append({"label": example["label"], "log_scores": log_scores})
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"  {idx}/{total} scored", flush=True)
    return cache


def nll_from_cache(cache, temperature):
    total = 0.0
    for item in cache:
        probs = probs_from_log_scores(item["log_scores"], temperature)
        gold = item["label"]
        total -= math.log(probs[gold] + 1e-12)
    return total / len(cache)


def fit_temperature(cache, lo=0.25, hi=8.0, steps=25):
    best_t = 1.0
    best_nll = float("inf")
    print(f"Searching temperature over {steps} values...", flush=True)
    for step in range(steps):
        t = lo + (hi - lo) * step / max(steps - 1, 1)
        nll = nll_from_cache(cache, t)
        if nll < best_nll:
            best_nll = nll
            best_t = t
    return best_t, best_nll


def choose_abstain_threshold(cache, temperature, thresholds=None):
    thresholds = thresholds or [round(x * 0.05, 2) for x in range(1, 20)]
    best = {"threshold": None, "accuracy": 0.0, "abstain_rate": 0.0}
    print(f"Choosing abstain threshold over {len(thresholds)} values...", flush=True)

    for threshold in thresholds:
        correct = 0
        abstained = 0
        for item in cache:
            probs = probs_from_log_scores(item["log_scores"], temperature)
            top_label = max(probs, key=probs.get)
            confidence = probs[top_label]
            pred = top_label if confidence >= threshold else "abstain"
            if pred == "abstain":
                abstained += 1
            if pred == item["label"]:
                correct += 1
        accuracy = correct / len(cache)
        abstain_rate = abstained / len(cache)
        if accuracy >= best["accuracy"]:
            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "abstain_rate": abstain_rate,
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", default="../data/val.jsonl")
    parser.add_argument("--out", default="../outputs/calibration.json")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Use only the first N val examples (faster on Colab)",
    )
    parser.add_argument("--temperature-steps", type=int, default=25)
    args = parser.parse_args()

    data = load_jsonl(args.data)
    if args.max_examples is not None:
        data = data[: args.max_examples]
        print(f"Using first {len(data)} validation examples", flush=True)

    print("Loading model...", flush=True)
    model = HFClassifier(args.model, temperature=0.0, adapter_path=args.adapter)
    print("Model ready. Starting calibration...", flush=True)

    cache = score_dataset(model, data)
    temperature, nll = fit_temperature(cache, steps=args.temperature_steps)
    threshold_info = choose_abstain_threshold(cache, temperature)

    payload = {
        "temperature": temperature,
        "val_nll": nll,
        "abstain_threshold": threshold_info["threshold"],
        "val_accuracy_at_threshold": threshold_info["accuracy"],
        "val_abstain_rate_at_threshold": threshold_info["abstain_rate"],
        "labels": LABELS,
        "n_val_examples": len(data),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved calibration to {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
