"""
Baseline / tuned eval harness for the algebra error-type classifier.

Measures:
  1. Accuracy
  2. Schema validity (clean single-label output)
  3. Consistency (repeat runs)
  4. Calibration (ECE + optional reliability diagram) when --score-labels is set

Examples:
  python run_baseline.py --selftest --data ../data/testset.jsonl
  python run_baseline.py --model Qwen/Qwen3-1.7B --data ../data/testset.jsonl --runs 5
  python run_baseline.py --model Qwen/Qwen3-1.7B --adapter ../outputs/lora --data ../data/testset.jsonl --calibration ../outputs/calibration.json --score-labels
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LABELS, SYSTEM_PROMPT, build_user_prompt, load_jsonl, parse_label
from metrics import expected_calibration_error, save_reliability_diagram, summarize_confusion
from model_utils import DummyModel, HFClassifier, apply_abstention, load_calibration


def evaluate(model, data, runs, score_labels=False, calibration=None):
    calibration = calibration or {"temperature": 1.0, "abstain_threshold": None}
    temperature = calibration.get("temperature", 1.0)
    per_example = []

    for example in data:
        user = build_user_prompt(example)
        preds = []
        clean_flags = []
        confidences = []

        if score_labels:
            _, top_label, confidence = model.score_labels(
                SYSTEM_PROMPT, user, temperature=temperature
            )
            pred = apply_abstention(top_label, confidence, calibration)
            preds = [pred]
            clean_flags = [pred == top_label and pred in LABELS]
            confidences = [confidence]
        else:
            for _ in range(runs):
                raw = model.generate(SYSTEM_PROMPT, user)
                label, clean = parse_label(raw)
                preds.append(label)
                clean_flags.append(clean)

        counts = Counter(preds)
        modal, modal_count = counts.most_common(1)[0]
        consistency = modal_count / len(preds)
        pred_for_acc = preds[0] if score_labels else modal

        per_example.append(
            {
                "id": example["id"],
                "gold": example["label"],
                "pred": pred_for_acc,
                "modal_pred": modal,
                "preds": preds,
                "consistency": consistency,
                "schema_valid_rate": sum(clean_flags) / len(clean_flags),
                "confidence": confidences[0] if confidences else None,
                "correct": pred_for_acc == example["label"],
            }
        )
    return per_example


def summarize(results, score_labels=False, calibration=None, out_dir=None):
    calibration = calibration or {}
    n = len(results)
    acc = sum(row["correct"] for row in results) / n
    mean_consistency = sum(row["consistency"] for row in results) / n
    mean_schema = sum(row["schema_valid_rate"] for row in results) / n

    print("\n=== EVAL RESULTS ===")
    print(f"Examples: {n}")
    print(f"Accuracy: {acc:.1%}")
    print(f"Mean schema validity: {mean_schema:.1%}")
    print(f"Mean consistency: {mean_consistency:.1%}")

    if score_labels:
        confidences = [row["confidence"] for row in results if row["confidence"] is not None]
        correct = [row["correct"] for row in results if row["confidence"] is not None]
        ece, bin_stats = expected_calibration_error(confidences, correct)
        print(f"Expected Calibration Error (ECE): {ece:.3f}")
        print(f"Calibration temperature: {calibration.get('temperature', 1.0)}")
        print(f"Abstain threshold: {calibration.get('abstain_threshold')}")
        if out_dir and bin_stats:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_reliability_diagram(
                bin_stats,
                out_dir / "reliability.png",
                title="Reliability diagram",
            )
            print(f"Saved reliability diagram to {out_dir / 'reliability.png'}")

    print("\nPer-example:")
    header = f"{'id':<8}{'gold':<28}{'pred':<28}{'ok':<4}{'consist':<9}{'schema'}"
    if score_labels:
        header += f"{'conf':<8}"
    print(header)
    for row in results:
        line = (
            f"{row['id']:<8}{row['gold']:<28}{str(row['pred']):<28}"
            f"{'Y' if row['correct'] else 'N':<4}{row['consistency']:<9.0%}"
            f"{row['schema_valid_rate']:.0%}"
        )
        if score_labels:
            line += f"{row['confidence']:.0%}" if row["confidence"] is not None else "n/a"
        print(line)

    print("\nConfusions (gold -> predicted), errors only:")
    confusions = summarize_confusion(
        [{"gold": row["gold"], "pred": row["pred"], "correct": row["correct"]} for row in results]
    )
    if not confusions:
        print("  (none)")
    for (gold, pred), count in sorted(confusions.items(), key=lambda item: -item[1]):
        print(f"  {gold:<28} -> {pred:<28} x{count}")

    abstain_rows = [row for row in results if row["gold"] == "abstain"]
    if abstain_rows:
        forced = sum(row["pred"] != "abstain" for row in abstain_rows)
        print(f"\nAbstain cases: {len(abstain_rows)} total, {forced} forced to a label")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model id")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory")
    parser.add_argument("--data", default="../data/test_holdout.jsonl")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Evaluate only the first N examples (faster on large holdout sets)",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7, help="generation temperature")
    parser.add_argument("--calibration", default=None, help="calibration.json from calibrate.py")
    parser.add_argument("--score-labels", action="store_true", help="score label log-probs (for ECE)")
    parser.add_argument("--out-dir", default="../outputs/eval", help="where to save plots/results")
    parser.add_argument("--save-predictions", default=None, help="write per-example JSONL")
    parser.add_argument(
        "--normalize-real",
        action="store_true",
        help="treat --data as a real-data dump (Eedi/DataShop) and map it to the taxonomy",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.normalize_real:
        from real_data import load_real_eval

        data = load_real_eval(args.data)
        needs = sum(1 for row in data if row.get("needs_review"))
        print(f"Loaded {len(data)} real examples (normalized to taxonomy)")
        if needs:
            print(f"WARNING: {needs} rows have unmapped misconceptions (needs_review=true)")
    else:
        data = load_jsonl(args.data)
    if args.max_examples is not None:
        data = data[: args.max_examples]
        print(f"Using first {len(data)} examples from {args.data}")
    calibration = load_calibration(args.calibration)

    if args.selftest:
        print("Running SELF-TEST with dummy model.")
        model = DummyModel(data)
    else:
        if not args.model:
            raise SystemExit("Provide --model or use --selftest")
        model = HFClassifier(args.model, temperature=args.temperature, adapter_path=args.adapter)

    runs = 1 if args.score_labels else args.runs
    results = evaluate(
        model,
        data,
        runs=runs,
        score_labels=args.score_labels,
        calibration=calibration,
    )
    summarize(
        results,
        score_labels=args.score_labels,
        calibration=calibration,
        out_dir=args.out_dir if args.score_labels else None,
    )

    if args.save_predictions:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")
        print(f"Saved predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
