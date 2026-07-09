"""
Litmus tagging harness for the SLM pivot decision (prompted / zero-shot ONLY).

This is the GATING experiment. It answers, BEFORE any item-bank build or
fine-tuning: can a *prompted* model tag a student's chosen wrong answer
(distractor) on an AP Bio MCQ to a MID-GRAINED misconception, and do it
CONSISTENTLY across repeated runs on NOVEL item stems?

    2x2 decision (see docs/litmus_plan.md):
      base 1.7B consistent?  frontier consistent?  ->  verdict
      -------------------------------------------------------------------
      yes                    (any)                 ->  no fine-tuning project
      no                     yes                   ->  distill frontier -> SLM
      no                     no                    ->  fine-tuning justified
                                                        (then check labels are
                                                         not ambiguous)

The bar is CONSISTENCY, not one-shot accuracy: a prompt that cannot tag
*reliably* is the capability gap the pivot would fill.

This harness is ADDITIVE. It imports model_utils / common(_bio) read-only and
reuses their patterns (HFClassifier, DummyModel spirit, parse_label contract,
run_baseline's modal-consistency metric). It does NOT fine-tune anything.

Usage:
  # Local smoke test (Windows CPU: only the dummy path runs here):
  python scripts/litmus_tagging.py --selftest --data data/litmus_apbio_seed.jsonl --runs 3

  # Base Qwen3-1.7B on GPU (Colab):
  python scripts/litmus_tagging.py --model Qwen/Qwen3-1.7B --data data/litmus_apbio_seed.jsonl --runs 5

  # Optional frontier model (needs OPENAI_API_KEY or ANTHROPIC_API_KEY + lib):
  python scripts/litmus_tagging.py --frontier auto --data data/litmus_apbio_seed.jsonl --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reused read-only from the existing project (do not reinvent these).
# SUBSTANTIVE_LABELS = the 3 coarse cognitive-kind labels the mid-grained
# misconceptions must map UP to (common_bio's v1 taxonomy minus `abstain`).
from common_bio import SUBSTANTIVE_LABELS as COARSE_LABELS  # noqa: E402
from metrics import summarize_confusion  # noqa: E402
from model_utils import HFClassifier  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data" / "litmus_apbio_seed.jsonl"
DEFAULT_MISCONCEPTIONS = REPO_ROOT / "data" / "apbio_misconceptions.json"


# --------------------------------------------------------------------- loading

def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_misconceptions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    misconceptions = payload["misconceptions"]
    # Reconcile with common_bio: every mid-grained misconception must roll up to
    # one of the coarse cognitive-kind labels the product actually consumes.
    bad = sorted({m["coarse"] for m in misconceptions} - set(COARSE_LABELS))
    if bad:
        raise SystemExit(
            f"Taxonomy has coarse labels not in common_bio.SUBSTANTIVE_LABELS: {bad}. "
            f"Allowed: {COARSE_LABELS}"
        )
    return misconceptions


def build_examples(items: list[dict], id_to_coarse: dict[str, str]) -> list[dict]:
    """Expand each item into one tagging example per tagged (wrong) distractor.

    Each example = a student who CHOSE that distractor; gold is the authored
    mid-grained misconception id for that choice.
    """
    examples: list[dict] = []
    unknown: set[str] = set()
    for item in items:
        for chosen, tag in item.get("distractor_tags", {}).items():
            gold_id = tag.get("misconception_id")
            if gold_id is None:
                continue
            if gold_id not in id_to_coarse:
                unknown.add(gold_id)
            examples.append(
                {
                    "item_id": item["id"],
                    "chosen": chosen,
                    "item": item,
                    "gold_misconception": gold_id,
                    "gold_coarse": id_to_coarse.get(gold_id, tag.get("error_type")),
                }
            )
    if unknown:
        print(
            "WARNING: seed uses misconception ids not in the taxonomy: "
            + ", ".join(sorted(unknown))
        )
    return examples


# --------------------------------------------------------------------- prompting

SYSTEM_PROMPT = (
    "You are an expert AP Biology tutor. A student answered a multiple-choice "
    "question incorrectly. Given the question and the specific wrong answer the "
    "student chose, identify WHICH misconception from a fixed list best explains "
    "that choice. You must pick exactly one id from the provided list. You do not "
    "explain, you do not add prose, you output only the misconception id."
)


def _format_choices(choices: dict) -> str:
    return "\n".join(f"  {key}. {choices[key]}" for key in sorted(choices))


def _format_candidate_list(misconceptions: list[dict]) -> str:
    return "\n".join(f"  {m['id']}: {m['name']}" for m in misconceptions)


def build_user_prompt(example: dict, misconceptions: list[dict]) -> str:
    """Constrained-classification prompt. The gold tag is NEVER included."""
    item = example["item"]
    chosen = example["chosen"]
    choices = item.get("choices", {})
    passage = item.get("passage")
    passage_block = f"Passage:\n{passage}\n\n" if passage else ""
    chosen_text = choices.get(chosen, "")

    return f"""Here is the fixed list of candidate misconceptions (id: short name):

{_format_candidate_list(misconceptions)}

Task: the student chose an incorrect answer. Decide which single misconception
from the list above best explains why that wrong answer was attractive to them.
Rules:
- Output only the misconception id, exactly as written in the list.
- Do not output the name, an explanation, punctuation, or any extra text.

{passage_block}Question: {item.get('stem', '')}
Choices:
{_format_choices(choices)}
Correct answer: {item.get('correct')}
Student's chosen (incorrect) answer: {chosen}. {chosen_text}

Misconception id:"""


# --------------------------------------------------------------------- parsing

def parse_misconception(
    raw: str,
    ids: list[str],
    name_to_id: dict[str, str],
) -> tuple[str | None, bool]:
    """Return (misconception_id, is_schema_valid).

    Mirrors common.parse_label's contract: exact match is clean; a recovered
    substring / name match is a valid id but NOT schema-clean.
    """
    text = raw.strip().lower()

    if text in ids:
        return text, True

    # earliest id mentioned anywhere in the text
    found = None
    earliest = len(text) + 1
    for candidate in ids:
        idx = text.find(candidate)
        if idx != -1 and idx < earliest:
            earliest = idx
            found = candidate
    if found is not None:
        return found, False

    # fall back to matching a short name (longest first to avoid partials)
    for name in sorted(name_to_id, key=len, reverse=True):
        if name and name in text:
            return name_to_id[name], False

    return None, False


def predict_tag(model, item: dict, chosen: str, misconceptions: list[dict], runs: int = 1):
    """Tag ONE (item, chosen-distractor) pair -> modal misconception id.

    Reusable entry point so other harnesses (e.g. litmus_generation.py's
    tag-fidelity verifier) can call the tagger WITHOUT duplicating the prompt /
    parsing logic. The gold/claimed tag is never shown to the model. Returns
    (modal_id_or_None, mean_schema_valid_rate).
    """
    ids = [m["id"] for m in misconceptions]
    name_to_id = {m["name"].strip().lower(): m["id"] for m in misconceptions}
    user = build_user_prompt({"item": item, "chosen": chosen}, misconceptions)

    preds, clean_flags = [], []
    for _ in range(max(1, runs)):
        raw = model.generate(SYSTEM_PROMPT, user)
        pid, is_clean = parse_misconception(raw, ids, name_to_id)
        preds.append(pid)
        clean_flags.append(is_clean)

    modal = Counter(preds).most_common(1)[0][0]
    return modal, sum(clean_flags) / len(clean_flags)


# --------------------------------------------------------------------- models

class LitmusDummyModel:
    """Fake tagger for --selftest. Reuses DummyModel's spirit (model_utils):
    keyed lookup of the built prompt -> gold, with run-to-run noise so the
    consistency / schema / accuracy plumbing is exercised end to end.
    """

    def __init__(self, prompt_to_gold: dict[str, str], ids: list[str]):
        self._map = prompt_to_gold
        self._ids = ids
        self._counter = 0

    def generate(self, system: str, user: str) -> str:
        self._counter += 1
        rng = random.Random(hash(user) + self._counter)
        gold = self._map.get(user)
        roll = rng.random()
        if gold is None:
            return rng.choice(self._ids)
        if roll < 0.70:
            return gold  # correct + clean
        if roll < 0.82:
            return f"The misconception is {gold}."  # correct but noisy schema
        if roll < 0.90:
            return "unknown_misconception"  # unparseable
        return rng.choice(self._ids)  # wrong


class SkipFrontier(Exception):
    """Raised when no usable frontier API key / library is available."""


class FrontierModel:
    """Optional frontier-API backend (OpenAI or Anthropic). Zero-shot only.

    Interface matches HFClassifier/DummyModel: generate(system, user) -> str.
    Raises SkipFrontier (never crashes the run) when the key/lib is missing.
    """

    def __init__(self, provider: str, model_name: str | None, temperature: float):
        self.temperature = temperature
        if provider == "auto":
            if os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            else:
                raise SkipFrontier(
                    "no OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment"
                )
        self.provider = provider

        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise SkipFrontier("OPENAI_API_KEY is not set")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise SkipFrontier(f"openai library not installed ({exc})") from exc
            self._client = OpenAI()
            self.model_name = model_name or "gpt-4o-mini"
        elif provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SkipFrontier("ANTHROPIC_API_KEY is not set")
            try:
                import anthropic
            except ImportError as exc:
                raise SkipFrontier(f"anthropic library not installed ({exc})") from exc
            self._client = anthropic.Anthropic()
            self.model_name = model_name or "claude-3-5-sonnet-latest"
        else:
            raise SkipFrontier(f"unknown frontier provider: {provider}")

    def generate(self, system: str, user: str) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=24,
            )
            return resp.choices[0].message.content or ""
        resp = self._client.messages.create(
            model=self.model_name,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=24,
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


# --------------------------------------------------------------------- eval

def evaluate(model, examples, misconceptions, runs):
    ids = [m["id"] for m in misconceptions]
    name_to_id = {m["name"].strip().lower(): m["id"] for m in misconceptions}
    id_to_coarse = {m["id"]: m["coarse"] for m in misconceptions}

    per_example = []
    for example in examples:
        user = build_user_prompt(example, misconceptions)
        preds, clean_flags = [], []
        for _ in range(runs):
            raw = model.generate(SYSTEM_PROMPT, user)
            pid, is_clean = parse_misconception(raw, ids, name_to_id)
            preds.append(pid)
            clean_flags.append(is_clean)

        counts = Counter(preds)
        modal, modal_count = counts.most_common(1)[0]
        consistency = modal_count / len(preds)
        pred_coarse = id_to_coarse.get(modal)

        per_example.append(
            {
                "item_id": example["item_id"],
                "chosen": example["chosen"],
                "gold": example["gold_misconception"],
                "gold_coarse": example["gold_coarse"],
                "pred": modal,
                "pred_coarse": pred_coarse,
                "preds": preds,
                "consistency": consistency,
                "schema_valid_rate": sum(clean_flags) / len(clean_flags),
                "correct": modal == example["gold_misconception"],
                "coarse_correct": pred_coarse is not None
                and pred_coarse == example["gold_coarse"],
            }
        )
    return per_example


def summarize(results, examples, backend_label):
    n = len(results)
    acc = sum(r["correct"] for r in results) / n
    coarse_acc = sum(r["coarse_correct"] for r in results) / n
    mean_consistency = sum(r["consistency"] for r in results) / n
    mean_schema = sum(r["schema_valid_rate"] for r in results) / n

    print("\n=== LITMUS TAGGING RESULTS ===")
    print(f"Backend: {backend_label}")
    print(f"Examples (item x chosen-distractor): {n}")
    print(f"Mid-grained tagging accuracy (modal vs gold): {acc:.1%}")
    print(f"Coarse cognitive-kind accuracy (mapped up):   {coarse_acc:.1%}")
    print(f"Mean consistency (modal agreement across runs): {mean_consistency:.1%}")
    print(f"Mean schema validity (clean single id):         {mean_schema:.1%}")

    print("\nPer-example:")
    header = f"{'item':<14}{'ch':<4}{'gold':<34}{'pred':<34}{'ok':<4}{'cok':<4}{'consist':<9}{'schema'}"
    print(header)
    for r in results:
        print(
            f"{r['item_id']:<14}{r['chosen']:<4}{r['gold']:<34}{str(r['pred']):<34}"
            f"{'Y' if r['correct'] else 'N':<4}{'Y' if r['coarse_correct'] else 'N':<4}"
            f"{r['consistency']:<9.0%}{r['schema_valid_rate']:.0%}"
        )

    print("\nMid-grained confusions (gold -> predicted), errors only:")
    confusions = summarize_confusion(
        [{"gold": r["gold"], "pred": r["pred"], "correct": r["correct"]} for r in results]
    )
    if not confusions:
        print("  (none)")
    for (gold, pred), count in sorted(confusions.items(), key=lambda kv: -kv[1]):
        print(f"  {gold:<34} -> {str(pred):<34} x{count}")

    print("\nCoarse confusions (gold -> predicted), errors only:")
    coarse_conf = summarize_confusion(
        [
            {"gold": r["gold_coarse"], "pred": r["pred_coarse"], "correct": r["coarse_correct"]}
            for r in results
        ]
    )
    if not coarse_conf:
        print("  (none)")
    for (gold, pred), count in sorted(coarse_conf.items(), key=lambda kv: -kv[1]):
        print(f"  {gold:<28} -> {str(pred):<28} x{count}")

    print("\nPer-misconception support (gold count / accuracy / mean consistency):")
    by_gold = defaultdict(list)
    for r in results:
        by_gold[r["gold"]].append(r)
    for gold in sorted(by_gold):
        rows = by_gold[gold]
        g_acc = sum(x["correct"] for x in rows) / len(rows)
        g_con = sum(x["consistency"] for x in rows) / len(rows)
        print(f"  {gold:<34} n={len(rows):<3} acc={g_acc:>5.0%} consist={g_con:>5.0%}")

    print("\n--- HOW TO READ THIS (the pivot 2x2) ---")
    print("  The bar is CONSISTENCY, not one-shot accuracy.")
    print("  * base 1.7B already consistent + accurate  -> NO fine-tuning project.")
    print("  * base 1.7B not, but a frontier model is    -> distill frontier -> deployable SLM.")
    print("  * neither is consistent                     -> fine-tuning is justified;")
    print("    then check the misconception labels are not ambiguous (low per-label")
    print("    accuracy with high consistency = the model is confidently wrong, often")
    print("    a label-ambiguity problem, not a capability one).")
    print("  Low schema validity alone = a parsing/prompt-format gap, not a capability gap.")
    print("  NOTE: DEFENSIBLE numbers need ~40-50 REAL tagged items + a GPU (base 1.7B)")
    print("  and, for the frontier arm, an API key. This seed set is AUTHORED PLACEHOLDER.")


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF model id (default path: Qwen/Qwen3-1.7B)")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--misconceptions", default=str(DEFAULT_MISCONCEPTIONS))
    parser.add_argument("--runs", type=int, default=5, help="repeats for the consistency metric")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--selftest", action="store_true", help="use the local dummy tagger")
    parser.add_argument(
        "--frontier",
        default=None,
        choices=["openai", "anthropic", "auto"],
        help="optional frontier-API arm (needs API key + lib; skips cleanly if absent)",
    )
    parser.add_argument("--frontier-model", default=None, help="override the frontier model name")
    parser.add_argument("--save-predictions", default=None, help="write per-example JSONL")
    args = parser.parse_args()

    misconceptions = load_misconceptions(args.misconceptions)
    id_to_coarse = {m["id"]: m["coarse"] for m in misconceptions}
    print(f"Loaded {len(misconceptions)} mid-grained misconceptions from {args.misconceptions}")

    items = load_jsonl(args.data)
    examples = build_examples(items, id_to_coarse)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    print(
        f"Loaded {len(items)} items -> {len(examples)} tagging examples "
        f"(item x chosen-distractor) from {args.data}"
    )

    if args.frontier:
        try:
            model = FrontierModel(args.frontier, args.frontier_model, args.temperature)
            backend_label = f"frontier:{model.provider}:{model.model_name}"
            print(f"Using frontier backend: {backend_label}")
        except SkipFrontier as exc:
            print(f"\n[frontier SKIPPED] {exc}.")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY (and install the client) to run it.")
            return
    elif args.selftest:
        print("Running SELF-TEST with the local dummy tagger (no GPU / no download).")
        prompt_to_gold = {
            build_user_prompt(ex, misconceptions): ex["gold_misconception"] for ex in examples
        }
        model = LitmusDummyModel(prompt_to_gold, [m["id"] for m in misconceptions])
        backend_label = "dummy (selftest)"
    else:
        model_id = args.model or "Qwen/Qwen3-1.7B"
        print(f"Loading HF model: {model_id}")
        model = HFClassifier(model_id, temperature=args.temperature)
        backend_label = f"hf:{model_id}"

    results = evaluate(model, examples, misconceptions, runs=args.runs)
    summarize(results, examples, backend_label)

    if args.save_predictions:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as handle:
            for r in results:
                handle.write(json.dumps(r) + "\n")
        print(f"\nSaved predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
