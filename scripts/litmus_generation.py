"""
Litmus GENERATION harness for the SLM pivot (prompted, zero/few-shot; NOT fine-tuned).

The pivot's PRIMARY v1 task is now CONDITIONAL GENERATION: given a spec
`{topic, target misconception(s) per distractor, difficulty, format}`, produce a
full misconception-TAGGED AP Bio item (stem/passage, choices, correct answer,
per-distractor misconception tag). This is spec->item, NOT free generation, so
the recommendation layer can request exactly what it needs and eval is checkable.

This harness builds the MEASUREMENT that shows whether generation beats baseline
-- it is NOT a production rejection-sampling / data-cleaning pipeline (see
docs/mcat_pivot_spec.md and docs/litmus_plan.md). It implements the verifiers
that ARE the metric:

  V1  Structural validity (deterministic code): well-formed schema, exactly one
      correct answer, N distractors, every distractor tagged with a taxonomy id
      from apbio_misconceptions.json, no duplicate/degenerate choices.  [anti-drift]
  spec-adherence: does the generated item match the requested topic + target
      misconception(s)?
  V3  Tag-fidelity: an INDEPENDENT tagger (litmus_tagging.predict_tag, a DIFFERENT
      model than the generator) re-reads each generated distractor and predicts a
      misconception; we check agreement with the generator's CLAIMED tag.  [crux]

Deliberately NOT built for v1 (documented as prod/future work): the solvability
"solve-it-blind" solver arm (V2), and the full rejection-sampling / scaled-human
pipeline. A small human-labeled sample (~30-50 items) is required to anchor the
tag-fidelity verifier (verifier<->human agreement) -- described in the docs, not
built here.

Independence caveat (enforced here): the tag-fidelity verifier must be a
DIFFERENT model than the generator (a frontier model or the separately-trained
tagger), never the generator grading itself. If no independent verifier is
configured, tag-fidelity is reported as N/A rather than faked.

Usage:
  # Local smoke test (Windows CPU: dummy path only):
  python scripts/litmus_generation.py --selftest --num-specs 5 --runs 1

  # Base Qwen3-1.7B generator on GPU, frontier verifier (Colab):
  python scripts/litmus_generation.py --model Qwen/Qwen3-1.7B --verifier-frontier auto --num-specs 20 --runs 2

  # Prompted frontier generator (the teacher / bar to beat):
  python scripts/litmus_generation.py --frontier auto --verifier-model Qwen/Qwen3-1.7B --num-specs 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse, don't reinvent: the tagger (as the independent verifier), model loaders,
# the misconception loader, and the frontier backend all come from litmus_tagging.
from litmus_tagging import (  # noqa: E402
    FrontierModel,
    HFClassifier,
    SkipFrontier,
    load_misconceptions,
    predict_tag,
)

DEFAULT_MISCONCEPTIONS = REPO_ROOT / "data" / "apbio_misconceptions.json"
DEFAULT_TEMPLATE = REPO_ROOT / "data" / "apbio_item_template.jsonl"

# Topics that make sense to render as passage/experimental-design items.
PASSAGE_TOPICS = {"experimental_design", "genetics"}
DIFFICULTIES = ["easy", "medium", "hard"]


# --------------------------------------------------------------------- specs

def build_specs(misconceptions, num_specs, n_choices=4, seed=0):
    """Build conditional-generation specs from the taxonomy.

    Each spec asks for one item with (n_choices - 1) distractors, each targeting
    a specific in-topic misconception id. Deterministic given the seed.
    """
    by_topic = defaultdict(list)
    for m in misconceptions:
        by_topic[m["topic"]].append(m["id"])

    topics = sorted(by_topic)
    rng = random.Random(seed)
    specs = []
    n_distractors = n_choices - 1
    i = 0
    while len(specs) < num_specs:
        topic = topics[i % len(topics)]
        pool = by_topic[topic]
        if len(pool) >= n_distractors:
            targets = rng.sample(pool, n_distractors)
        else:  # small topic: sample with replacement so the spec is still well-formed
            targets = [rng.choice(pool) for _ in range(n_distractors)]
        specs.append(
            {
                "spec_id": f"spec_{len(specs):03d}_{topic}",
                "topic": topic,
                "difficulty": DIFFICULTIES[len(specs) % len(DIFFICULTIES)],
                "format": "passage" if topic in PASSAGE_TOPICS else "standalone",
                "n_choices": n_choices,
                "targets": targets,
            }
        )
        i += 1
    return specs


# --------------------------------------------------------------------- prompting

GEN_SYSTEM_PROMPT = (
    "You are an expert AP Biology item writer. You generate ONE multiple-choice "
    "item that exactly matches a given spec, with every wrong answer (distractor) "
    "deliberately built to embody a specific named misconception. You output ONLY "
    "a single JSON object in the required schema, with no prose, no markdown fences, "
    "and no commentary."
)


def _candidate_block(misconceptions, target_ids):
    """Show the target misconceptions (id: name -- description) the item must embody."""
    by_id = {m["id"]: m for m in misconceptions}
    lines = []
    for mid in target_ids:
        m = by_id.get(mid)
        if m:
            lines.append(f"  {m['id']}: {m['name']} -- {m['description']}")
    return "\n".join(lines)


def _exemplar_block(template_items):
    if not template_items:
        return ""
    ex = template_items[0]
    slim = {
        "id": ex.get("id"),
        "topic": ex.get("topic"),
        "difficulty": ex.get("difficulty"),
        "passage": ex.get("passage"),
        "stem": ex.get("stem"),
        "choices": ex.get("choices"),
        "correct": ex.get("correct"),
        "distractor_tags": {
            k: {
                "error_type": v.get("error_type"),
                "misconception_id": v.get("misconception_id", "<a taxonomy id>"),
                "misconception": v.get("misconception"),
                "rationale": v.get("rationale"),
            }
            for k, v in ex.get("distractor_tags", {}).items()
        },
    }
    return "Example of the required JSON shape (structure only, not content):\n" + json.dumps(
        slim, indent=2
    )


def build_gen_prompt(spec, misconceptions, template_items):
    n = spec["n_choices"]
    n_distractors = n - 1
    keys = [chr(ord("A") + i) for i in range(n)]
    passage_rule = (
        "Include a short 'passage' (a few sentences of data / experimental setup) "
        "since format is 'passage'."
        if spec["format"] == "passage"
        else "Set 'passage' to null since format is 'standalone'."
    )
    targets = spec["targets"]
    target_lines = "\n".join(
        f"  - one distractor for: {mid}" for mid in targets
    )

    return f"""Generate ONE AP Biology multiple-choice item that matches this spec.

Spec:
  topic: {spec['topic']}
  difficulty: {spec['difficulty']}
  format: {spec['format']}
  choices: {n} total ({keys[0]}..{keys[-1]}); EXACTLY ONE is correct.
  distractors: {n_distractors}, each embodying one target misconception below.

Target misconceptions (each must be embodied by exactly one distractor, and tagged
with its id in distractor_tags):
{_candidate_block(misconceptions, targets)}

Requirements:
- Output a single JSON object with keys: id, topic, difficulty, passage, stem,
  choices, correct, distractor_tags.
- 'choices' is an object mapping each of {keys} to distinct, plausible answer text.
- 'correct' is the single key of the correct choice.
- 'distractor_tags' maps EACH non-correct choice key to an object with:
  error_type, misconception_id (one of the target ids above, exactly), misconception
  (short text), rationale.
- Distractor assignment (one per target):
{target_lines}
- {passage_rule}
- No two choices may be identical; no empty choices.
- Output ONLY the JSON object. No markdown, no explanation.

{_exemplar_block(template_items)}

JSON:"""


# --------------------------------------------------------------------- parsing

def parse_generated_item(raw: str):
    """Robustly extract a JSON item from model output. Malformed -> None (drift)."""
    if not raw:
        return None
    text = raw.strip()
    # strip markdown fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text)
    # 1) whole thing
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # 2) first balanced {...} block
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


# --------------------------------------------------------------------- verifiers

def verify_structural(item, taxonomy_ids):
    """V1 deterministic structural validity. Returns (ok, reasons)."""
    reasons = []
    if not isinstance(item, dict):
        return False, ["not a JSON object"]

    stem = item.get("stem")
    if not isinstance(stem, str) or not stem.strip():
        reasons.append("missing/empty stem")

    choices = item.get("choices")
    if not isinstance(choices, dict) or len(choices) < 3:
        reasons.append("needs >=3 choices")
        return False, reasons

    texts = [str(v).strip().lower() for v in choices.values()]
    if any(not t for t in texts):
        reasons.append("empty choice text")
    if len(set(texts)) != len(texts):
        reasons.append("duplicate/degenerate choices")

    correct = item.get("correct")
    if correct not in choices:
        reasons.append("correct not a valid choice key")

    tags = item.get("distractor_tags")
    if not isinstance(tags, dict):
        reasons.append("missing distractor_tags")
        return False, reasons

    expected_distractors = {k for k in choices if k != correct}
    if set(tags) != expected_distractors:
        reasons.append("distractor_tags keys != non-correct choices")

    for key, tag in tags.items():
        if not isinstance(tag, dict):
            reasons.append(f"tag {key} not an object")
            continue
        mid = tag.get("misconception_id")
        if mid not in taxonomy_ids:
            reasons.append(f"tag {key} misconception_id not in taxonomy")

    return (len(reasons) == 0), reasons


def verify_spec_adherence(item, spec):
    """Did the item match the requested topic + target misconception(s)?"""
    reasons = []
    if str(item.get("topic", "")).strip().lower() != spec["topic"].lower():
        reasons.append("topic mismatch")
    claimed = {
        tag.get("misconception_id")
        for tag in item.get("distractor_tags", {}).values()
        if isinstance(tag, dict)
    }
    missing = [t for t in spec["targets"] if t not in claimed]
    if missing:
        reasons.append(f"missing target misconceptions: {missing}")
    return (len(reasons) == 0), reasons


def verify_tag_fidelity(item, verifier, tagger_runs):
    """V3: independent tagger vs the generator's CLAIMED tag, per distractor.

    Returns (n_agree, n_checked, details). verifier is None -> caller skips.
    """
    n_agree = 0
    n_checked = 0
    details = []
    for key, tag in item.get("distractor_tags", {}).items():
        claimed = tag.get("misconception_id") if isinstance(tag, dict) else None
        if claimed is None:
            continue
        pred, _ = verifier.predict(item, key, runs=tagger_runs)
        agree = pred == claimed
        n_checked += 1
        n_agree += int(agree)
        details.append({"choice": key, "claimed": claimed, "verifier": pred, "agree": agree})
    return n_agree, n_checked, details


# --------------------------------------------------------------------- verifier backends

class ModelTagVerifier:
    """Independent tag-fidelity verifier backed by a real model (frontier or a
    separate HF tagger). Reuses litmus_tagging.predict_tag."""

    def __init__(self, model, misconceptions, label):
        self.model = model
        self.misconceptions = misconceptions
        self.label = label

    def predict(self, item, chosen, runs=1):
        return predict_tag(self.model, item, chosen, self.misconceptions, runs=runs)


class DummyTagVerifier:
    """Stand-in verifier for --selftest: simulates a DECENT-but-imperfect
    independent tagger (agrees with the claimed tag ~70% of the time, else emits
    a different id) so the fidelity plumbing runs without a model."""

    label = "dummy-tagger (selftest)"

    def __init__(self, ids, agree_p=0.70, seed=0):
        self._ids = ids
        self._agree_p = agree_p
        self._counter = 0

    def predict(self, item, chosen, runs=1):
        self._counter += 1
        tag = item.get("distractor_tags", {}).get(chosen, {})
        claimed = tag.get("misconception_id") if isinstance(tag, dict) else None
        rng = random.Random(hash((item.get("stem"), chosen)) + self._counter)
        if claimed is not None and rng.random() < self._agree_p:
            return claimed, 1.0
        return rng.choice(self._ids), 1.0


# --------------------------------------------------------------------- generators

class DummyGenerator:
    """Fake generator for --selftest. Emits schema-ish JSON from the spec with a
    realistic amount of DRIFT (malformed JSON, dropped tags, wrong topic) so the
    validity / adherence / fidelity metrics are all exercised end to end."""

    def __init__(self, prompt_to_spec, misconceptions, seed=0):
        self._map = prompt_to_spec
        self._by_id = {m["id"]: m for m in misconceptions}
        self._id_to_coarse = {m["id"]: m["coarse"] for m in misconceptions}
        self._counter = 0

    def _item_from_spec(self, spec, rng):
        keys = [chr(ord("A") + i) for i in range(spec["n_choices"])]
        correct_key = keys[0]
        choices = {correct_key: f"Correct answer about {spec['topic']} (v{rng.randint(1, 9999)})"}
        tags = {}
        for key, mid in zip(keys[1:], spec["targets"]):
            m = self._by_id.get(mid, {})
            choices[key] = f"Distractor: {m.get('name', mid)} (v{rng.randint(1, 9999)})"
            tags[key] = {
                "error_type": self._id_to_coarse.get(mid),
                "misconception_id": mid,
                "misconception": m.get("name", mid),
                "rationale": f"Embodies {mid}.",
            }
        return {
            "id": spec["spec_id"] + f"_gen{rng.randint(0, 9999)}",
            "topic": spec["topic"],
            "difficulty": spec["difficulty"],
            "passage": "Short passage." if spec["format"] == "passage" else None,
            "stem": f"Generated {spec['difficulty']} {spec['topic']} question {rng.randint(0, 999999)}?",
            "choices": choices,
            "correct": correct_key,
            "distractor_tags": tags,
        }

    def generate(self, system, user):
        self._counter += 1
        spec = self._map.get(user)
        rng = random.Random(hash(user) + self._counter)
        if spec is None:
            return "{}"
        item = self._item_from_spec(spec, rng)
        roll = rng.random()
        if roll < 0.15:  # malformed JSON -> invalid (drift)
            return json.dumps(item)[: len(json.dumps(item)) // 2]
        if roll < 0.27:  # drop one distractor tag -> structural invalid (drift)
            if item["distractor_tags"]:
                item["distractor_tags"].pop(next(iter(item["distractor_tags"])))
        elif roll < 0.37:  # wrong topic -> spec-adherence failure
            item["topic"] = "unrelated_topic"
        text = json.dumps(item)
        if roll > 0.7:  # sometimes wrap in prose to test robust extraction
            text = "Here is the item you requested:\n" + text + "\nDone."
        return text


# --------------------------------------------------------------------- eval

def _norm_stem(item):
    return re.sub(r"\s+", " ", str(item.get("stem", "")).strip().lower())


def run_generation(generator, verifier, specs, misconceptions, runs, tagger_runs):
    taxonomy_ids = {m["id"] for m in misconceptions}
    rows = []
    seen_stems = set()  # dedup pattern reused from generate_dataset._try_add_example

    for spec in specs:
        prompt = spec["_prompt"]
        for _ in range(runs):
            raw = generator.generate(GEN_SYSTEM_PROMPT, prompt)
            item = parse_generated_item(raw)

            row = {
                "spec_id": spec["spec_id"],
                "topic": spec["topic"],
                "parsed": item is not None,
                "valid": False,
                "valid_reasons": [],
                "spec_ok": False,
                "spec_reasons": [],
                "fidelity_agree": 0,
                "fidelity_checked": 0,
                "faithful": False,
                "duplicate": False,
            }

            if item is None:
                row["valid_reasons"] = ["unparseable JSON"]
                rows.append(row)
                continue

            valid, vreasons = verify_structural(item, taxonomy_ids)
            row["valid"] = valid
            row["valid_reasons"] = vreasons

            spec_ok, sreasons = verify_spec_adherence(item, spec)
            row["spec_ok"] = spec_ok
            row["spec_reasons"] = sreasons

            stem = _norm_stem(item)
            if stem and stem in seen_stems:
                row["duplicate"] = True
            elif stem:
                seen_stems.add(stem)

            if valid and verifier is not None:
                agree, checked, _ = verify_tag_fidelity(item, verifier, tagger_runs)
                row["fidelity_agree"] = agree
                row["fidelity_checked"] = checked
                row["faithful"] = checked > 0 and agree == checked

            rows.append(row)
    return rows


def summarize(rows, backend_label, verifier_label, fidelity_on):
    n = len(rows)
    n_parsed = sum(r["parsed"] for r in rows)
    n_valid = sum(r["valid"] for r in rows)
    n_spec = sum(r["spec_ok"] for r in rows)
    n_dupe = sum(r["duplicate"] for r in rows)
    n_unique = n - n_dupe

    total_checked = sum(r["fidelity_checked"] for r in rows)
    total_agree = sum(r["fidelity_agree"] for r in rows)
    n_faithful = sum(r["faithful"] for r in rows)
    n_valid_faithful = sum(r["valid"] and r["faithful"] for r in rows)

    def pct(a, b):
        return f"{(a / b):.1%}" if b else "n/a"

    print("\n=== LITMUS GENERATION RESULTS ===")
    print(f"Generator backend: {backend_label}")
    print(f"Tag-fidelity verifier: {verifier_label}")
    print(f"Items generated (specs x runs): {n}")
    print(f"Parse rate (valid JSON):            {pct(n_parsed, n)}")
    print(f"V1 structural validity rate:        {pct(n_valid, n)}   [anti-drift]")
    print(f"Spec-adherence rate:                {pct(n_spec, n)}")
    print(f"Diversity / dedup rate (unique):    {pct(n_unique, n)}")
    if fidelity_on:
        print(f"V3 tag-fidelity rate (per distractor): {pct(total_agree, total_checked)}   [crux]")
        print(f"  (distractors agreeing: {total_agree}/{total_checked})")
        print(f"Items fully faithful (all distractors): {pct(n_faithful, n)}")
        print(f"YIELD (valid AND faithful / generated): {pct(n_valid_faithful, n)}")
    else:
        print("V3 tag-fidelity: SKIPPED (no independent verifier configured).")
        print("  Configure --verifier-frontier or --verifier-model (a DIFFERENT model")
        print("  than the generator) to compute the crux tag-fidelity metric.")
        print(f"YIELD (valid only, fidelity N/A):   {pct(n_valid, n)}")

    print("\nTop structural-invalidity reasons (the drift signal):")
    reason_counts = Counter()
    for r in rows:
        if not r["valid"]:
            for reason in (r["valid_reasons"] or ["(unknown)"]):
                reason_counts[reason] += 1
    if not reason_counts:
        print("  (none - all structurally valid)")
    for reason, count in reason_counts.most_common(8):
        print(f"  {reason:<44} x{count}")

    print("\nPer-topic validity / spec-adherence:")
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    for topic in sorted(by_topic):
        tr = by_topic[topic]
        print(
            f"  {topic:<22} n={len(tr):<3} valid={pct(sum(x['valid'] for x in tr), len(tr)):>6} "
            f"spec={pct(sum(x['spec_ok'] for x in tr), len(tr)):>6}"
        )

    print("\n--- HOW TO READ THIS (the generation 2x2 / distillation gate) ---")
    print("  Thesis: controllable, anti-drift STRUCTURED GENERATION (not calibrated")
    print("  abstention). The win is reliably schema'd, validly-TAGGED items.")
    print("  * base 1.7B validity/fidelity/yield already high -> a prompt suffices;")
    print("    the SLM's value is deployment efficiency (cheap/on-device at scale).")
    print("  * base 1.7B DRIFTS (low validity/fidelity) but a prompted FRONTIER model")
    print("    is high -> distill the frontier teacher into a deployable small model")
    print("    that generates valid tagged items where base-small can't. <- the pivot.")
    print("  * neither reliable -> the task/taxonomy or schema needs rework first.")
    print("  Independence: the verifier is a DIFFERENT model than the generator; a")
    print("  ~30-50-item human-labeled sample must anchor verifier<->human agreement")
    print("  before the tag-fidelity number is trusted. This seed run uses PLACEHOLDER")
    print("  specs; DEFENSIBLE numbers need a GPU (base 1.7B) + a frontier API key.")


# --------------------------------------------------------------------- main

def _load_template_items(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except FileNotFoundError:
        return []


def _make_generator(args, specs, misconceptions):
    """Return (generator, backend_label). Handles --frontier / --selftest / --model."""
    if args.frontier:
        model = FrontierModel(args.frontier, args.frontier_model, args.temperature)
        return model, f"frontier:{model.provider}:{model.model_name}"
    if args.selftest:
        prompt_to_spec = {spec["_prompt"]: spec for spec in specs}
        return DummyGenerator(prompt_to_spec, misconceptions), "dummy-generator (selftest)"
    model_id = args.model or "Qwen/Qwen3-1.7B"
    print(f"Loading HF generator: {model_id}")
    return HFClassifier(model_id, temperature=args.temperature), f"hf:{model_id}"


def _make_verifier(args, misconceptions):
    """Return (verifier_or_None, label, fidelity_on). Enforces independence."""
    ids = [m["id"] for m in misconceptions]
    if args.selftest:
        return DummyTagVerifier(ids), DummyTagVerifier.label, True
    if args.verifier_frontier:
        try:
            model = FrontierModel(args.verifier_frontier, args.verifier_frontier_model, 0.0)
            label = f"frontier:{model.provider}:{model.model_name}"
            return ModelTagVerifier(model, misconceptions, label), label, True
        except SkipFrontier as exc:
            print(f"[verifier SKIPPED] frontier verifier unavailable: {exc}")
            return None, "none", False
    if args.verifier_model:
        print(f"Loading HF verifier (independent tagger): {args.verifier_model}")
        model = HFClassifier(args.verifier_model, temperature=0.0)
        label = f"hf:{args.verifier_model}"
        return ModelTagVerifier(model, misconceptions, label), label, True
    return None, "none (independence requires a separate verifier)", False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF generator model id (default Qwen/Qwen3-1.7B)")
    parser.add_argument("--misconceptions", default=str(DEFAULT_MISCONCEPTIONS))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="few-shot exemplar items")
    parser.add_argument("--num-specs", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=None, help="cap specs after building")
    parser.add_argument("--n-choices", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1, help="generations per spec")
    parser.add_argument("--tagger-runs", type=int, default=1, help="verifier repeats per distractor")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selftest", action="store_true", help="dummy generator + dummy verifier")
    parser.add_argument(
        "--frontier",
        default=None,
        choices=["openai", "anthropic", "auto"],
        help="prompted frontier GENERATOR arm (the teacher). Skips cleanly if no key/lib.",
    )
    parser.add_argument("--frontier-model", default=None)
    parser.add_argument(
        "--verifier-frontier",
        default=None,
        choices=["openai", "anthropic", "auto"],
        help="independent tag-fidelity verifier via frontier API (must differ from generator)",
    )
    parser.add_argument("--verifier-frontier-model", default=None)
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="independent tag-fidelity verifier via a separate HF tagger model id",
    )
    parser.add_argument("--save-predictions", default=None, help="write per-item JSONL")
    args = parser.parse_args()

    misconceptions = load_misconceptions(args.misconceptions)
    print(f"Loaded {len(misconceptions)} misconceptions from {args.misconceptions}")

    specs = build_specs(misconceptions, args.num_specs, n_choices=args.n_choices, seed=args.seed)
    if args.max_examples is not None:
        specs = specs[: args.max_examples]
    template_items = _load_template_items(args.template)
    for spec in specs:
        spec["_prompt"] = build_gen_prompt(spec, misconceptions, template_items)
    print(f"Built {len(specs)} conditional-generation specs across "
          f"{len({s['topic'] for s in specs})} topics.")

    # Generator arm.
    if args.frontier:
        try:
            generator, backend_label = _make_generator(args, specs, misconceptions)
            print(f"Generator backend: {backend_label}")
        except SkipFrontier as exc:
            print(f"\n[generator SKIPPED] {exc}.")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY (and install the client) to run it.")
            return
    else:
        generator, backend_label = _make_generator(args, specs, misconceptions)

    verifier, verifier_label, fidelity_on = _make_verifier(args, misconceptions)

    rows = run_generation(
        generator, verifier, specs, misconceptions,
        runs=args.runs, tagger_runs=args.tagger_runs,
    )
    summarize(rows, backend_label, verifier_label, fidelity_on)

    if args.save_predictions:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
        print(f"\nSaved per-item verifier results to {args.save_predictions}")


if __name__ == "__main__":
    main()
