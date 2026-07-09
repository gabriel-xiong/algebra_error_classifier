"""
Base-vs-tuned GENERATION eval — the make-or-break piece (see docs/behavior_spec.md).

For each held-out scenario (data/eval_scenarios.jsonl) we build the exact
generation prompt used in SFT, ask BOTH a base and a tuned model to generate,
parse the JSON item, and score each output on the Behavior Spec rubric:

  spec_adherence      programmatic, ALL topics (valid JSON? 4 distinct choices?
                      one correct? every wrong option tagged with EXACTLY the
                      requested misconceptions, one each?)  <- the behavioral check
  distractor_mapping  genetics: recomputed from the model's own declared cross
                      (objective); conceptual: LLM judge
  task_quality        genetics: answer recomputed-correct + distractors wrong;
                      conceptual: LLM judge

Reports mean per dimension base vs tuned + delta, a reliability rate (% fully
on-spec), and the forbidden-failure rate (% with a mis-mapped distractor).

Generators (--base/--tuned):
  hf:<path_or_hub_id>   a HuggingFace causal LM (transformers; runs on GPU/Colab)
  mock:good             returns a by-construction correct item (simulated tuned)
  mock:bad              returns a corrupted item (simulated base) — for offline
                        plumbing tests and to show a non-zero delta
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import conceptual_engine as engine
import gen_cellresp
import gen_enzymes
import gen_genetics
import gen_spec
import judge as judge_mod
from validate_corpus import (_offspring, _p_dominant, _p_hom_rec, _p_hom_dom,
                             _frac, _CANONICAL)

REPO = Path(__file__).resolve().parent.parent
DIMS = ["spec_adherence", "distractor_mapping", "task_quality"]
_CONCEPTUAL = {gen_cellresp.TOPIC: (gen_cellresp.FRAMES, gen_cellresp.ERROR_TYPE),
               gen_enzymes.TOPIC: (gen_enzymes.FRAMES, gen_enzymes.ERROR_TYPE)}


# ------------------------------------------------------------------ generators

def _matching_item(scenario: dict, rng: random.Random):
    """Produce a by-construction item that satisfies the scenario (for mocks)."""
    want = scenario["misconception_ids"]
    if scenario["topic"] == "genetics":
        for _ in range(2000):
            it = gen_genetics.generate_item(rng, 0)
            if it and gen_spec.misconception_ids_of(it) == want:
                return it
    else:
        frames, err = _CONCEPTUAL[scenario["topic"]]
        for _ in range(4000):
            it = engine.generate_item(rng, 0, frames=frames, topic=scenario["topic"],
                                      error_type=err, id_prefix="mock", generator="mock")
            if (it and it["authoring"]["frame"] == scenario.get("frame")
                    and gen_spec.misconception_ids_of(it) == want):
                return it
    raise RuntimeError(f"could not build a matching item for {scenario['scenario_id']}")


class MockGenerator:
    def __init__(self, kind: str, seed: int = 0):
        self.kind = kind  # "good" | "bad"
        self.rng = random.Random(seed)

    def generate(self, scenario: dict) -> str:
        item = _matching_item(scenario, self.rng)
        target = gen_spec.item_to_target(item)
        if self.kind == "good":
            return json.dumps(target)
        # "bad": rotate the misconception_id labels among distractors so each
        # distractor's TEXT no longer matches its tag -> mapping should fail,
        # and half the time wrap in prose to break spec_adherence too.
        letters = list(target["distractor_tags"])
        mids = [target["distractor_tags"][L]["misconception_id"] for L in letters]
        rot = mids[1:] + mids[:1]
        for L, m in zip(letters, rot):
            target["distractor_tags"][L]["misconception_id"] = m
        blob = json.dumps(target)
        if self.rng.random() < 0.5:
            return "Sure! Here is the item:\n" + blob + "\nHope that helps."
        return blob


class HFGenerator:
    """transformers causal LM. Lazy imports so this file loads without torch."""

    def __init__(self, path: str, max_new_tokens: int = 512, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype="auto", device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, scenario: dict) -> str:
        system, user = gen_spec.build_generation_prompt(
            scenario["topic"], scenario["misconception_ids"])
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        try:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)


def make_generator(spec: str):
    if spec.startswith("mock:"):
        return MockGenerator(spec.split(":", 1)[1])
    if spec.startswith("hf:"):
        return HFGenerator(spec.split(":", 1)[1])
    raise ValueError(f"unknown generator spec: {spec} (use mock:good|mock:bad|hf:<path>)")


# ------------------------------------------------------------------ scoring

def _parse_item(raw: str) -> dict | None:
    obj = judge_mod._extract_json(raw)
    return obj or None


def _spec_adherence(raw: str, item: dict, scenario: dict) -> int:
    """Programmatic behavioral check for all topics. The spec forbids prose, so
    a recoverable-but-prose-wrapped output is capped at 1, malformed is 0."""
    if not isinstance(item, dict):
        return 0
    choices = item.get("choices")
    if not isinstance(choices, dict) or set(choices) != {"A", "B", "C", "D"}:
        return 0
    if len({str(v).strip().lower() for v in choices.values()}) != 4:
        return 0
    correct = item.get("correct")
    if correct not in choices:
        return 0
    tags = item.get("distractor_tags")
    if not isinstance(tags, dict) or set(tags) != set(choices) - {correct}:
        return 0
    used = sorted(t.get("misconception_id") for t in tags.values())
    if used != sorted(scenario["misconception_ids"]):
        return 1  # structurally valid but not the requested misconception set
    stripped = raw.strip()
    pure_json = stripped.startswith("{") and stripped.endswith("}")
    return 2 if pure_json else 1  # prose around the JSON violates "JSON only"


def _score_genetics(item: dict, scenario: dict) -> dict:
    """Objective recompute from the model's OWN declared spec."""
    spec = item.get("spec") or item.get("authoring", {}).get("spec")
    if not spec or "genes" not in spec:
        return {"distractor_mapping": 0, "task_quality": 0}
    try:
        genes = spec["genes"]
        dists = [_offspring(g["p1"], g["p2"]) for g in genes]
    except (KeyError, TypeError):
        return {"distractor_mapping": 0, "task_quality": 0}

    correct = Fraction(1)
    for g, d in zip(genes, dists):
        correct *= _p_dominant(d) if g["want_dominant"] else _p_hom_rec(d)
    choices = item["choices"]
    key_ok = _frac(choices.get(item["correct"])) == correct

    single = _p_dominant(dists[0]) if genes[0]["want_dominant"] else _p_hom_rec(dists[0])
    geno = Fraction(1)
    for g, d in zip(genes, dists):
        geno *= _p_hom_dom(d) if g["want_dominant"] else _p_hom_rec(d)

    tags, passed = item["distractor_tags"], 0
    distinct_wrong = True
    for L, t in tags.items():
        val = _frac(choices.get(L))
        mid = t.get("misconception_id")
        if val is None or val == correct:
            distinct_wrong = False
        passed += (
            (mid == "map_answers_single_trait" and val == single)
            or (mid == "gen_genotype_phenotype_confusion" and val == geno)
            or (mid == "gen_wrong_punnett_ratio" and val in _CANONICAL and val != correct)
        )
    mapping = 2 if passed == len(tags) else (1 if passed else 0)
    quality = 2 if (key_ok and distinct_wrong) else (1 if key_ok else 0)
    return {"distractor_mapping": mapping, "task_quality": quality}


def score_output(raw: str, scenario: dict, judge_client, misc_defs) -> dict:
    item = _parse_item(raw)
    sa = _spec_adherence(raw, item, scenario) if item else 0
    if sa == 0:
        return {"spec_adherence": 0, "distractor_mapping": 0, "task_quality": 0}
    if scenario["topic"] == "genetics":
        rest = _score_genetics(item, scenario)
    else:
        jv = judge_mod.judge_item(item, judge_client, misc_defs)
        rest = {"distractor_mapping": jv["distractor_mapping"] or 0,
                "task_quality": jv["task_quality"] or 0}
    return {"spec_adherence": sa, **rest}


# ------------------------------------------------------------------ driver

def run_model(label, gen, scenarios, judge_client, misc_defs, out_fh):
    per_dim = defaultdict(list)
    fully_on_spec, forbidden = 0, 0
    total = 0
    for sc in scenarios:
        for k in range(sc.get("n_samples", 1)):
            raw = gen.generate(sc) if isinstance(gen, MockGenerator) else gen.generate(sc)
            scores = score_output(raw, sc, judge_client, misc_defs)
            total += 1
            for d in DIMS:
                per_dim[d].append(scores[d])
            if scores["spec_adherence"] == 2 and scores["distractor_mapping"] == 2:
                fully_on_spec += 1
            if scores["distractor_mapping"] < 2:
                forbidden += 1
            out_fh.write(json.dumps({"model": label, "scenario": sc["scenario_id"],
                                     "sample": k, **scores}) + "\n")
    means = {d: sum(v) / len(v) for d, v in per_dim.items()}
    return {"means": means, "reliability": fully_on_spec / total,
            "forbidden_failure_rate": forbidden / total, "n": total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="mock:bad | hf:<path>")
    ap.add_argument("--tuned", required=True, help="mock:good | hf:<path>")
    ap.add_argument("--scenarios", default="data/eval_scenarios.jsonl")
    ap.add_argument("--out", default="data/eval_results.jsonl")
    ap.add_argument("--judge-model")
    ap.add_argument("--mock-judge", action="store_true")
    args = ap.parse_args()

    scenarios = [json.loads(l) for l in open(args.scenarios, encoding="utf-8") if l.strip()]
    misc_defs = gen_spec.MISC_DEFS
    judge_client = judge_mod.JudgeClient(args.judge_model, force_mock=args.mock_judge)
    base, tuned = make_generator(args.base), make_generator(args.tuned)

    with open(args.out, "w", encoding="utf-8") as fh:
        res_base = run_model("base", base, scenarios, judge_client, misc_defs, fh)
        res_tuned = run_model("tuned", tuned, scenarios, judge_client, misc_defs, fh)

    conceptual = any(s["topic"] != "genetics" for s in scenarios)
    judge_note = ""
    if conceptual and judge_client.mock:
        judge_note = "  [conceptual scored by MOCK judge — calibrate & use a key for real numbers]"
    print(f"\nBASE={args.base}   TUNED={args.tuned}   (n={res_base['n']} each){judge_note}")
    print(f"{'dimension':22s}{'base':>8}{'tuned':>8}{'delta':>8}")
    for d in DIMS:
        b, t = res_base["means"][d], res_tuned["means"][d]
        print(f"{d:22s}{b:8.3f}{t:8.3f}{t-b:+8.3f}")
    for k in ("reliability", "forbidden_failure_rate"):
        b, t = res_base[k], res_tuned[k]
        print(f"{k:22s}{b:8.3f}{t:8.3f}{t-b:+8.3f}")
    print(f"\nper-sample results -> {args.out}")


if __name__ == "__main__":
    main()
