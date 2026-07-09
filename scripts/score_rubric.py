"""
Generation-quality RUBRIC scorer — the heart of the eval.

The project deliverable is a GENERATION model. The eval prompts both a baseline
model and our fine-tuned model to generate AP Bio items from held-out specs, then
scores each generated item on this rubric and compares aggregates. This module is
the scorer; scripts/eval_generation.py drives baseline-vs-fine-tuned.

Rubric dimensions (each 0..1; `overall` is their mean over applicable dims):

  well_formed            structural: exactly 4 choices, a valid correct key,
                         every wrong option tagged with a misconception (GATE:
                         if 0, the item scores 0 overall — an ill-formed item is
                         unusable regardless of content).
  choice_differentiation the 4 options are mutually distinct (no duplicate/again
                         paraphrased options). Programmatic here; a semantic
                         LLM check is available via --judge for near-duplicates.
  answer_correctness     the keyed answer is actually correct.
                           - genetics: RECOMPUTED from the item's own declared
                             cross spec (objective, no judge).
                           - conceptual: LLM judge, else `null` (needs_judge).
  distractor_mapping     each distractor actually expresses its claimed
                         misconception ("oh, that's the ___ error").
                           - genetics: distractor value must equal the output of
                             the named error operator recomputed from the spec
                             (objective). wrong-ratio must be a canonical ratio.
                           - conceptual: LLM judge, else null.
  single_error           each distractor is genuinely wrong (!= correct) and
                         carries exactly one named misconception (no duplicates).

`answer_correctness` / `distractor_mapping` are the two dimensions where a
fine-tuned model is expected to beat baseline: baselines drift and mis-map tags.
For genetics those two are OBJECTIVE, giving the comparison a hard ground truth.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from gen_genetics import (
    GeneSpec, CANONICAL_FRACTIONS,
    solve_correct, _locus_pheno_prob, _locus_geno_prob,
)


def _frac(s: str) -> Fraction | None:
    try:
        return Fraction(str(s).strip())
    except (ValueError, ZeroDivisionError):
        return None


def _genes_from_spec(spec: dict) -> list[GeneSpec]:
    """Reconstruct minimal GeneSpecs (values only) from the emitted spec."""
    return [
        GeneSpec(letter=g["letter"], trait="", dom="", rec="",
                 p1=g["p1"], p2=g["p2"], want_dominant=g["want_dominant"])
        for g in spec["genes"]
    ]


# ------------------------------------------------------------ structural checks

def score_well_formed(item: dict) -> float:
    choices = item.get("choices", {})
    if len(choices) != 4:
        return 0.0
    if item.get("correct") not in choices:
        return 0.0
    tags = item.get("distractor_tags", {})
    if set(tags) != set(choices) - {item["correct"]}:
        return 0.0
    for t in tags.values():
        if not (t.get("misconception_id") and t.get("error_type")):
            return 0.0
    return 1.0


def score_choice_differentiation(item: dict) -> float:
    vals = [str(v).strip().lower() for v in item.get("choices", {}).values()]
    return 1.0 if len(set(vals)) == len(vals) and vals else 0.0


def score_single_error(item: dict) -> float:
    choices = item.get("choices", {})
    correct_val = str(choices.get(item.get("correct"), "")).strip().lower()
    tags = item.get("distractor_tags", {})
    ids = [t.get("misconception_id") for t in tags.values()]
    if len(set(ids)) != len(ids):  # duplicate misconception in one item
        return 0.0
    wrong_ok = sum(
        1 for L, t in tags.items()
        if str(choices.get(L, "")).strip().lower() != correct_val
    )
    return wrong_ok / len(tags) if tags else 0.0


# ------------------------------------------------- objective genetics scoring

def _genetics_scores(item: dict) -> dict:
    spec = item.get("authoring", {}).get("spec")
    choices = item["choices"]
    if not spec:
        return {"answer_correctness": None, "distractor_mapping": None}
    genes = _genes_from_spec(spec)
    correct = solve_correct(genes)

    key_val = _frac(choices.get(item["correct"]))
    answer_correctness = 1.0 if key_val is not None and key_val == correct else 0.0

    single_trait_val = _locus_pheno_prob(genes[0])
    geno_conf_val = Fraction(1)
    for g in genes:
        geno_conf_val *= _locus_geno_prob(g)

    passed = 0
    tags = item["distractor_tags"]
    for letter, tag in tags.items():
        val = _frac(choices.get(letter))
        mid = tag.get("misconception_id")
        if val is None:
            continue
        if mid == "map_answers_single_trait":
            passed += (val == single_trait_val)
        elif mid == "gen_genotype_phenotype_confusion":
            passed += (val == geno_conf_val)
        elif mid == "gen_wrong_punnett_ratio":
            passed += (val in CANONICAL_FRACTIONS and val != correct)
        # unknown operator for genetics -> not objectively verifiable -> fail
    distractor_mapping = passed / len(tags) if tags else 0.0
    return {"answer_correctness": answer_correctness,
            "distractor_mapping": distractor_mapping}


# ------------------------------------------------------- LLM-judge (conceptual)

def build_mapping_judge_prompt(item: dict, letter: str, misconception_desc: str) -> str:
    """Prompt for an LLM judge to score conceptual distractor mapping.

    Used by eval_generation.py when a judge callable is supplied; kept here so the
    rubric definition and its judge prompts live in one place.
    """
    return (
        "You are grading one wrong answer choice on a biology multiple-choice item.\n"
        f"Question: {item.get('stem')}\n"
        f"Choice: {item['choices'].get(letter)}\n"
        f"Claimed misconception: {misconception_desc}\n\n"
        "Answer strict JSON: {\"maps\": true|false, \"is_wrong\": true|false}\n"
        "- maps: does the choice specifically express THAT misconception "
        "(not merely 'a wrong statement')?\n"
        "- is_wrong: is the choice actually incorrect for this question?"
    )


# ------------------------------------------------------------------- aggregate

def score_item(item: dict, judge=None, misconceptions: dict | None = None) -> dict:
    wf = score_well_formed(item)
    if wf == 0.0:
        return {"well_formed": 0.0, "overall": 0.0,
                "choice_differentiation": 0.0, "answer_correctness": 0.0,
                "distractor_mapping": 0.0, "single_error": 0.0}

    scores = {
        "well_formed": 1.0,
        "choice_differentiation": score_choice_differentiation(item),
        "single_error": score_single_error(item),
    }

    if item.get("topic") == "genetics":
        scores.update(_genetics_scores(item))
    else:
        # Conceptual: objective recompute impossible. Use judge if provided.
        if judge is not None:
            m = _judge_conceptual(item, judge, misconceptions or {})
            scores.update(m)
        else:
            scores["answer_correctness"] = None
            scores["distractor_mapping"] = None

    applicable = [v for k, v in scores.items() if v is not None]
    scores["overall"] = sum(applicable) / len(applicable) if applicable else 0.0
    return scores


def _judge_conceptual(item: dict, judge, misconceptions: dict) -> dict:
    maps_ok, wrong_ok, n = 0, 0, 0
    for letter, tag in item["distractor_tags"].items():
        mid = tag.get("misconception_id", "")
        desc = misconceptions.get(mid, {}).get("description", mid)
        raw = judge(build_mapping_judge_prompt(item, letter, desc))
        try:
            verdict = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            verdict = {}
        maps_ok += bool(verdict.get("maps"))
        wrong_ok += bool(verdict.get("is_wrong"))
        n += 1
    return {"distractor_mapping": maps_ok / n if n else 0.0,
            "answer_correctness": wrong_ok / n if n else 0.0}


def score_file(path: str, judge=None) -> dict:
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    dims = ["well_formed", "choice_differentiation", "answer_correctness",
            "distractor_mapping", "single_error", "overall"]
    sums = {d: 0.0 for d in dims}
    counts = {d: 0 for d in dims}
    for it in items:
        s = score_item(it, judge=judge)
        for d in dims:
            if s.get(d) is not None:
                sums[d] += s[d]
                counts[d] += 1
    return {d: (sums[d] / counts[d] if counts[d] else None) for d in dims} | \
           {"_n": len(items)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in args.files:
        res = score_file(f)
        n = res.pop("_n")
        print(f"\n{f}  (n={n})")
        for k, v in res.items():
            print(f"  {k:24s} {v if v is None else round(v, 4)}")


if __name__ == "__main__":
    main()
