"""
DRAFT — shared taxonomy, prompts, and parsing for the AP Bio error-TYPING layer.

This is the pivot analogue of `common.py`. It is intentionally self-contained and
illustrative: it does NOT import from or overwrite `common.py`, so the algebra
project stays intact as the working reference implementation.

Key differences from the algebra `common.py` (see docs/mcat_pivot_spec.md):
  - Taxonomy is carved by cognitive KIND (declarative / procedural / comprehension)
    instead of by algebraic operation.
  - The v1 input is an AP Bio multiple-choice ITEM + the student's chosen
    DISTRACTOR (a zero-infrastructure, model-layer BEHAVIORAL signal), instead of
    worked equation steps. Response TIMING is a deferred Phase-2+ enhancement,
    contingent on real instrumented data (see below and spec sections 2, 7, 9).
  - `abstain` maps to the product's "predict-and-confirm" deferral, not a dead end.

v1 taxonomy decision (owner-resolved): `careless_slip` is NOT in the v1 label set.
A careless slip and a genuine content gap can select the exact same distractor, so
a slip is not separable from a single distractor choice without timing or
repeated-attempt signal. `careless_slip` is therefore deferred to Phase 2 (see
DEFERRED_LABELS) and only becomes a live label once real timing / retry data
exists. We will NOT fabricate or simulate timing to surface it earlier.

Structure (LABELS, TAXONOMY_TEXT, SYSTEM_PROMPT, build_user_prompt,
build_chat_messages, build_sft_messages, format_chat, parse_label, load/write_jsonl)
mirrors `common.py` so the rest of the pipeline (prepare_sft, train_sft, calibrate,
run_baseline, model_utils.score_labels) can be re-pointed with minimal changes.
"""

from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------- taxonomy

# v1 labels: inferable from the CHOSEN DISTRACTOR alone (the SLM's strength).
LABELS = [
    "content_gap",
    "reasoning_error",
    "misread_or_passage_mapping",
    "abstain",
]

# Deferred to Phase 2+: requires real behavioral timing and/or repeated-attempt
# data to separate from content_gap on the same distractor. Not emitted in v1.
DEFERRED_LABELS = [
    "careless_slip",
]

SUBSTANTIVE_LABELS = [label for label in LABELS if label != "abstain"]

# Which score each substantive label primarily feeds (SPOV4: declarative vs
# procedural measured separately). Used by the three-score model, not by the SLM.
# careless_slip (deferred) would map to "neither" once timing makes it separable.
LABEL_TO_SCORE = {
    "content_gap": "memory",           # declarative retention
    "reasoning_error": "performance",  # procedural / application
    "misread_or_passage_mapping": "performance",
}

TAXONOMY_TEXT = """\
content_gap: the student lacks the underlying fact or concept the item tests. The chosen distractor reflects a wrong or missing fact, not a reasoning slip. (Declarative / recall failure.)
reasoning_error: the student has the facts but misapplied them - a wrong inference, a mis-integration of two concepts, or a classic application trap. (Procedural / application failure.)
misread_or_passage_mapping: the student mapped the passage or stem to the wrong quantity, missed a qualifier (EXCEPT, NOT, increases/decreases), or mis-read a figure or axis. The biology may be known; the failure is passage-to-question mapping. (Comprehension failure.)
abstain: the chosen distractor does not support a confident single label, or two error types fit equally well. Defer to the student via predict-and-confirm."""

SYSTEM_PROMPT = (
    "You are an expert AP Biology tutor classifying WHY a student picked a wrong "
    "answer on a multiple-choice item. You reason about why the chosen distractor "
    "is attractive and what kind of failure it reveals. You assign exactly one "
    "error-type label from a fixed list. You do not explain, you do not add prose, "
    "you output only the label."
)


# --------------------------------------------------------------------- prompting

def _format_choices(choices: dict) -> str:
    """Render the answer choices block, marking the correct and chosen options."""
    lines = []
    for key in sorted(choices):
        lines.append(f"  {key}. {choices[key]}")
    return "\n".join(lines)


def _behavior_line(attempt: dict | None) -> str | None:
    """Optional response-time line for the prompt (Phase 2+ only).

    v1 is distractor-only, so this returns None unless real timing is attached.
    Timing is a deferred enhancement (see module docstring / spec section 2, 7): we do
    not fabricate or simulate it, so no placeholder line is emitted when absent.
    """
    if not attempt:
        return None
    rt = attempt.get("response_time_ms")
    exp = attempt.get("expected_time_ms")
    if rt is None:
        return None
    if exp:
        ratio = rt / exp
        speed = "fast" if ratio < 0.4 else ("slow" if ratio > 1.5 else "typical")
        return f"Response time: {rt} ms (expected ~{exp} ms, {speed} for this item)."
    return f"Response time: {rt} ms."


def build_user_prompt(item: dict, attempt: dict | None = None) -> str:
    """Build the classification prompt from an AP Bio item + chosen distractor.

    `item` follows the schema in data/apbio_item_template.jsonl (passage, stem,
    choices, correct, distractor_tags). `attempt` is the optional attempt record
    (chosen, response_time_ms, expected_time_ms). If `attempt` is None, `item`
    may itself carry a `chosen` key (convenient for eval rows).
    """
    chosen = (attempt or item).get("chosen")
    passage = item.get("passage")
    passage_block = f"Passage:\n{passage}\n\n" if passage else ""
    choices_block = _format_choices(item.get("choices", {}))
    correct = item.get("correct")

    behavior = _behavior_line(attempt)
    behavior_block = f"\n{behavior}" if behavior else ""

    return f"""Here is the fixed list of error-type labels and what each means:

{TAXONOMY_TEXT}

Classify WHY the student chose their (incorrect) answer into exactly ONE label.
Rules:
- Output only the label string, lowercase, exactly as written in the list.
- Do not output any explanation, punctuation, or extra text.
- If the chosen distractor does not support a confident single label, output: abstain

{passage_block}Question: {item.get('stem', '')}
Choices:
{choices_block}
Correct answer: {correct}
Student's chosen answer: {chosen}{behavior_block}

Label:"""


def build_chat_messages(item: dict, attempt: dict | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item, attempt)},
    ]


def gold_label(item: dict, attempt: dict | None = None) -> str | None:
    """Resolve the gold error-type for a chosen distractor from the item's tags."""
    chosen = (attempt or item).get("chosen")
    tags = item.get("distractor_tags", {})
    if chosen in tags:
        return tags[chosen].get("error_type")
    return item.get("label")  # fall back to an explicit label if provided


def build_sft_messages(item: dict, attempt: dict | None = None) -> list[dict]:
    """Chat messages for SFT; assistant reply is the gold error-type label."""
    messages = build_chat_messages(item, attempt)
    label = gold_label(item, attempt)
    messages.append({"role": "assistant", "content": label})
    return messages


def format_chat(
    tokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Apply chat template; disable Qwen3 thinking mode when supported.

    Identical to common.format_chat — kept here so this module is self-contained.
    """
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# --------------------------------------------------------------------- parsing

def parse_label(raw: str) -> tuple[str | None, bool]:
    """Return (label, is_schema_valid). Identical contract to common.parse_label."""
    text = raw.strip().lower()

    if text in LABELS:
        return text, True

    found = None
    earliest = len(text) + 1
    for label in LABELS:
        idx = text.find(label)
        if idx != -1 and idx < earliest:
            earliest = idx
            found = label
    if found is not None:
        return found, False

    return None, False


# ------------------------------------------ timing gate (DEFERRED, Phase 2+ only)

def apply_timing_gate(
    probs: dict[str, float],
    attempt: dict | None,
    *,
    fast_ratio: float = 0.4,
    slip_boost: float = 0.25,
) -> dict[str, float]:
    """DEFERRED slip-vs-effort re-weighting. NOT part of the v1 engine.

    This helper is a Phase-2+ sketch (mcat_pivot_spec.md sections 2, 7). It is
    inert in v1 for two reasons:
      1. `careless_slip` is not in the v1 LABELS (it is in DEFERRED_LABELS), so the
         re-weight below is a no-op unless a caller has explicitly added the
         deferred label to `probs`.
      2. It requires REAL response-time data (`response_time_ms` / `expected_time_ms`)
         from an instrumented UI that does not exist yet. We will NOT fabricate or
         simulate timing to activate this path, because reporting simulated-timing
         results would undermine the project's honest-measurement thesis (see spec
         section 9). Thresholds here are placeholders to be fit on real data.

    Returns a renormalized copy; does nothing if timing is unavailable.
    """
    if not attempt:
        return dict(probs)
    rt = attempt.get("response_time_ms")
    exp = attempt.get("expected_time_ms")
    if not rt or not exp:
        return dict(probs)

    out = dict(probs)
    ratio = rt / exp
    if ratio < fast_ratio and "careless_slip" in out:
        out["careless_slip"] = out["careless_slip"] + slip_boost
    elif ratio > 1.5 and "careless_slip" in out:
        out["careless_slip"] = max(out["careless_slip"] - slip_boost, 0.0)

    total = sum(out.values())
    if total <= 0:
        return dict(probs)
    return {label: value / total for label, value in out.items()}


def apply_predict_and_confirm(
    probs: dict[str, float],
    *,
    confirm_threshold: float | None = None,
    tie_margin: float = 0.1,
) -> dict:
    """DRAFT decision layer: commit to a label or defer to the student.

    Mirrors model_utils.apply_abstention, but instead of collapsing to "abstain"
    it returns a structured action for the predict-and-confirm UI (spec section 4.2 / 7).
      - "commit": confident single label.
      - "confirm": low confidence OR top-2 tie -> surface top-2, ask the student.
    """
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_conf = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else (None, 0.0)

    low_conf = confirm_threshold is not None and top_conf < confirm_threshold
    tie = (top_conf - runner[1]) < tie_margin

    if low_conf or tie:
        return {
            "action": "confirm",
            "top2": [top_label, runner[0]],
            "confidence": top_conf,
        }
    return {"action": "commit", "label": top_label, "confidence": top_conf}


# --------------------------------------------------------------------- io

def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------- smoke test

if __name__ == "__main__":
    # Illustrative self-check that the prompt builder and parser run end to end.
    demo_item = {
        "id": "apbio_cellresp_0001",
        "stem": "In the electron transport chain, what is the direct role of oxygen?",
        "choices": {
            "A": "Final electron acceptor",
            "B": "Donates electrons to NADH",
            "C": "Phosphorylates ADP directly",
            "D": "Catalyzes the citric acid cycle",
        },
        "correct": "A",
        "distractor_tags": {
            "B": {"error_type": "content_gap"},
            "C": {"error_type": "reasoning_error"},
            "D": {"error_type": "content_gap"},
        },
    }
    # v1: distractor-only, no timing attached.
    demo_attempt = {"chosen": "C"}

    print(build_user_prompt(demo_item, demo_attempt))
    print("\nv1 LABELS:", LABELS)
    print("DEFERRED_LABELS (Phase 2+):", DEFERRED_LABELS)
    print("Gold label:", gold_label(demo_item, demo_attempt))
    print("parse_label('reasoning_error'):", parse_label("reasoning_error"))

    # v1 error-type distribution over the four labels (no careless_slip).
    demo_probs = {
        "content_gap": 0.28,
        "reasoning_error": 0.55,
        "misread_or_passage_mapping": 0.12,
        "abstain": 0.05,
    }
    print("Decision:", apply_predict_and_confirm(demo_probs, confirm_threshold=0.5))

    # Phase-2+ helper is inert in v1 (no timing, careless_slip not present).
    gated = apply_timing_gate(demo_probs, demo_attempt)
    print("Timing gate (deferred, no-op in v1):",
          {k: round(v, 3) for k, v in gated.items()})
