"""
Loader + taxonomy mapping for REAL student-error data (Eedi, PSLC DataShop, etc.).

Real corpora describe misconceptions in free text and use varied field names. This
module normalizes an arbitrary real row into the same schema the classifier expects
(problem / correct_answer / student_answer / student_work / label / id) and provides a
best-effort keyword mapping from a free-text misconception onto the 7-label taxonomy.

The mapping is a *starting point* for hand-labeling, not ground truth: real data should
be reviewed by a human. `map_misconception_to_label` returns (label, confident) where
`confident` is False when the text matched nothing and the row was defaulted.

See docs/real_data.md for sourcing and licensing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LABELS, SUBSTANTIVE_LABELS, load_jsonl, write_jsonl

# Ordered: earlier keywords win. Phrases are matched case-insensitively as substrings
# against the free-text misconception / bug description.
MISCONCEPTION_KEYWORDS: list[tuple[str, str]] = [
    ("distribut", "distribution_property_error"),
    ("expand", "distribution_property_error"),
    ("order of operation", "distribution_property_error"),
    ("bracket", "distribution_property_error"),
    ("parenthes", "distribution_property_error"),
    ("both sides", "equality_balance_error"),
    ("balance", "equality_balance_error"),
    ("one side", "equality_balance_error"),
    ("inverse operation", "operation_inverse_error"),
    ("multiply instead of divid", "operation_inverse_error"),
    ("divide instead of multipl", "operation_inverse_error"),
    ("wrong operation", "operation_inverse_error"),
    ("negative sign", "negative_sign_error"),
    ("sign error", "negative_sign_error"),
    ("minus sign", "negative_sign_error"),
    ("flip the sign", "negative_sign_error"),
    ("did not flip", "negative_sign_error"),
    ("combine unlike", "variable_error"),
    ("like terms", "variable_error"),
    ("combined constant", "variable_error"),
    ("coefficient with constant", "variable_error"),
    ("conjoin", "variable_error"),
    ("computation", "arithmetic_slip"),
    ("arithmetic", "arithmetic_slip"),
    ("calculation error", "arithmetic_slip"),
    ("adds incorrectly", "arithmetic_slip"),
    ("multiplies incorrectly", "arithmetic_slip"),
]

# Common alternative field names in real dumps -> our canonical keys.
FIELD_ALIASES = {
    "problem": ["problem", "question", "question_text", "prompt", "stem", "equation"],
    "correct_answer": ["correct_answer", "correct", "answer", "correct_option", "solution"],
    "student_answer": ["student_answer", "distractor", "chosen", "response", "selected_answer"],
    "student_work": ["student_work", "work", "steps", "working", "rationale", "reasoning"],
    "label": ["label", "gold", "gold_label", "taxonomy_label"],
    "misconception": ["misconception", "misconception_name", "bug", "bug_message", "diagnosis"],
    "id": ["id", "question_id", "row_id", "uid"],
}


def _first_present(row: dict, keys: list[str]):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def map_misconception_to_label(text: str | None) -> tuple[str, bool]:
    """Map free-text misconception onto the taxonomy. Returns (label, confident)."""
    if not text:
        return "abstain", False
    low = str(text).strip().lower()
    if low in LABELS:
        return low, True
    for needle, label in MISCONCEPTION_KEYWORDS:
        if needle in low:
            return label, True
    return "abstain", False


def normalize_real_example(row: dict, idx: int = 0) -> dict:
    """Normalize an arbitrary real-data row into the classifier's schema."""
    problem = _first_present(row, FIELD_ALIASES["problem"])
    if problem is None:
        raise ValueError(f"row {idx}: no recognizable problem/question field: {list(row)}")

    correct = _first_present(row, FIELD_ALIASES["correct_answer"])
    student = _first_present(row, FIELD_ALIASES["student_answer"])
    work = _first_present(row, FIELD_ALIASES["student_work"])
    rid = _first_present(row, FIELD_ALIASES["id"]) or f"real{idx:05d}"

    # Prefer an explicit taxonomy label; otherwise map the free-text misconception.
    label = _first_present(row, FIELD_ALIASES["label"])
    mapped_confident = True
    if label is None:
        label, mapped_confident = map_misconception_to_label(
            _first_present(row, FIELD_ALIASES["misconception"])
        )

    if label not in LABELS:
        raise ValueError(f"row {idx}: label '{label}' not in taxonomy {LABELS}")

    out = {
        "id": str(rid),
        "problem": str(problem),
        "correct_answer": str(correct) if correct is not None else "",
        "student_answer": str(student) if student is not None else "",
        "student_work": str(work) if work not in (None, "") else None,
        "label": label,
    }
    if not mapped_confident:
        out["needs_review"] = True
        misc = _first_present(row, FIELD_ALIASES["misconception"])
        if misc:
            out["raw_misconception"] = str(misc)
    return out


def load_real_eval(path: str | Path) -> list[dict]:
    """Load a real-data JSONL and normalize every row to the classifier schema."""
    raw = load_jsonl(path)
    return [normalize_real_example(row, idx) for idx, row in enumerate(raw)]


def main():
    """CLI: normalize a raw real-data dump and report mapping coverage."""
    parser = argparse.ArgumentParser(description="Normalize real student-error data.")
    parser.add_argument("--in", dest="inp", required=True, help="raw real-data JSONL")
    parser.add_argument("--out", required=True, help="normalized JSONL in classifier schema")
    args = parser.parse_args()

    rows = load_real_eval(args.inp)
    write_jsonl(args.out, rows)

    from collections import Counter

    needs = sum(1 for r in rows if r.get("needs_review"))
    print(f"Normalized {len(rows)} rows -> {args.out}")
    print(f"Rows needing human review (unmapped misconception): {needs}")
    print("Label distribution:", dict(Counter(r["label"] for r in rows)))
    if needs:
        print(
            "\nReview rows with \"needs_review\": true and set a correct taxonomy label "
            "(or keep abstain) before using as an eval set."
        )


if __name__ == "__main__":
    main()
