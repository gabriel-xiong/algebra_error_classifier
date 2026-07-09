"""Shared taxonomy, prompts, and parsing for the algebra error classifier."""

from __future__ import annotations

import json
from pathlib import Path

LABELS = [
    "equality_balance_error",
    "negative_sign_error",
    "variable_error",
    "operation_inverse_error",
    "distribution_property_error",
    "arithmetic_slip",
    "abstain",
]

SUBSTANTIVE_LABELS = [label for label in LABELS if label != "abstain"]

TAXONOMY_TEXT = """\
equality_balance_error: operated on one side of the equation only, dropped the equals sign, or otherwise did not keep both sides balanced.
negative_sign_error: dropped or mishandled a negative sign, or moved a term across the equals sign without flipping its sign.
variable_error: combined unlike terms, conjoined a constant and a variable term (e.g. wrote 2 + 3x as 5x), or combined variable terms with the wrong sign.
operation_inverse_error: used the wrong inverse operation (e.g. divided when they should have multiplied).
distribution_property_error: misapplied the distributive property or order of operations (e.g. distributed to only one term inside parentheses).
arithmetic_slip: used the correct procedure and correct operations, but made a pure computation mistake (e.g. 6 + 8 = 13).
abstain: the visible information does not support a confident single label. Use this when no work is shown and the answer is consistent with more than one error type, or when the shown work fits two distinct error types equally well."""

SYSTEM_PROMPT = (
    "You are an expert algebra teacher classifying why a student got a linear-equation "
    "problem wrong. You assign exactly one error-type label from a fixed list. You do "
    "not explain, you do not add prose, you output only the label."
)


def build_user_prompt(example: dict) -> str:
    work = example["student_work"] if example.get("student_work") else (
        "(no work shown, only the final answer)"
    )
    return f"""Here is the fixed list of error-type labels and what each means:

{TAXONOMY_TEXT}

Classify the student's error into exactly ONE label from the list above.
Rules:
- Output only the label string, lowercase, exactly as written in the list.
- Do not output any explanation, punctuation, or extra text.
- If the visible information does not support a confident single label, output: abstain

Problem: {example['problem']}
Correct answer: {example['correct_answer']}
Student's final answer: {example['student_answer']}
Student's shown work: {work}

Label:"""


def build_chat_messages(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
    ]


def build_sft_messages(example: dict) -> list[dict]:
    """Chat messages for supervised fine-tuning (assistant reply is the gold label)."""
    messages = build_chat_messages(example)
    messages.append({"role": "assistant", "content": example["label"]})
    return messages


def format_chat(
    tokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Apply chat template; disable Qwen3 thinking mode when supported."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_label(raw: str) -> tuple[str | None, bool]:
    """Return (label, is_schema_valid)."""
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


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
