"""
The GENERATION SPEC: the shared contract between training and eval.

A *scenario* (model input) is a topic + a set of target misconceptions. The
model's job (see docs/behavior_spec.md) is to return one valid JSON MCQ that
embeds each requested misconception as a distractor. This module is the single
source of truth for:

  - build_generation_prompt(topic, misconception_ids) -> (system, user)
        the prompt given to BOTH the base and the fine-tuned model at eval, and
        the user turn of every SFT example.
  - item_to_target(item)          canonical JSON the model should emit.
  - item_to_sft_messages(item)    [system, user(spec), assistant(target)] for SFT.
  - item_to_scenario(item) / combo_key(item)   derive the input spec / a hashable
        (topic, frame, misconception-set) key used to split train vs held-out.

Keeping this in one place guarantees the fine-tune is trained on exactly the
prompt format it is later evaluated on.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_MISC_PATH = REPO / "data" / "apbio_misconceptions.json"


def load_misconception_defs() -> dict[str, dict]:
    data = json.loads(_MISC_PATH.read_text(encoding="utf-8"))
    return {m["id"]: m for m in data["misconceptions"]}


MISC_DEFS = load_misconception_defs()

SYSTEM_PROMPT = (
    "You are an AP Biology item writer. You generate multiple-choice questions in "
    "which every wrong answer is a deliberate, named misconception. You output "
    "ONLY a single JSON object and nothing else: no prose, no explanation, no "
    "markdown code fences, before or after the JSON."
)


def _misconception_lines(misconception_ids: list[str]) -> str:
    lines = []
    for mid in misconception_ids:
        d = MISC_DEFS.get(mid, {})
        name = d.get("name", mid)
        desc = d.get("description", "")
        lines.append(f'  - {mid} ("{name}"): {desc}')
    return "\n".join(lines)


def build_generation_prompt(topic: str, misconception_ids: list[str]) -> tuple[str, str]:
    """The exact prompt used at train time and eval time."""
    genetics_note = ""
    if topic == "genetics":
        genetics_note = (
            '\n- Because this is genetics, also include a "spec" field: '
            '{"genes":[{"letter","p1","p2","want_dominant"}]} where p1/p2 are one '
            'of "hom_dom"|"het"|"hom_rec", and ensure the correct answer is the '
            "mathematically correct fraction for that cross."
        )
    user = f"""Write one AP Biology multiple-choice item on the topic: {topic}.

Embed EXACTLY these misconceptions, one per wrong answer choice. Each wrong
choice must be a statement/answer a student holding that specific misconception
would pick:

{_misconception_lines(misconception_ids)}

Output a single JSON object with these fields:
- "stem": the question text.
- "choices": an object with keys "A","B","C","D" (four distinct options).
- "correct": the letter of the single correct option.
- "distractor_tags": an object mapping each WRONG option's letter to
  {{"misconception_id": <one of the ids above>}}. Every wrong option must appear,
  and the three misconceptions must be used exactly once each.{genetics_note}

Output only the JSON object."""
    return SYSTEM_PROMPT, user


def item_to_target(item: dict) -> dict:
    """Canonical output object the model should learn to emit for an item."""
    target = {
        "stem": item["stem"],
        "choices": item["choices"],
        "correct": item["correct"],
        "distractor_tags": {
            L: {"misconception_id": t["misconception_id"]}
            for L, t in item["distractor_tags"].items()
        },
    }
    spec = item.get("authoring", {}).get("spec")
    if item.get("topic") == "genetics" and spec:
        target["spec"] = spec
    return target


def misconception_ids_of(item: dict) -> list[str]:
    return sorted(t["misconception_id"] for t in item["distractor_tags"].values())


def item_to_scenario(item: dict) -> dict:
    return {"topic": item["topic"], "misconception_ids": misconception_ids_of(item)}


def item_to_sft_messages(item: dict) -> list[dict]:
    system, user = build_generation_prompt(item["topic"], misconception_ids_of(item))
    target = json.dumps(item_to_target(item), separators=(",", ":"))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]


def combo_key(item: dict) -> tuple:
    """Hashable identity for train/held-out splitting: (topic, frame, misc-set).

    Genetics has no frame (fixed misconception triple), so its combo is a single
    bucket — held out by reliability sampling, not by combination (see scenarios).
    """
    frame = item.get("authoring", {}).get("frame", "_genetics")
    return (item["topic"], frame, frozenset(misconception_ids_of(item)))


if __name__ == "__main__":
    # Smoke: show a scenario prompt + the SFT target for one genetics item.
    demo = json.loads((REPO / "data" / "gen_genetics.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    sys_p, user_p = build_generation_prompt(demo["topic"], misconception_ids_of(demo))
    print(user_p)
    print("\n--- assistant target ---")
    print(json.dumps(item_to_target(demo), indent=2))
