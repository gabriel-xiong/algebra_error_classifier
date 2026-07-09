"""
Shared frame-based generator engine for CONCEPTUAL (role-swap) AP Bio topics.

A conceptual topic supplies FRAMES (see gen_cellresp.py / gen_enzymes.py): each
frame is a question whose correct answer is a true statement, plus a pool of
>=3 misconception-tagged false statements that all plausibly answer that stem.
This engine assembles/verifies/dedups items from any such frame set, so each
topic module only authors content, not plumbing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def sig(stem: str, choices: dict) -> str:
    return stem + "||" + "|".join(sorted(choices.values()))


def generate_item(rng: random.Random, idx: int, *, frames, topic, error_type,
                  id_prefix: str, generator: str, exclude=None) -> dict | None:
    """Return one item, or None if the sampled (frame, misconception-set) is in
    `exclude` (a set of (frame_id, frozenset(misconception_ids))) — used to keep
    held-out eval combos out of the training data."""
    frame = rng.choice(frames)
    stem = rng.choice(frame["stems"])
    correct_text = rng.choice(frame["correct"])

    mids = rng.sample(list(frame["distractors"]), 3)
    if exclude and (frame["id"], frozenset(mids)) in exclude:
        return None
    distractors = [(mid, rng.choice(frame["distractors"][mid])) for mid in mids]

    options = [("__correct__", correct_text)] + list(distractors)
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    choices, tags, correct_letter = {}, {}, None
    for letter, (mid, text) in zip(letters, options):
        choices[letter] = text
        if mid == "__correct__":
            correct_letter = letter
        else:
            tags[letter] = {
                "error_type": error_type[mid],
                "misconception_id": mid,
                "rationale": f"This choice expresses the misconception '{mid}': "
                             f"a plausible but incorrect statement a student "
                             f"holding that belief would select.",
            }

    return {
        "id": f"{id_prefix}_{idx:04d}",
        "topic": topic,
        "subtopic": frame["subtopic"],
        "knowledge_type": "declarative",
        "difficulty": "medium",
        "passage": None,
        "stem": stem,
        "choices": choices,
        "correct": correct_letter,
        "distractor_tags": tags,
        "authoring": {"source": "by_construction", "generator": generator,
                      "frame": frame["id"]},
    }


def verify_item(item: dict) -> tuple[bool, str]:
    """Structural integrity (conceptual items can't be recomputed like genetics)."""
    choices = item["choices"]
    if len(choices) != 4:
        return False, "expected 4 choices"
    if len({v.strip().lower() for v in choices.values()}) != 4:
        return False, "choices not distinct"
    if item["correct"] not in choices:
        return False, "correct letter not among choices"
    if set(item["distractor_tags"]) != set(choices) - {item["correct"]}:
        return False, "every wrong option must be tagged"
    ids = [t["misconception_id"] for t in item["distractor_tags"].values()]
    if len(set(ids)) != len(ids):
        return False, "duplicate misconception in one item"
    return True, "ok"


def run(*, frames, topic, error_type, id_prefix, generator, default_out) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=500)
    ap.add_argument("-o", "--out", default=default_out)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kw = dict(frames=frames, topic=topic, error_type=error_type,
              id_prefix=id_prefix, generator=generator)

    if args.selftest:
        items = [generate_item(rng, i, **kw) for i in range(20)]
        ok = sum(verify_item(it)[0] for it in items)
        print(f"[{topic}] generated 20, verified {ok}/20")
        print(json.dumps(items[0], indent=2))
        return

    rows, seen, tries = [], set(), 0
    while len(rows) < args.count and tries < args.count * 60:
        tries += 1
        it = generate_item(rng, len(rows), **kw)
        if it is None:
            continue
        s = sig(it["stem"], it["choices"])
        if s in seen or not verify_item(it)[0]:
            continue
        seen.add(s)
        rows.append(it)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[{topic}] wrote {len(rows)} items -> {out} (requested {args.count})")
