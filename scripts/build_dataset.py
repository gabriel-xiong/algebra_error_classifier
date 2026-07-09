"""
Canonical corpus builder — produces the training set AND a held-out eval
scenario set that is DISJOINT from training by construction.

Train/held-out split strategy (see docs/behavior_spec.md):

  Conceptual (cellular_respiration, enzymes): the input scenario is
  (topic, misconception-triple). For every frame that offers >=4 competing
  misconceptions we RESERVE one 3-subset as a held-out combination and exclude
  it from training. The tuned model is then evaluated on misconception
  combinations it never saw during SFT (in-distribution generalization).

  Genetics: every genetics item uses the SAME fixed misconception triple, so it
  cannot be held out by combination. Instead genetics is a RELIABILITY scenario:
  the fixed spec is run many times at eval and each generation is checked
  (valid + answer recomputed-correct + not a verbatim copy of any training item).

Outputs:
  data/gen_train.jsonl      the balanced 50/50 corpus (items)
  data/gen_sft.jsonl        SFT chat messages (prompt->target) for training
  data/eval_scenarios.jsonl held-out scenarios for base-vs-tuned eval
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

import conceptual_engine as engine
import gen_cellresp
import gen_enzymes
import gen_genetics
import gen_spec

REPO = Path(__file__).resolve().parent.parent
SEED = 0

CONCEPTUAL = [
    (gen_cellresp.TOPIC, gen_cellresp.FRAMES, gen_cellresp.ERROR_TYPE, "gen_cellresp"),
    (gen_enzymes.TOPIC, gen_enzymes.FRAMES, gen_enzymes.ERROR_TYPE, "gen_enzymes"),
]


def reserved_heldout(frames) -> set:
    """One deterministic 3-subset per frame with >=4 misconceptions."""
    out = set()
    for fr in frames:
        mids = sorted(fr["distractors"])
        if len(mids) >= 4:
            out.add((fr["id"], frozenset(sorted(mids)[:3])))
    return out


def build_conceptual(rng: random.Random):
    rows, exclude_by_topic = [], {}
    for topic, frames, err, prefix in CONCEPTUAL:
        exclude = reserved_heldout(frames)
        exclude_by_topic[topic] = (frames, exclude)
        seen, tries, made = set(), 0, 0
        # generous cap; conceptual space is bounded
        while tries < 200_000 and made < 5000:
            tries += 1
            it = engine.generate_item(
                rng, len(rows), frames=frames, topic=topic, error_type=err,
                id_prefix=prefix, generator=f"scripts/{prefix}.py", exclude=exclude)
            if it is None:
                continue
            s = engine.sig(it["stem"], it["choices"])
            if s in seen or not engine.verify_item(it)[0]:
                continue
            seen.add(s)
            rows.append(it)
            made += 1
            if tries - made > 3000 and made > 0:  # ran dry -> stop this topic
                break
    return rows, exclude_by_topic


def build_genetics(rng: random.Random, count: int):
    rows, seen, tries = [], set(), 0
    while len(rows) < count and tries < count * 40:
        tries += 1
        it = gen_genetics.generate_item(rng, len(rows))
        if not it or it["stem"] in seen:
            continue
        seen.add(it["stem"])
        rows.append(it)
    return rows


def build_eval_scenarios(exclude_by_topic: dict, genetics_triple: list[str]) -> list[dict]:
    scenarios, sid = [], 0
    for topic, (frames, exclude) in exclude_by_topic.items():
        for frame_id, mid_set in sorted(exclude, key=lambda x: (x[0], sorted(x[1]))):
            scenarios.append({
                "scenario_id": f"eval_{sid:03d}",
                "topic": topic,
                "frame": frame_id,
                "misconception_ids": sorted(mid_set),
                "mode": "heldout_combo",
                "n_samples": 3,
            })
            sid += 1
    scenarios.append({
        "scenario_id": f"eval_{sid:03d}",
        "topic": "genetics",
        "misconception_ids": genetics_triple,
        "mode": "reliability",
        "n_samples": 40,
    })
    return scenarios


def main() -> None:
    rng = random.Random(SEED)
    conceptual = build_conceptual(rng)
    concept_rows, exclude_by_topic = conceptual
    genetics_rows = build_genetics(rng, count=len(concept_rows))  # 50/50

    all_rows = concept_rows + genetics_rows
    rng.shuffle(all_rows)
    for i, r in enumerate(all_rows):
        r["id"] = f"gen_{i:05d}"

    data = REPO / "data"
    with open(data / "gen_train.jsonl", "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    sft = [{"messages": gen_spec.item_to_sft_messages(r)} for r in all_rows]
    with open(data / "gen_sft.jsonl", "w", encoding="utf-8") as fh:
        for r in sft:
            fh.write(json.dumps(r) + "\n")
    # 95/5 train/val split for training-loss monitoring (the REAL generalization
    # test is eval_scenarios, not this in-distribution val).
    n_val = max(1, len(sft) // 20)
    val, train = sft[:n_val], sft[n_val:]
    for name, part in (("gen_sft_train.jsonl", train), ("gen_sft_val.jsonl", val)):
        with open(data / name, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")

    genetics_triple = gen_spec.misconception_ids_of(genetics_rows[0])
    scenarios = build_eval_scenarios(exclude_by_topic, genetics_triple)
    with open(data / "eval_scenarios.jsonl", "w", encoding="utf-8") as fh:
        for s in scenarios:
            fh.write(json.dumps(s) + "\n")

    # Guarantee: no held-out combo leaked into training.
    train_combos = {gen_spec.combo_key(r) for r in all_rows}
    leaks = [s for s in scenarios if s["mode"] == "heldout_combo"
             and (s["topic"], s["frame"], frozenset(s["misconception_ids"])) in train_combos]

    import collections
    topic_counts = collections.Counter(r["topic"] for r in all_rows)
    print(f"train items: {len(all_rows)}  by topic: {dict(topic_counts)}")
    print(f"  conceptual={len(concept_rows)}  genetics={len(genetics_rows)} "
          f"(50/50 procedural vs conceptual)")
    print(f"eval scenarios: {len(scenarios)} "
          f"({sum(s['mode']=='heldout_combo' for s in scenarios)} held-out combos "
          f"+ 1 genetics reliability)")
    print(f"held-out combos leaked into training: {len(leaks)}  "
          f"{'OK' if not leaks else 'LEAK!'}")


if __name__ == "__main__":
    main()
