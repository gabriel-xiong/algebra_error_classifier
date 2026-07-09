"""
By-construction generator for AP Bio CELLULAR RESPIRATION items (v1).

This is the CONCEPTUAL / role-swap counterpart to the procedural genetics
generator. Cellular-respiration misconceptions are false *statements* rather
than buggy computations, so we cannot auto-verify a distractor by recomputation.
Instead we curate FRAMES: each frame is a question whose correct answer is a
true statement, paired with a pool of misconception-tagged false statements that
are all plausible-but-wrong answers to THAT question. Each frame guarantees >=3
competing misconceptions, so every generated item has 3 distinctly-named errors.

Tags are trustworthy by construction because a human authored each false
statement to express exactly one misconception. Surface variety (stem / correct
/ distractor paraphrases + which 3-of-N misconceptions are used + option order)
expands a handful of frames into hundreds of distinct items. Genetics carries
training volume; this generator supplies conceptual diversity so the fine-tuned
model learns the generation SKILL, not one topic's template.

The eval's LLM-judge later measures whether a MODEL's generations map cleanly to
misconceptions; these authored items are the training target for that behavior.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TOPIC = "cellular_respiration"

# misconception_id -> coarse error_type (from data/apbio_misconceptions.json)
ERROR_TYPE = {
    "cr_o2_is_electron_donor": "content_gap",
    "cr_fermentation_makes_atp": "reasoning_error",
    "cr_etc_runs_anaerobically": "content_gap",
    "cr_glycolysis_needs_o2": "content_gap",
    "cr_o2_phosphorylates_adp": "reasoning_error",
    "cr_nadh_makes_atp_directly": "content_gap",
    "cr_lactate_to_pyruvate_backward": "content_gap",
}


# Each frame: stems + correct-answer paraphrases + a pool of misconception
# distractors (>=3), each with paraphrase variants. All strings within a frame
# must be mutually non-overlapping in meaning.
FRAMES = [
    {
        "id": "atp_production",
        "subtopic": "atp_yield",
        "stems": [
            "Which statement correctly describes how most ATP is produced in aerobic respiration?",
            "In a cell with plenty of oxygen, where does the bulk of ATP come from?",
        ],
        "correct": [
            "Most ATP is made by oxidative phosphorylation as electrons pass down the electron transport chain to oxygen.",
            "The majority of ATP is produced by the electron transport chain and ATP synthase using oxygen as the final acceptor.",
        ],
        "distractors": {
            "cr_fermentation_makes_atp": [
                "Fermentation generates most of the cell's ATP directly.",
                "The bulk of ATP is produced directly by the fermentation reactions.",
            ],
            "cr_nadh_makes_atp_directly": [
                "NADH is used directly as the cell's ATP for most cellular work.",
                "Most ATP is simply the NADH molecules produced earlier in respiration.",
            ],
            "cr_o2_phosphorylates_adp": [
                "Oxygen directly phosphorylates ADP into ATP.",
                "Oxygen itself adds a phosphate to ADP to form most of the ATP.",
            ],
            "cr_etc_runs_anaerobically": [
                "The electron transport chain makes it whether or not oxygen is present.",
                "An electron transport chain that runs without oxygen supplies most of it.",
            ],
        },
    },
    {
        "id": "no_oxygen",
        "subtopic": "anaerobic_conditions",
        "stems": [
            "What happens to cellular respiration when oxygen is not available?",
            "In the absence of oxygen, how is a cell able to keep making any ATP?",
        ],
        "correct": [
            "The electron transport chain halts, and the cell relies on glycolysis with fermentation regenerating NAD+.",
            "Oxidative phosphorylation stops, but glycolysis continues because fermentation regenerates NAD+.",
        ],
        "distractors": {
            "cr_etc_runs_anaerobically": [
                "The electron transport chain continues at its normal rate without oxygen.",
                "The electron transport chain keeps running normally even with no oxygen present.",
            ],
            "cr_glycolysis_needs_o2": [
                "All of respiration, including glycolysis, stops immediately because glycolysis needs oxygen.",
                "Glycolysis also shuts down, since it cannot proceed without oxygen.",
            ],
            "cr_fermentation_makes_atp": [
                "Fermentation takes over as the main direct producer of large amounts of ATP.",
                "The cell switches to fermentation, which now directly produces most of its ATP.",
            ],
            "cr_nadh_makes_atp_directly": [
                "The cell spends its stored NADH directly as ATP to get by.",
                "NADH is used straight as ATP until oxygen returns.",
            ],
        },
    },
    {
        "id": "fermentation_role",
        "subtopic": "fermentation",
        "stems": [
            "What is the primary role of fermentation in a cell running without oxygen?",
            "Why does fermentation matter when oxygen is unavailable?",
        ],
        "correct": [
            "It regenerates NAD+ so that glycolysis can continue producing ATP.",
            "It recycles NAD+ from NADH, allowing glycolysis to keep going.",
        ],
        "distractors": {
            "cr_fermentation_makes_atp": [
                "It directly produces the majority of the cell's ATP.",
                "Fermentation itself is the step that generates most ATP.",
            ],
            "cr_lactate_to_pyruvate_backward": [
                "It converts lactate back into pyruvate to regenerate NAD+.",
                "It breaks lactate down into pyruvate, restoring NAD+ in the process.",
            ],
            "cr_nadh_makes_atp_directly": [
                "It converts NADH directly into ATP for the cell to use.",
                "It turns NADH straight into usable ATP.",
            ],
            "cr_etc_runs_anaerobically": [
                "It keeps the electron transport chain running normally without oxygen.",
                "It lets the electron transport chain operate as usual in the absence of oxygen.",
            ],
        },
    },
    {
        "id": "role_of_oxygen",
        "subtopic": "electron_transport_chain",
        "stems": [
            "What is the direct role of oxygen in the electron transport chain?",
            "During aerobic respiration, oxygen acts primarily as which of the following?",
        ],
        "correct": [
            "It is the final (terminal) electron acceptor at the end of the electron transport chain.",
            "It accepts electrons at the end of the electron transport chain, forming water.",
        ],
        "distractors": {
            "cr_o2_is_electron_donor": [
                "It donates electrons into the start of the electron transport chain.",
                "It is the initial electron donor that feeds the transport chain.",
            ],
            "cr_o2_phosphorylates_adp": [
                "It directly phosphorylates ADP to produce ATP.",
                "It adds a phosphate to ADP, directly forming ATP.",
            ],
            "cr_etc_runs_anaerobically": [
                "It has no essential role, since the electron transport chain runs fine without it.",
                "It is unnecessary, because the transport chain operates with or without oxygen.",
            ],
        },
    },
    {
        "id": "electron_carriers",
        "subtopic": "electron_carriers",
        "stems": [
            "What is the role of NADH produced during glycolysis and the citric acid cycle?",
            "How does NADH contribute to ATP production in respiration?",
        ],
        "correct": [
            "It carries electrons to the electron transport chain, where their energy drives ATP synthesis.",
            "It delivers high-energy electrons to the electron transport chain.",
        ],
        "distractors": {
            "cr_nadh_makes_atp_directly": [
                "It is itself the ATP that the cell uses for energy.",
                "NADH is spent directly as ATP for cellular work.",
            ],
            "cr_o2_is_electron_donor": [
                "It receives its electrons directly from oxygen.",
                "It is loaded with electrons donated by oxygen.",
            ],
            "cr_fermentation_makes_atp": [
                "Its only role is in fermentation, which is the step that directly produces the ATP.",
                "It is used only in fermentation, and fermentation is what actually generates the ATP.",
            ],
        },
    },
    {
        "id": "glycolysis_basics",
        "subtopic": "glycolysis",
        "stems": [
            "Which statement about glycolysis is correct?",
            "What is true of glycolysis in a typical cell?",
        ],
        "correct": [
            "It occurs in the cytosol and does not require oxygen.",
            "It takes place in the cytoplasm and can proceed whether or not oxygen is present.",
        ],
        "distractors": {
            "cr_glycolysis_needs_o2": [
                "It requires oxygen and cannot proceed without it.",
                "It halts whenever oxygen is unavailable, since it is an aerobic step.",
            ],
            "cr_etc_runs_anaerobically": [
                "It feeds the electron transport chain, which runs equally well with or without oxygen.",
                "Its products drive an electron transport chain that needs no oxygen.",
            ],
            "cr_nadh_makes_atp_directly": [
                "Its NADH is used directly as ATP by the cell.",
                "The NADH it makes is spent straight as ATP.",
            ],
            "cr_fermentation_makes_atp": [
                "It is a form of fermentation that directly produces most of the cell's ATP.",
                "It is essentially fermentation and directly yields the bulk of the ATP.",
            ],
        },
    },
]


def _sig(stem: str, choices: dict) -> str:
    return stem + "||" + "|".join(sorted(choices.values()))


def generate_item(rng: random.Random, idx: int) -> dict:
    frame = rng.choice(FRAMES)
    stem = rng.choice(frame["stems"])
    correct_text = rng.choice(frame["correct"])

    mids = rng.sample(list(frame["distractors"]), 3)
    distractors = [(mid, rng.choice(frame["distractors"][mid])) for mid in mids]

    options = [("__correct__", correct_text)] + [(mid, txt) for mid, txt in distractors]
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    choices, tags, correct_letter = {}, {}, None
    for letter, (mid, text) in zip(letters, options):
        choices[letter] = text
        if mid == "__correct__":
            correct_letter = letter
        else:
            tags[letter] = {
                "error_type": ERROR_TYPE[mid],
                "misconception_id": mid,
                "rationale": f"This choice expresses the misconception '{mid}': "
                             f"a plausible but incorrect statement a student holding "
                             f"that belief would select.",
            }

    return {
        "id": f"gen_cellresp_{idx:04d}",
        "topic": TOPIC,
        "subtopic": frame["subtopic"],
        "knowledge_type": "declarative",
        "difficulty": "medium",
        "passage": None,
        "stem": stem,
        "choices": choices,
        "correct": correct_letter,
        "distractor_tags": tags,
        "authoring": {
            "source": "by_construction",
            "generator": "scripts/gen_cellresp.py",
            "frame": frame["id"],
        },
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--count", type=int, default=300)
    ap.add_argument("-o", "--out", default="data/gen_cellresp.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.selftest:
        items = [generate_item(rng, i) for i in range(20)]
        ok = sum(verify_item(it)[0] for it in items)
        print(f"generated 20, verified {ok}/20")
        print(json.dumps(items[0], indent=2))
        return

    rows, seen, tries = [], set(), 0
    while len(rows) < args.count and tries < args.count * 60:
        tries += 1
        it = generate_item(rng, len(rows))
        sig = _sig(it["stem"], it["choices"])
        if sig in seen or not verify_item(it)[0]:
            continue
        seen.add(sig)
        rows.append(it)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} items -> {out}  "
          f"(requested {args.count}; unique cap reached if fewer)")


if __name__ == "__main__":
    main()
