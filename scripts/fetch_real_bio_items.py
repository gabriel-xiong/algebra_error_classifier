"""
Acquire + normalize a REAL biology multiple-choice question set as the EVAL-ONLY
pool for the AP Bio / MCAT pivot (spec §9/§10: synthetic-train / real-eval).

WHY THIS EXISTS
---------------
The pivot's item bank is `synthetic-train / real-eval`: real, openly-licensed,
exam-style bio MCQs are reserved *exclusively* as the held-out eval set and are
NEVER trained on (docs/mcat_pivot_spec.md §9.1, §10.1). This script pulls such
items from openly-licensed datasets and normalizes them into the project's AP Bio
item schema (data/apbio_item_template.jsonl), mirroring the conventions in
scripts/real_data.py (field aliases, best-effort taxonomy mapping, provenance).

SOURCING RULES (important — see docs/real_data.md)
--------------------------------------------------
We do NOT scrape copyrighted/paywalled exam content (College Board AP, AAMC MCAT).
We use openly-licensed, exam-style biology MCQ datasets instead, priority order:

  1. MMLU `high_school_biology` + `college_biology` (cais/mmlu)  [MIT license]
     Clean 4-option single-best-answer with an answer key; closest to AP-Bio.
  2. SciQ (allenai/sciq)                                         [CC BY-NC 3.0]
     Crowdsourced science MCQs w/ support text + 3 distractors + correct answer;
     filtered here to biology-relevant items.
  3. ARC (allenai/ai2_arc, Challenge + Easy)                     [CC BY-SA 4.0]
     Grade-school science MCQs; filtered here to biology-relevant items.

WHAT IT PRODUCES
----------------
`data/real_bio_eval_raw.jsonl` — a RAW, UNTAGGED eval pool. Every wrong option
carries `{"tag": null, "needs_tagging": true}` because real static items have a
correct answer but NO student-chosen distractor and NO misconception tags. This is
a raw pool awaiting the tagging + human-verification NEXT STEP (see docs/real_data.md).

Distractor-choice error-typing needs a *student-selected* distractor, which these
static items don't have. So for eval, the protocol is to treat each wrong option as
a *potential* chosen distractor to be tagged (frontier-assisted draft, then human
verify ~40-50 to form the gold eval set). Real = eval only.

SCHEMA (mirrors data/apbio_item_template.jsonl, plus raw-eval provenance fields)
-------------------------------------------------------------------------------
  {
    "id": "mmlu_hsbio_test_0007",
    "topic": "evolution",              # best-effort -> one of the 7 taxonomy topics,
                                       #   or "unmapped" if no confident match
    "subtopic": null,
    "mcat_skill": null,                # unknown for real items
    "knowledge_type": null,            # unknown for real items
    "difficulty": null,
    "passage": null,                   # present for SciQ support text, else null
    "stem": "....",
    "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},   # apbio dict form
    "correct": "A",                    # letter key of the correct choice
    "correct_answer": "directional selection.",   # convenience: the correct text
    "distractor_tags": {               # ONLY wrong options; all await tagging
      "B": {"tag": null, "needs_tagging": true},
      "C": {"tag": null, "needs_tagging": true},
      "D": {"tag": null, "needs_tagging": true}
    },
    "source": "mmlu:high_school_biology:test",
    "source_license": "MIT",
    "provenance": "real_eval",
    "authoring": {
      "source": "real_eval", "validated_by": null,
      "notes": "RAW eval pool. Untagged: no per-distractor misconception, no student
                choice/timing. Awaiting frontier-assisted tag draft + human verify."
    }
  }

DESIGN NOTE — `choices` is a letter-keyed dict (A/B/C/D), not a bare list, to match
data/apbio_item_template.jsonl exactly so the pool is directly consumable by
scripts/common_bio.build_user_prompt (which reads `item["choices"]` as a dict) and
by the tagger. `correct` is the letter; `correct_answer` carries the raw text.

USAGE
-----
  # default: all three sources, up to ~400 items
  python scripts/fetch_real_bio_items.py --out data/real_bio_eval_raw.jsonl

  # only MMLU, cap 150, only mapped-topic items
  python scripts/fetch_real_bio_items.py --sources mmlu --max-items 150 --require-mapped-topic

  # pick sources + a topic filter
  python scripts/fetch_real_bio_items.py --sources mmlu sciq --topics evolution genetics

If the `datasets` HF download fails in this environment, the script reports the
failure per-source and continues with whatever succeeded (it never fabricates items).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_bio import write_jsonl  # noqa: E402  (self-contained JSONL writer)

# --------------------------------------------------------------------- topics

# The 7 taxonomy topics (data/apbio_misconceptions.json _meta.topics).
TAXONOMY_TOPICS = [
    "evolution",
    "genetics",
    "cellular_respiration",
    "photosynthesis",
    "membrane_transport",
    "enzymes",
    "experimental_design",
]

# Best-effort keyword -> topic mapping. Ordered: more specific phrases first so a
# stem that mentions several themes lands on the most specific one. Matched
# case-insensitively as substrings against the stem (+ choices). This is a
# STARTING POINT for hand-review, exactly like real_data.MISCONCEPTION_KEYWORDS —
# not ground truth.
TOPIC_KEYWORDS: list[tuple[str, str]] = [
    # cellular respiration
    ("electron transport chain", "cellular_respiration"),
    ("oxidative phosphorylation", "cellular_respiration"),
    ("citric acid cycle", "cellular_respiration"),
    ("krebs cycle", "cellular_respiration"),
    ("glycolysis", "cellular_respiration"),
    ("fermentation", "cellular_respiration"),
    ("cellular respiration", "cellular_respiration"),
    ("aerobic respiration", "cellular_respiration"),
    ("anaerobic", "cellular_respiration"),
    ("nadh", "cellular_respiration"),
    ("atp synthase", "cellular_respiration"),
    # photosynthesis
    ("calvin cycle", "photosynthesis"),
    ("light-dependent reaction", "photosynthesis"),
    ("light dependent reaction", "photosynthesis"),
    ("light reaction", "photosynthesis"),
    ("photosystem", "photosynthesis"),
    ("photolysis", "photosynthesis"),
    ("chlorophyll", "photosynthesis"),
    ("chloroplast", "photosynthesis"),
    ("photosynthesis", "photosynthesis"),
    # membrane transport
    ("facilitated diffusion", "membrane_transport"),
    ("active transport", "membrane_transport"),
    ("passive transport", "membrane_transport"),
    ("sodium-potassium pump", "membrane_transport"),
    ("sodium potassium pump", "membrane_transport"),
    ("hypertonic", "membrane_transport"),
    ("hypotonic", "membrane_transport"),
    ("isotonic", "membrane_transport"),
    ("osmosis", "membrane_transport"),
    ("plasmolysis", "membrane_transport"),
    ("turgor", "membrane_transport"),
    ("semipermeable", "membrane_transport"),
    ("selectively permeable", "membrane_transport"),
    ("concentration gradient", "membrane_transport"),
    ("diffuse across", "membrane_transport"),
    ("cell membrane", "membrane_transport"),
    ("plasma membrane", "membrane_transport"),
    ("phospholipid", "membrane_transport"),
    # enzymes
    ("activation energy", "enzymes"),
    ("active site", "enzymes"),
    ("competitive inhibitor", "enzymes"),
    ("noncompetitive inhibitor", "enzymes"),
    ("non-competitive inhibitor", "enzymes"),
    ("enzyme", "enzymes"),
    ("catalyst", "enzymes"),
    ("catalyze", "enzymes"),
    ("substrate", "enzymes"),
    ("denatur", "enzymes"),
    # genetics
    ("punnett", "genetics"),
    ("dihybrid", "genetics"),
    ("monohybrid", "genetics"),
    ("heterozygous", "genetics"),
    ("homozygous", "genetics"),
    ("independent assortment", "genetics"),
    ("incomplete dominance", "genetics"),
    ("codominance", "genetics"),
    ("sex-linked", "genetics"),
    ("sex linked", "genetics"),
    ("x-linked", "genetics"),
    ("allele", "genetics"),
    ("genotype", "genetics"),
    ("phenotype", "genetics"),
    ("dominant", "genetics"),
    ("recessive", "genetics"),
    ("heredity", "genetics"),
    ("inheritance", "genetics"),
    ("chromosome", "genetics"),
    ("meiosis", "genetics"),
    ("mutation", "genetics"),
    ("dna replication", "genetics"),
    ("transcription", "genetics"),
    ("translation", "genetics"),
    ("codon", "genetics"),
    ("messenger rna", "genetics"),
    ("mrna", "genetics"),
    # evolution
    ("natural selection", "evolution"),
    ("directional selection", "evolution"),
    ("stabilizing selection", "evolution"),
    ("disruptive selection", "evolution"),
    ("sexual selection", "evolution"),
    ("genetic drift", "evolution"),
    ("gene flow", "evolution"),
    ("hardy-weinberg", "evolution"),
    ("hardy weinberg", "evolution"),
    ("speciation", "evolution"),
    ("common ancestor", "evolution"),
    ("phylogen", "evolution"),
    ("adaptation", "evolution"),
    ("fitness", "evolution"),
    ("darwin", "evolution"),
    ("evolv", "evolution"),
    ("evolution", "evolution"),
    ("survival of the fittest", "evolution"),
    # experimental design (checked late so content topics win first)
    ("independent variable", "experimental_design"),
    ("dependent variable", "experimental_design"),
    ("control group", "experimental_design"),
    ("controlled experiment", "experimental_design"),
    ("hypothesis", "experimental_design"),
    ("experimental design", "experimental_design"),
]

# Keywords used to decide whether a general-science item (SciQ / ARC) is
# biology-relevant enough to keep. Broad on purpose; topic mapping is separate.
BIOLOGY_RELEVANCE_KEYWORDS = [
    "cell", "gene", "dna", "rna", "protein", "enzyme", "organism", "species",
    "evolut", "photosynth", "respiration", "mitochond", "chloroplast", "membrane",
    "osmosis", "diffusion", "allele", "chromosome", "phenotype", "genotype",
    "mutation", "meiosis", "mitosis", "tissue", "organ", "blood", "hormone",
    "bacteria", "virus", "fungi", "plant", "animal", "ecosystem", "population",
    "predator", "prey", "photosyn", "digest", "immune", "nervous", "muscle",
    "reproduc", "inherit", "natural selection", "biolog", "nucleus", "ribosome",
    "metabolis", "glucose", "atp", "homeostasis", "vaccin", "antibod", "amino acid",
]

_LETTERS = "ABCDEFGH"


def map_topic(text: str) -> tuple[str, bool]:
    """Best-effort map a stem (+choices) to one of the 7 taxonomy topics.

    Returns (topic, confident). `confident` is False (topic="unmapped") when no
    keyword matched, so a human can review/skip those before using as gold eval.
    """
    low = text.lower()
    for needle, topic in TOPIC_KEYWORDS:
        if needle in low:
            return topic, True
    return "unmapped", False


def is_biology(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in BIOLOGY_RELEVANCE_KEYWORDS)


# --------------------------------------------------------------------- normalize

def _norm_stem(stem: str) -> str:
    """Normalized key for dedup: lowercase, collapse whitespace, strip punctuation."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", stem.lower())).strip()


def make_item(
    *,
    item_id: str,
    stem: str,
    choices: list[str],
    correct_index: int,
    source: str,
    source_license: str,
    passage: str | None = None,
) -> dict | None:
    """Assemble one normalized apbio-schema item, or None if it fails validity.

    Validity: exactly one correct answer index in range, >= 3 non-empty choices.
    """
    stem = (stem or "").strip()
    choices = [str(c).strip() for c in choices if str(c).strip()]
    if not stem or len(choices) < 3:
        return None
    if not (0 <= correct_index < len(choices)):
        return None
    if len(choices) > len(_LETTERS):
        return None

    choice_map = {_LETTERS[i]: choices[i] for i in range(len(choices))}
    correct_letter = _LETTERS[correct_index]

    # Every WRONG option awaits tagging (no student choice / no misconception yet).
    distractor_tags = {
        letter: {"tag": None, "needs_tagging": True}
        for letter in choice_map
        if letter != correct_letter
    }

    topic, _confident = map_topic(stem + " " + " ".join(choices))

    return {
        "id": item_id,
        "topic": topic,
        "subtopic": None,
        "mcat_skill": None,
        "knowledge_type": None,
        "difficulty": None,
        "passage": passage.strip() if passage and passage.strip() else None,
        "stem": stem,
        "choices": choice_map,
        "correct": correct_letter,
        "correct_answer": choice_map[correct_letter],
        "distractor_tags": distractor_tags,
        "source": source,
        "source_license": source_license,
        "provenance": "real_eval",
        "authoring": {
            "source": "real_eval",
            "validated_by": None,
            "notes": (
                "RAW eval pool item. UNTAGGED: no per-distractor misconception, no "
                "student choice, no timing. Awaiting frontier-assisted tag draft + "
                "human verification (~40-50) to form the gold eval set. Real = eval only."
            ),
        },
    }


# --------------------------------------------------------------------- sources

def fetch_mmlu(max_items: int) -> tuple[list[dict], dict]:
    """MMLU high_school_biology + college_biology (test+validation). MIT license."""
    from datasets import load_dataset

    out: list[dict] = []
    meta = {"license": "MIT", "attempted": True, "error": None}
    configs = ["high_school_biology", "college_biology"]
    short = {"high_school_biology": "hsbio", "college_biology": "colbio"}
    try:
        for cfg in configs:
            for split in ["test", "validation"]:
                ds = load_dataset("cais/mmlu", cfg, split=split)
                for i, row in enumerate(ds):
                    if len(out) >= max_items:
                        break
                    item = make_item(
                        item_id=f"mmlu_{short[cfg]}_{split}_{i:04d}",
                        stem=row["question"],
                        choices=list(row["choices"]),
                        correct_index=int(row["answer"]),
                        source=f"mmlu:{cfg}:{split}",
                        source_license="MIT",
                    )
                    if item:
                        out.append(item)
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break
    except Exception as exc:  # noqa: BLE001 — report, never fabricate
        meta["error"] = f"{type(exc).__name__}: {exc}"
    meta["count"] = len(out)
    return out, meta


def fetch_sciq(max_items: int) -> tuple[list[dict], dict]:
    """SciQ (allenai/sciq), filtered to biology-relevant items. CC BY-NC 3.0.

    SciQ gives correct_answer + distractor1..3 + optional support text. We build a
    deterministic choice order (correct first, then distractors) so we never rely
    on a random seed; downstream tagging treats options by content, not position.
    """
    from datasets import load_dataset

    out: list[dict] = []
    meta = {"license": "CC BY-NC 3.0", "attempted": True, "error": None}
    try:
        for split in ["train", "validation", "test"]:
            ds = load_dataset("allenai/sciq", split=split)
            for i, row in enumerate(ds):
                if len(out) >= max_items:
                    break
                q = row.get("question", "")
                support = row.get("support", "") or ""
                if not is_biology(q + " " + support):
                    continue
                correct = row.get("correct_answer", "")
                distractors = [
                    row.get("distractor1", ""),
                    row.get("distractor2", ""),
                    row.get("distractor3", ""),
                ]
                choices = [correct] + distractors
                item = make_item(
                    item_id=f"sciq_{split}_{i:05d}",
                    stem=q,
                    choices=choices,
                    correct_index=0,  # correct placed first, deterministically
                    source=f"sciq:{split}",
                    source_license="CC BY-NC 3.0",
                    passage=support if support.strip() else None,
                )
                if item:
                    out.append(item)
            if len(out) >= max_items:
                break
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
    meta["count"] = len(out)
    return out, meta


def fetch_arc(max_items: int) -> tuple[list[dict], dict]:
    """ARC (allenai/ai2_arc) Challenge+Easy, filtered to biology. CC BY-SA 4.0."""
    from datasets import load_dataset

    out: list[dict] = []
    meta = {"license": "CC BY-SA 4.0", "attempted": True, "error": None}
    try:
        for cfg in ["ARC-Challenge", "ARC-Easy"]:
            for split in ["test", "validation", "train"]:
                ds = load_dataset("allenai/ai2_arc", cfg, split=split)
                for i, row in enumerate(ds):
                    if len(out) >= max_items:
                        break
                    q = row.get("question", "")
                    if not is_biology(q):
                        continue
                    labels = list(row["choices"]["label"])
                    texts = list(row["choices"]["text"])
                    answer_key = row.get("answerKey", "")
                    if answer_key not in labels:
                        continue
                    correct_index = labels.index(answer_key)
                    item = make_item(
                        item_id=f"arc_{cfg.lower().replace('-', '')}_{split}_{i:05d}",
                        stem=q,
                        choices=texts,
                        correct_index=correct_index,
                        source=f"ai2_arc:{cfg}:{split}",
                        source_license="CC BY-SA 4.0",
                    )
                    if item:
                        out.append(item)
                if len(out) >= max_items:
                    break
            if len(out) >= max_items:
                break
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"{type(exc).__name__}: {exc}"
    meta["count"] = len(out)
    return out, meta


SOURCE_FETCHERS = {
    "mmlu": fetch_mmlu,
    "sciq": fetch_sciq,
    "arc": fetch_arc,
}


# --------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire + normalize openly-licensed real bio MCQs as the real-eval pool."
    )
    parser.add_argument(
        "--out", default="data/real_bio_eval_raw.jsonl",
        help="output JSONL (normalized raw eval pool)",
    )
    parser.add_argument(
        "--sources", nargs="+", default=["mmlu", "sciq", "arc"],
        choices=list(SOURCE_FETCHERS),
        help="which sources to pull, priority order (default: all)",
    )
    parser.add_argument(
        "--max-items", type=int, default=400,
        help="approx TOTAL cap across sources (per-source budget derived from this)",
    )
    parser.add_argument(
        "--per-source-max", type=int, default=None,
        help="hard per-source cap (overrides the derived budget)",
    )
    parser.add_argument(
        "--topics", nargs="+", default=None, choices=TAXONOMY_TOPICS,
        help="keep only items mapped to these taxonomy topics",
    )
    parser.add_argument(
        "--require-mapped-topic", action="store_true",
        help="drop items whose topic could not be mapped to the 7-topic taxonomy",
    )
    args = parser.parse_args()

    # Derive a per-source budget so the priority order (mmlu > sciq > arc) fills first.
    per_source = args.per_source_max or args.max_items

    collected: list[dict] = []
    source_meta: dict[str, dict] = {}
    seen_stems: set[str] = set()
    dropped_dedup = 0

    for name in args.sources:
        remaining = args.max_items - len(collected)
        if remaining <= 0:
            source_meta[name] = {"attempted": False, "count": 0, "error": "skipped (cap reached)"}
            continue
        budget = min(per_source, remaining)
        items, meta = SOURCE_FETCHERS[name](budget)
        source_meta[name] = meta
        if meta.get("error"):
            print(f"[WARN] source '{name}' failed: {meta['error']}", file=sys.stderr)
        # Dedup across all sources by normalized stem.
        kept_from_source = 0
        for it in items:
            key = _norm_stem(it["stem"])
            if key in seen_stems:
                dropped_dedup += 1
                continue
            seen_stems.add(key)
            collected.append(it)
            kept_from_source += 1
        source_meta[name]["kept_after_dedup"] = kept_from_source

    # Topic filters (applied after acquisition so counts are transparent).
    dropped_topic = 0
    if args.topics or args.require_mapped_topic:
        filtered = []
        for it in collected:
            if args.require_mapped_topic and it["topic"] == "unmapped":
                dropped_topic += 1
                continue
            if args.topics and it["topic"] not in args.topics:
                dropped_topic += 1
                continue
            filtered.append(it)
        collected = filtered

    write_jsonl(args.out, collected)

    # ------------------------------------------------------------- summary report
    per_source_kept = Counter(it["source"].split(":")[0] for it in collected)
    per_topic = Counter(it["topic"] for it in collected)
    with_passage = sum(1 for it in collected if it["passage"])

    print("=" * 72)
    print("REAL BIO EVAL POOL — acquisition summary")
    print("=" * 72)
    print(f"Output: {args.out}")
    print(f"TOTAL items written: {len(collected)}")
    print()
    print("Per-source (acquired -> kept after dedup):")
    for name in args.sources:
        m = source_meta.get(name, {})
        status = "OK" if not m.get("error") else f"FAILED ({m['error']})"
        print(
            f"  {name:6s} license={m.get('license', '?'):14s} "
            f"acquired={m.get('count', 0):4d} kept={m.get('kept_after_dedup', 0):4d}  {status}"
        )
    print(f"  (family counts in final pool: {dict(per_source_kept)})")
    print()
    print("Per-topic distribution (best-effort taxonomy map):")
    for topic in TAXONOMY_TOPICS + ["unmapped"]:
        if per_topic.get(topic):
            print(f"  {topic:22s} {per_topic[topic]:4d}")
    print()
    print(f"Items with passage/support text: {with_passage}")
    print(f"Dropped by dedup (duplicate stems): {dropped_dedup}")
    if args.topics or args.require_mapped_topic:
        print(f"Dropped by topic filter: {dropped_topic}")
    print()
    print("REALITY FLAGS:")
    print("  * These items are UNTAGGED: no per-distractor misconception, no student")
    print("    choice, no timing. Every wrong option has needs_tagging=true.")
    print("  * This is a RAW POOL. NEXT STEP: draft per-distractor misconception tags")
    print("    (frontier-assisted), then HUMAN-verify ~40-50 to form the gold eval set.")
    print("  * Real items = EVAL ONLY (never trained on); see docs/mcat_pivot_spec.md §9/§10.")
    print("=" * 72)


if __name__ == "__main__":
    main()
