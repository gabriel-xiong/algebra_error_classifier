"""
Frontier-assisted tag DRAFTING for the real AP-Bio eval pool.

This is the step BETWEEN the raw, untagged real eval pool
(`data/real_bio_eval_raw.jsonl`, 750 items harvested by
`scripts/fetch_real_bio_items.py`) and a HUMAN-VERIFIED gold eval set. It does
NOT produce gold labels. It produces DRAFTS a human (the owner) then verifies.

Why drafts, not gold (see docs/real_data.md, docs/mcat_pivot_spec.md 9):
  Real = eval only. These static MCQs have a correct answer but NO
  student-chosen distractor and NO misconception tags. We treat EACH WRONG
  option as a potential chosen distractor and ask a frontier model to draft the
  best-fit mid-grained misconception (from data/apbio_misconceptions.json), a
  one-line rationale, a confidence, and a `no_fit` flag when nothing cleanly
  applies. Real distractors will NOT always map to an authored misconception --
  that mismatch is expected and is useful signal on taxonomy coverage.

Outputs (two, side by side):
  1. data/real_bio_eval_drafted.jsonl  -- same schema as the raw pool, but each
     wrong option's distractor_tags entry is filled with the draft
     {tag, rationale, confidence, no_fit, needs_review, tag_source}. The correct
     option stays untagged. provenance stays "real_eval".
  2. data/real_bio_eval_worksheet.csv  -- one row per (item x wrong-distractor)
     with the draft PLUS blank columns the owner fills in:
     confirmed(Y/N), corrected_misconception_id, notes. This is the
     human-verification worksheet that yields the gold set.

Sampling: the raw pool's `topic` is a best-effort map onto the 7 taxonomy
topics; ~444/750 land on "unmapped". We draft only the AP-Bio-mapped core
(topic in the 7 taxonomy topics) and STRATIFY across those topics with a
per-topic cap so small topics (enzymes, experimental_design) are represented
while the large ones (genetics, evolution) stay largest.

API key handling (CRITICAL -- never leak):
  The key is loaded, in order, from: env OPENAI_API_KEY, env ANTHROPIC_API_KEY,
  or a gitignored local file (.local/.env then .env) parsed for those names.
  The provider is auto-detected from whichever is present. The key value is
  NEVER printed, logged, or written to any file. If NO key is found the script
  does NOT fail: it writes the worksheet with EMPTY draft columns and prints a
  clear "set OPENAI_API_KEY / ANTHROPIC_API_KEY (or add it to .local/.env) and
  re-run" message.

Usage:
  # Local plumbing smoke test (no key, no network, Windows CPU):
  python scripts/draft_tags.py --selftest

  # Real drafts (needs a key in env or .local/.env; auto-detects provider):
  python scripts/draft_tags.py --n 80 --seed 0

  # Restrict topics:
  python scripts/draft_tags.py --topics genetics evolution --n 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the litmus tagging machinery read-only (frontier backend + prompt
# helpers). We do NOT modify it; drafting adds a richer JSON-returning call on
# top of the same provider/auth plumbing.
from litmus_tagging import (  # noqa: E402
    FrontierModel,
    SkipFrontier,
    _format_candidate_list,
    _format_choices,
    load_misconceptions,
)

DEFAULT_RAW = REPO_ROOT / "data" / "real_bio_eval_raw.jsonl"
DEFAULT_MISCONCEPTIONS = REPO_ROOT / "data" / "apbio_misconceptions.json"
DEFAULT_DRAFTED = REPO_ROOT / "data" / "real_bio_eval_drafted.jsonl"
DEFAULT_WORKSHEET = REPO_ROOT / "data" / "real_bio_eval_worksheet.csv"

# The 7 AP-Bio taxonomy topics (anything else, incl. "unmapped", is excluded).
TAXONOMY_TOPICS = [
    "evolution",
    "genetics",
    "cellular_respiration",
    "photosynthesis",
    "membrane_transport",
    "enzymes",
    "experimental_design",
]

STEM_TRIM = 160
LOW_CONFIDENCE = 0.5  # reporting threshold for "low-confidence" drafts


# --------------------------------------------------------------------- loading

def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wrong_options(item: dict) -> list[str]:
    """The candidate distractor letters = every option carrying needs_tagging.

    Falls back to "every choice that is not the correct one" if distractor_tags
    is absent, so the drafter is robust to minor schema drift.
    """
    tags = item.get("distractor_tags")
    if tags:
        return sorted(tags.keys())
    correct = item.get("correct")
    return sorted(k for k in item.get("choices", {}) if k != correct)


# ------------------------------------------------------------------- sampling

def stratified_sample(
    items: list[dict],
    topics: list[str],
    n: int,
    seed: int,
    cap: int | None = None,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Stratified sample of ~n items across the requested mapped topics.

    Allocation is proportional to each topic's availability but (a) floored so
    small topics are represented and (b) capped so no single topic dominates.
    The seed only controls WHICH items are drawn within a topic; the per-topic
    allocation is deterministic given (topics, counts, n, cap). Returns
    (selected_items, allocation, available_counts).
    """
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if item.get("topic") in topics:
            by_topic[item["topic"]].append(item)

    available = {t: len(by_topic[t]) for t in topics if by_topic[t]}
    if not available:
        return [], {}, {}

    active = sorted(available)
    total_available = sum(available.values())
    n = min(n, total_available)

    if cap is None:
        # No topic should exceed ~35% of the sample (keeps genetics/evolution
        # largest without letting them swamp the small topics).
        cap = max(1, round(0.35 * n))

    # Floor: give every active topic a baseline slice (bounded by availability),
    # so tiny topics like experimental_design always show up.
    floor = max(1, min(n // len(active), 8))
    alloc = {t: min(available[t], floor) for t in active}

    # Distribute the remainder proportional to leftover availability, capped.
    guard = 0
    while sum(alloc.values()) < n and guard < 10_000:
        guard += 1
        remaining = n - sum(alloc.values())
        room = {
            t: min(available[t], cap) - alloc[t]
            for t in active
            if min(available[t], cap) - alloc[t] > 0
        }
        if not room:
            break
        weights = {t: available[t] - alloc[t] for t in room}
        wtotal = sum(weights.values()) or 1
        progressed = False
        for t in sorted(room, key=lambda x: -weights[x]):
            budget = n - sum(alloc.values())
            if budget <= 0:
                break
            give = max(1, round(remaining * weights[t] / wtotal))
            give = min(give, room[t], budget)
            if give > 0:
                alloc[t] += give
                progressed = True
        if not progressed:
            break

    rng = random.Random(seed)
    selected: list[dict] = []
    for t in active:
        pool = list(by_topic[t])
        rng.shuffle(pool)
        selected.extend(pool[: alloc[t]])
    rng.shuffle(selected)
    return selected, alloc, available


# ---------------------------------------------------------------- key loading

def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv-style file for KEY=VALUE lines. Never logs values."""
    found: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return found
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") and value:
            found[key] = value
    return found


def load_api_key_into_env() -> str | None:
    """Ensure a key is in os.environ (from env or a gitignored local file).

    Order: env OPENAI_API_KEY, env ANTHROPIC_API_KEY, .local/.env, .env.
    Returns the detected provider ("openai" | "anthropic") or None. NEVER prints
    or returns the secret itself.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"

    for candidate in (REPO_ROOT / ".local" / ".env", REPO_ROOT / ".env"):
        if not candidate.is_file():
            continue
        parsed = _parse_env_file(candidate)
        # Load into the process env so the reused FrontierModel picks it up.
        for key, value in parsed.items():
            os.environ.setdefault(key, value)
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
    return None


# ---------------------------------------------------------------- draft prompt

DRAFT_SYSTEM_PROMPT = (
    "You are an expert AP Biology tutor building an evaluation set. A student "
    "answered a multiple-choice question incorrectly by choosing a specific "
    "wrong option. Your job is to DRAFT which mid-grained misconception from a "
    "fixed list best explains that specific wrong choice, so a human reviewer "
    "can verify it. Real distractors do not always map cleanly to a listed "
    "misconception; when none fits, say so honestly. You output ONLY a single "
    "JSON object, no prose, no markdown fences."
)


def build_draft_prompt(
    item: dict,
    chosen: str,
    topic_misconceptions: list[dict],
    all_misconceptions: list[dict],
) -> str:
    """Constrained-drafting prompt: topic candidates FIRST, then the full list."""
    choices = item.get("choices", {})
    chosen_text = choices.get(chosen, "")
    passage = item.get("passage")
    passage_block = f"Passage:\n{passage}\n\n" if passage else ""

    topic_block = (
        _format_candidate_list(topic_misconceptions)
        if topic_misconceptions
        else "  (none scoped to this topic)"
    )
    return f"""Candidate misconceptions for THIS item's topic ({item.get('topic')}) -- prefer these (id: name):

{topic_block}

Full misconception list (all topics; use only if none above fits, id: name):

{_format_candidate_list(all_misconceptions)}

{passage_block}Question: {item.get('stem', '')}
Choices:
{_format_choices(choices)}
Correct answer: {item.get('correct')}. {item.get('correct_answer', '')}
Student's chosen (incorrect) answer: {chosen}. {chosen_text}

Decide which single misconception best explains why THIS wrong answer was
attractive to the student. Respond with ONLY this JSON object:
{{"misconception_id": "<exact id from a list, or null>", "rationale": "<one short sentence>", "confidence": <number 0.0-1.0>, "no_fit": <true|false>}}
Set "no_fit": true and "misconception_id": null when no listed misconception
cleanly applies to this specific wrong choice."""


# ----------------------------------------------------------------- draft parse

def _extract_json(raw: str) -> dict | None:
    """Best-effort: pull the first {...} JSON object out of a model reply."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def normalize_draft(parsed: dict | None, valid_ids: set[str]) -> dict:
    """Coerce a parsed model reply into the draft record shape."""
    if not parsed:
        return {"tag": None, "rationale": "", "confidence": None, "no_fit": True}

    raw_id = parsed.get("misconception_id")
    if isinstance(raw_id, str):
        raw_id = raw_id.strip()
    no_fit = bool(parsed.get("no_fit", False))
    if raw_id in (None, "", "null", "none") or raw_id not in valid_ids:
        tag = None
        no_fit = True
    else:
        tag = raw_id

    confidence = parsed.get("confidence")
    try:
        confidence = round(float(confidence), 3)
        confidence = min(max(confidence, 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = None

    rationale = parsed.get("rationale") or ""
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {
        "tag": tag,
        "rationale": rationale.strip().replace("\n", " "),
        "confidence": confidence,
        "no_fit": no_fit if tag is None else False,
    }


# --------------------------------------------------------------------- drafters

class FrontierDrafter:
    """Wraps the reused FrontierModel with a JSON-returning drafting call.

    Reuses the litmus FrontierModel purely for provider auto-detection, key/lib
    presence checks (raises SkipFrontier when absent) and the underlying client.
    Retries with exponential backoff on transient API errors; a persistent
    failure is surfaced to the caller as a draft_failed record, never a crash.
    """

    tag_source = "frontier_draft"

    def __init__(self, provider: str, model_name: str | None, temperature: float,
                 max_retries: int = 4):
        self._model = FrontierModel(provider, model_name, temperature)
        self.provider = self._model.provider
        self.model_name = self._model.model_name
        self.max_retries = max_retries

    def _call(self, system: str, user: str) -> str:
        # Larger token budget than litmus (needs room for JSON + rationale).
        if self.provider == "openai":
            resp = self._model._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._model.temperature,
                max_tokens=200,
            )
            return resp.choices[0].message.content or ""
        resp = self._model._client.messages.create(
            model=self.model_name,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self._model.temperature,
            max_tokens=300,
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    def draft(self, item, chosen, topic_misc, all_misc, valid_ids):
        prompt = build_draft_prompt(item, chosen, topic_misc, all_misc)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                raw = self._call(DRAFT_SYSTEM_PROMPT, prompt)
                record = normalize_draft(_extract_json(raw), valid_ids)
                record["tag_source"] = self.tag_source
                return record
            except Exception as exc:  # noqa: BLE001 - transient API/rate errors
                last_err = exc
                # exponential backoff with a small cap; do not log the key
                time.sleep(min(2 ** attempt, 16))
        return {
            "tag": None,
            "rationale": f"draft_failed: {type(last_err).__name__}",
            "confidence": None,
            "no_fit": False,
            "tag_source": "draft_failed",
        }


class DummyDrafter:
    """Offline drafter for --selftest: assigns a random in-topic misconception.

    Exercises the full script path (sampling -> draft records -> jsonl + csv)
    on Windows CPU with no key and no network. Deterministic given the seed.
    """

    tag_source = "selftest_dummy"

    def __init__(self, by_topic_ids: dict[str, list[str]], all_ids: list[str], seed: int):
        self._by_topic = by_topic_ids
        self._all = all_ids
        self._rng = random.Random(seed)

    def draft(self, item, chosen, topic_misc, all_misc, valid_ids):
        pool = self._by_topic.get(item.get("topic")) or self._all
        roll = self._rng.random()
        if roll < 0.15:  # exercise the no_fit path too
            return {
                "tag": None,
                "rationale": "selftest: no listed misconception cleanly applies.",
                "confidence": round(self._rng.uniform(0.2, 0.5), 3),
                "no_fit": True,
                "tag_source": self.tag_source,
            }
        tag = self._rng.choice(pool)
        return {
            "tag": tag,
            "rationale": f"selftest dummy draft for chosen option {chosen}.",
            "confidence": round(self._rng.uniform(0.4, 0.95), 3),
            "no_fit": False,
            "tag_source": self.tag_source,
        }


# ----------------------------------------------------------------- draft build

def draft_items(drafter, items, misconceptions):
    """Return (drafted_items, worksheet_rows, stats)."""
    id_to_name = {m["id"]: m["name"] for m in misconceptions}
    valid_ids = set(id_to_name)
    by_topic_misc: dict[str, list[dict]] = defaultdict(list)
    for m in misconceptions:
        by_topic_misc[m["topic"]].append(m)

    drafted_items: list[dict] = []
    worksheet_rows: list[dict] = []
    stats = {
        "items": 0,
        "distractors": 0,
        "no_fit": 0,
        "low_confidence": 0,
        "draft_failed": 0,
        "per_topic_items": Counter(),
        "per_topic_distractors": Counter(),
    }

    for item in items:
        topic = item.get("topic")
        stats["items"] += 1
        stats["per_topic_items"][topic] += 1
        topic_misc = by_topic_misc.get(topic, [])

        new_item = json.loads(json.dumps(item))  # deep copy, preserve schema
        tags = new_item.setdefault("distractor_tags", {})

        for letter in wrong_options(item):
            stats["distractors"] += 1
            stats["per_topic_distractors"][topic] += 1
            record = drafter.draft(item, letter, topic_misc, misconceptions, valid_ids)

            if record.get("tag_source") == "draft_failed":
                stats["draft_failed"] += 1
            if record.get("no_fit"):
                stats["no_fit"] += 1
            conf = record.get("confidence")
            if conf is not None and conf < LOW_CONFIDENCE:
                stats["low_confidence"] += 1

            tags[letter] = {
                "tag": record.get("tag"),
                "rationale": record.get("rationale", ""),
                "confidence": record.get("confidence"),
                "no_fit": bool(record.get("no_fit")),
                "needs_review": True,
                "tag_source": record.get("tag_source", drafter.tag_source),
            }

            stem = (item.get("stem") or "").replace("\n", " ")
            if len(stem) > STEM_TRIM:
                stem = stem[: STEM_TRIM - 1].rstrip() + "\u2026"
            worksheet_rows.append(
                {
                    "id": item.get("id"),
                    "topic": topic,
                    "stem": stem,
                    "choice_letter": letter,
                    "choice_text": item.get("choices", {}).get(letter, ""),
                    "correct_answer": item.get("correct_answer", ""),
                    "drafted_misconception_id": record.get("tag") or "",
                    "drafted_misconception_name": id_to_name.get(record.get("tag"), ""),
                    "rationale": record.get("rationale", ""),
                    "confidence": "" if conf is None else conf,
                    "no_fit": bool(record.get("no_fit")),
                    "confirmed(Y/N)": "",
                    "corrected_misconception_id": "",
                    "notes": "",
                }
            )
        drafted_items.append(new_item)
    return drafted_items, worksheet_rows, stats


def empty_worksheet_rows(items):
    """Worksheet rows with BLANK draft columns (no-key path)."""
    rows = []
    for item in items:
        topic = item.get("topic")
        stem = (item.get("stem") or "").replace("\n", " ")
        if len(stem) > STEM_TRIM:
            stem = stem[: STEM_TRIM - 1].rstrip() + "\u2026"
        for letter in wrong_options(item):
            rows.append(
                {
                    "id": item.get("id"),
                    "topic": topic,
                    "stem": stem,
                    "choice_letter": letter,
                    "choice_text": item.get("choices", {}).get(letter, ""),
                    "correct_answer": item.get("correct_answer", ""),
                    "drafted_misconception_id": "",
                    "drafted_misconception_name": "",
                    "rationale": "",
                    "confidence": "",
                    "no_fit": "",
                    "confirmed(Y/N)": "",
                    "corrected_misconception_id": "",
                    "notes": "",
                }
            )
    return rows


WORKSHEET_COLUMNS = [
    "id",
    "topic",
    "stem",
    "choice_letter",
    "choice_text",
    "correct_answer",
    "drafted_misconception_id",
    "drafted_misconception_name",
    "rationale",
    "confidence",
    "no_fit",
    "confirmed(Y/N)",
    "corrected_misconception_id",
    "notes",
]


def write_worksheet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKSHEET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------- report

def report_stats(stats, alloc, backend_label, drafted_path, worksheet_path):
    print("\n=== DRAFT TAGGING SUMMARY ===")
    print(f"Backend: {backend_label}")
    print(f"Items drafted:        {stats['items']}")
    print(f"Distractors drafted:  {stats['distractors']}")
    print(f"  no_fit (no clean misconception): {stats['no_fit']}")
    print(f"  low-confidence (<{LOW_CONFIDENCE}):          {stats['low_confidence']}")
    print(f"  draft_failed (API error):        {stats['draft_failed']}")
    print("\nPer-topic (items / distractors):")
    for topic in sorted(stats["per_topic_items"]):
        print(
            f"  {topic:<22} items={stats['per_topic_items'][topic]:<4}"
            f" distractors={stats['per_topic_distractors'][topic]}"
        )
    print(f"\nDrafted eval jsonl: {drafted_path}")
    print(f"Verification worksheet: {worksheet_path}")


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--misconceptions", default=str(DEFAULT_MISCONCEPTIONS))
    parser.add_argument("--drafted-out", default=None,
                        help="drafted jsonl path (default: data/real_bio_eval_drafted.jsonl)")
    parser.add_argument("--worksheet-out", default=None,
                        help="worksheet csv path (default: data/real_bio_eval_worksheet.csv)")
    parser.add_argument("--n", type=int, default=80, help="target sample size (~80)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topics", nargs="*", default=None,
                        help=f"subset of taxonomy topics (default: all of {TAXONOMY_TOPICS})")
    parser.add_argument("--cap", type=int, default=None,
                        help="max items per topic (default: ~35%% of n)")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "openai", "anthropic"])
    parser.add_argument("--frontier-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--selftest", action="store_true",
                        help="offline dummy drafter (no key, no network)")
    args = parser.parse_args()

    topics = args.topics or TAXONOMY_TOPICS
    unknown = [t for t in topics if t not in TAXONOMY_TOPICS]
    if unknown:
        raise SystemExit(f"--topics contains non-taxonomy topics: {unknown}. "
                         f"Allowed: {TAXONOMY_TOPICS}")

    misconceptions = load_misconceptions(args.misconceptions)
    print(f"Loaded {len(misconceptions)} misconceptions from {args.misconceptions}")

    items = load_jsonl(args.raw)
    print(f"Loaded {len(items)} raw items from {args.raw}")

    selected, alloc, available = stratified_sample(
        items, topics, args.n, args.seed, args.cap
    )
    print(f"Mapped-topic availability: "
          + ", ".join(f"{t}={c}" for t, c in sorted(available.items())))
    print(f"Stratified allocation (target n={args.n}): "
          + ", ".join(f"{t}={c}" for t, c in sorted(alloc.items()))
          + f"  -> {len(selected)} items selected")
    if not selected:
        raise SystemExit("No mapped-topic items to draft. Check --topics / raw pool.")

    # Default output paths; selftest writes to the gitignored outputs/ dir so it
    # never clobbers the real deliverables.
    if args.selftest:
        drafted_path = Path(args.drafted_out or REPO_ROOT / "outputs" / "selftest_drafted.jsonl")
        worksheet_path = Path(args.worksheet_out or REPO_ROOT / "outputs" / "selftest_worksheet.csv")
    else:
        drafted_path = Path(args.drafted_out or DEFAULT_DRAFTED)
        worksheet_path = Path(args.worksheet_out or DEFAULT_WORKSHEET)

    # ---- pick a drafter -------------------------------------------------
    if args.selftest:
        by_topic_ids: dict[str, list[str]] = defaultdict(list)
        for m in misconceptions:
            by_topic_ids[m["topic"]].append(m["id"])
        drafter = DummyDrafter(by_topic_ids, [m["id"] for m in misconceptions], args.seed)
        backend_label = "dummy (selftest)"
        print("Running SELF-TEST with the offline dummy drafter (no key / no network).")
    else:
        provider = load_api_key_into_env()
        if provider is None:
            # No key anywhere: ship the EMPTY worksheet + a clear instruction.
            rows = empty_worksheet_rows(selected)
            write_worksheet(worksheet_path, rows)
            print("\n[NO API KEY FOUND] No OPENAI_API_KEY or ANTHROPIC_API_KEY in the")
            print("environment, and none in .local/.env or .env.")
            print(f"Wrote an EMPTY-DRAFT worksheet ({len(rows)} rows) to: {worksheet_path}")
            print("The draft columns are blank. To produce real drafts:")
            print("  1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment, OR")
            print("     add it to .local/.env (gitignored) as OPENAI_API_KEY=... ")
            print("  2. Re-run:  python scripts/draft_tags.py --n {} --seed {}".format(
                args.n, args.seed))
            print("(No drafted jsonl written; nothing was tagged. Key is never logged.)")
            return

        try:
            drafter = FrontierDrafter(
                provider if args.provider == "auto" else args.provider,
                args.frontier_model,
                args.temperature,
            )
        except SkipFrontier as exc:
            # Key present but library missing, etc. -> empty worksheet + message.
            rows = empty_worksheet_rows(selected)
            write_worksheet(worksheet_path, rows)
            print(f"\n[FRONTIER UNAVAILABLE] {exc}.")
            print(f"Wrote an EMPTY-DRAFT worksheet ({len(rows)} rows) to: {worksheet_path}")
            print("Install the provider client (pip install openai / anthropic) and re-run.")
            return
        backend_label = f"frontier:{drafter.provider}:{drafter.model_name}"
        print(f"Using frontier backend: {backend_label} "
              f"(key loaded from environment/local file; never logged)")

    drafted_items, worksheet_rows, stats = draft_items(drafter, selected, misconceptions)
    write_jsonl(drafted_path, drafted_items)
    write_worksheet(worksheet_path, worksheet_rows)
    report_stats(stats, alloc, backend_label, drafted_path, worksheet_path)

    print("\nNEXT STEP (owner): open the worksheet, fill confirmed(Y/N) /")
    print("corrected_misconception_id / notes for each row, keep the solid ones")
    print("(~40-50) as the GOLD eval set, then run the base-1.7B tagging litmus")
    print("against it.")


if __name__ == "__main__":
    main()
