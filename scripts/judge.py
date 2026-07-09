"""
LLM-as-judge for the Behavior Spec (docs/behavior_spec.md), plus a
judge-vs-human CALIBRATION harness.

The judge scores one generated item on the spec's rubric dimensions
(0/1/2): spec_adherence, distractor_mapping, task_quality. It is used at eval
time for CONCEPTUAL items (genetics is scored programmatically by score_rubric,
no judge needed). Because "the judge says the tuned model wins" is itself
unfalsifiable, the calibration harness measures judge-vs-human agreement on a
small hand-labeled gold set before we trust the judge's verdicts.

Subcommands:
  score  FILE            judge every item in FILE (needs an API key)
  make-gold FILE         export a random calibration sample + a blank human
                         worksheet CSV (data/judge_calibration_*.{jsonl,csv})
  calibrate WORKSHEET    after a human fills the worksheet, report judge-vs-human
                         agreement per dimension

Key handling mirrors draft_tags.py: env OPENAI_API_KEY / ANTHROPIC_API_KEY, then
a gitignored .local/.env or .env. The key is never printed or written anywhere.
Without a key, `score` degrades to a MOCK judge (all-2s) so the plumbing runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIMS = ["spec_adherence", "distractor_mapping", "task_quality"]


# ------------------------------------------------------------------ key + client

def _parse_env_file(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_api_key_into_env() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    for candidate in (REPO / ".local" / ".env", REPO / ".env"):
        if candidate.is_file():
            for k, v in _parse_env_file(candidate).items():
                os.environ.setdefault(k, v)
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


class JudgeClient:
    """Frontier judge (OpenAI/Anthropic) with a JSON-sized token budget, or a
    deterministic MOCK when no key/lib is available (keeps pipelines runnable)."""

    def __init__(self, model_name: str | None = None, force_mock: bool = False):
        self.provider = None if force_mock else load_api_key_into_env()
        self.mock = self.provider is None
        self.model_name = model_name
        if self.mock:
            return
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
            # gpt-4o-mini FAILED calibration (caught 0/15, then 8/15 mis-maps);
            # gpt-4o hit 15/15 = 100% agreement with human. Default to the
            # calibrated model. Override with --model only after re-calibrating.
            self.model_name = model_name or "gpt-4o"
        else:
            import anthropic
            self._client = anthropic.Anthropic()
            self.model_name = model_name or "claude-3-5-sonnet-latest"

    def generate(self, system: str, user: str) -> str:
        if self.mock:
            # All-2s: lets the harness run end-to-end offline. Real verdicts need a key.
            return json.dumps({d: 2 for d in DIMS} | {"answer_correct": True,
                              "per_distractor": {}, "notes": "MOCK (no API key)"})
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model_name, temperature=0, max_tokens=600,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return resp.choices[0].message.content or ""
        resp = self._client.messages.create(
            model=self.model_name, max_tokens=600, temperature=0, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in resp.content if hasattr(b, "text"))


# ------------------------------------------------------------------ judge prompt

JUDGE_SYSTEM = (
    "You are a strict AP Biology assessment reviewer. You grade a generated "
    "multiple-choice item against a fixed rubric and output ONLY a JSON object."
)


def build_judge_prompt(item: dict, misc_defs: dict) -> tuple[str, str]:
    tags = item.get("distractor_tags", {})
    # Give the judge the FULL misconception menu so it can name the best-fit
    # itself, and explicitly list which one each option CLAIMS.
    menu = "\n".join(f'  - {mid}: {d.get("description", "")}'
                     for mid, d in misc_defs.items())
    claim_lines = []
    for L, t in tags.items():
        claim_lines.append(f'  {L} CLAIMS tag "{t.get("misconception_id", "?")}"')
    choices = "\n".join(f"  {L}. {c}" for L, c in item.get("choices", {}).items())
    user = f"""Grade this generated AP Biology item. Do NOT assume the claimed
tags are correct — verify each one adversarially.

Stem: {item.get('stem')}
Choices:
{choices}
Claimed correct answer: {item.get('correct')}

Full misconception menu (id: description):
{menu}

Each WRONG option claims a misconception:
{chr(10).join(claim_lines)}

STEP 1 — For each wrong option, independently decide which misconception id in
the menu its TEXT actually expresses (pick the single best fit). Ignore the
claimed tag while doing this.
STEP 2 — A distractor "maps" ONLY IF the id you chose in step 1 equals the id it
claims. If the text fits a DIFFERENT misconception than claimed, maps = false.

Then score 0/1/2:
- spec_adherence: 2 = valid 4-choice item, one correct, every wrong option tagged.
- distractor_mapping: 2 = every distractor maps (per step 2); 1 = some; 0 = none.
- task_quality: 2 = keyed answer correct and distractors genuinely wrong/plausible.

Output ONLY JSON (per_distractor REQUIRED, one entry per wrong option):
{{"per_distractor":{{"<letter>":{{"actual_id":"<from menu>","claimed_id":"<claimed>","maps":true|false}}}},"spec_adherence":0-2,"distractor_mapping":0-2,"task_quality":0-2,"answer_correct":true|false,"notes":"<short>"}}"""
    return JUDGE_SYSTEM, user


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}


def judge_item(item: dict, client: JudgeClient, misc_defs: dict) -> dict:
    system, user = build_judge_prompt(item, misc_defs)
    verdict = _extract_json(client.generate(system, user))
    out = {}
    for d in DIMS:
        v = verdict.get(d)
        out[d] = int(v) if isinstance(v, (int, float)) and 0 <= v <= 2 else None
    # Prefer the per-distractor verdicts for mapping: derive the score from the
    # explicit maps flags rather than trusting the model's holistic number
    # (which gpt-4o-mini rubber-stamps to 2). This is what made calibration pass.
    pd = verdict.get("per_distractor")
    if isinstance(pd, dict) and pd:
        maps = [bool(v.get("maps")) for v in pd.values() if isinstance(v, dict)]
        if maps:
            out["distractor_mapping"] = 2 if all(maps) else (0 if not any(maps) else 1)
    out["answer_correct"] = verdict.get("answer_correct")
    out["notes"] = verdict.get("notes", "")
    return out


# --------------------------------------------------------------- calibration

def make_gold(items: list[dict], n: int, seed: int):
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    sample = pool[:n]
    gold_path = REPO / "data" / "judge_calibration.jsonl"
    csv_path = REPO / "data" / "judge_calibration_worksheet.csv"
    with open(gold_path, "w", encoding="utf-8") as fh:
        for it in sample:
            fh.write(json.dumps(it) + "\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "topic", "stem"] + [f"human_{d}" for d in DIMS] + ["notes"])
        for it in sample:
            w.writerow([it["id"], it["topic"], it["stem"][:120], "", "", "", ""])
    return gold_path, csv_path, len(sample)


def calibrate(worksheet: str, misc_defs: dict, model_name: str | None,
              force_mock: bool = False):
    """Compare human labels (filled worksheet) to fresh judge scores."""
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(REPO / "data" / "judge_calibration.jsonl", encoding="utf-8")}
    client = JudgeClient(model_name, force_mock=force_mock)
    rows = list(csv.DictReader(open(worksheet, encoding="utf-8")))
    agree = {d: [] for d in DIMS}
    scored = 0
    for row in rows:
        if not row.get(f"human_{DIMS[0]}"):
            continue  # unlabeled row
        item = gold.get(row["id"])
        if not item:
            continue
        jv = judge_item(item, client, misc_defs)
        scored += 1
        for d in DIMS:
            try:
                human = int(row[f"human_{d}"])
            except (ValueError, KeyError):
                continue
            if jv[d] is not None:
                agree[d].append(1 if jv[d] == human else 0)
    print(f"calibration on {scored} human-labeled items"
          f"{' (MOCK judge — no API key)' if client.mock else ''}:")
    for d in DIMS:
        a = agree[d]
        rate = sum(a) / len(a) if a else None
        print(f"  {d:20s} exact-agreement: "
              f"{rate if rate is None else round(rate, 3)}  (n={len(a)})")
    print("target: >~0.8 agreement before trusting judge verdicts in the eval")


# --------------------------------------------------------------------- cli

def main() -> None:
    import gen_spec
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("file"); s.add_argument("--model")
    s.add_argument("--mock", action="store_true", help="force offline mock judge")
    g = sub.add_parser("make-gold"); g.add_argument("file")
    g.add_argument("-n", type=int, default=40); g.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("calibrate"); c.add_argument("worksheet"); c.add_argument("--model")
    c.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    misc_defs = gen_spec.MISC_DEFS

    if args.cmd == "make-gold":
        items = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
        gp, cp, n = make_gold(items, args.n, args.seed)
        print(f"wrote {n} items -> {gp}\nfill human_* columns in -> {cp}")
    elif args.cmd == "calibrate":
        calibrate(args.worksheet, misc_defs, args.model, force_mock=args.mock)
    elif args.cmd == "score":
        client = JudgeClient(args.model, force_mock=args.mock)
        items = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
        agg = {d: [] for d in DIMS}
        for it in items:
            jv = judge_item(it, client, misc_defs)
            for d in DIMS:
                if jv[d] is not None:
                    agg[d].append(jv[d])
        tag = " (MOCK — no API key)" if client.mock else f" ({client.model_name})"
        print(f"judged {len(items)} items{tag}:")
        for d in DIMS:
            print(f"  {d:20s} mean {round(sum(agg[d])/len(agg[d]),3) if agg[d] else None}")


if __name__ == "__main__":
    main()
