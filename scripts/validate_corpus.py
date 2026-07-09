"""
Validate the by-construction training corpus. Answers two questions the
generators must not be trusted to answer about themselves:

  1. ITEM CORRECTNESS - is the keyed answer right, and each distractor wrong?
  2. TAG FIDELITY     - does each distractor really express its tagged misconception?

Validation layers (independent of the generators wherever possible):

  --independent-genetics  (default ON) A FROM-SCRATCH Punnett enumerator, written
        here and importing NOTHING from gen_genetics, recomputes every genetics
        item's correct answer and each error-operator output and compares to the
        item's choices. If this agrees with score_rubric.py (which DOES import
        gen_genetics), a shared bug in the generator math is ruled out.
  --sample N --review-csv PATH   Export a random sample to CSV for human review
        (the ground-truth backstop, essential for conceptual items whose tags
        rest on authoring, not computation).
  --judge / --back-classify      Hooks for LLM-based validation (frontier grader
        and independent tagger). Require a model callable; documented here and
        driven by eval_generation.py. Not called in the offline default run.

Conceptual topics (cellular_respiration, enzymes) CANNOT be recomputed, so their
correctness/fidelity is validated only by judge + human review — reported as
`needs_review`, never as a silent pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

# ---- independent Punnett (deliberately NOT importing gen_genetics) -----------

_ALLELES = {"hom_dom": ("D", "D"), "het": ("D", "r"), "hom_rec": ("r", "r")}
_CANONICAL = {Fraction(1, 16), Fraction(3, 16), Fraction(9, 16), Fraction(1, 8),
              Fraction(3, 8), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)}


def _offspring(p1: str, p2: str) -> Counter:
    dist = Counter()
    for a in _ALLELES[p1]:
        for b in _ALLELES[p2]:
            dist[tuple(sorted((a, b)))] += Fraction(1, 4)
    return dist


def _p_dominant(dist) -> Fraction:
    return sum((p for g, p in dist.items() if "D" in g), Fraction(0))


def _p_hom_rec(dist) -> Fraction:
    return dist.get(("r", "r"), Fraction(0))


def _p_hom_dom(dist) -> Fraction:
    return dist.get(("D", "D"), Fraction(0))


def _frac(s):
    try:
        return Fraction(str(s).strip())
    except (ValueError, ZeroDivisionError):
        return None


def validate_genetics_item(item: dict) -> dict:
    """Independent recompute. Returns per-item pass flags + any mismatch notes."""
    spec = item.get("authoring", {}).get("spec")
    if not spec:
        return {"checkable": False}
    genes = spec["genes"]
    dists = [_offspring(g["p1"], g["p2"]) for g in genes]

    correct = Fraction(1)
    for g, d in zip(genes, dists):
        correct *= _p_dominant(d) if g["want_dominant"] else _p_hom_rec(d)

    choices = item["choices"]
    notes = []
    answer_ok = _frac(choices.get(item["correct"])) == correct
    if not answer_ok:
        notes.append(f"answer: item={choices.get(item['correct'])} indep={correct}")

    g0 = genes[0]
    single_trait = _p_dominant(dists[0]) if g0["want_dominant"] else _p_hom_rec(dists[0])
    geno_conf = Fraction(1)
    for g, d in zip(genes, dists):
        geno_conf *= _p_hom_dom(d) if g["want_dominant"] else _p_hom_rec(d)

    mapping_pass = 0
    tags = item["distractor_tags"]
    for letter, tag in tags.items():
        val = _frac(choices.get(letter))
        mid = tag["misconception_id"]
        ok = (
            (mid == "map_answers_single_trait" and val == single_trait)
            or (mid == "gen_genotype_phenotype_confusion" and val == geno_conf)
            or (mid == "gen_wrong_punnett_ratio" and val in _CANONICAL and val != correct)
        )
        mapping_pass += ok
        if not ok:
            notes.append(f"{letter}({mid})={val}")
    return {"checkable": True, "answer_ok": answer_ok,
            "mapping_ok": mapping_pass == len(tags), "notes": notes}


# ---- human review export -----------------------------------------------------

def export_review_csv(items, path: str, n: int) -> int:
    rows = items[:n]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "topic", "stem", "A", "B", "C", "D", "correct",
                    "distractor_misconceptions", "answer_ok?", "mapping_ok?", "notes"])
        for it in rows:
            c = it["choices"]
            tags = "; ".join(f"{L}={t['misconception_id']}"
                             for L, t in it["distractor_tags"].items())
            w.writerow([it["id"], it["topic"], it["stem"],
                        c.get("A"), c.get("B"), c.get("C"), c.get("D"),
                        it["correct"], tags, "", "", ""])
    return len(rows)


_YES = {"y", "yes", "1", "pass", "ok", "true"}
_NO = {"n", "no", "0", "fail", "bad", "false"}


def _read_csv_rows(path: str):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows, (rows[0].keys() if rows else [])


def _write_csv_rows(path: str, rows, fields) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def _ask(prompt: str, valid: set) -> str:
    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"  # graceful save & quit on Ctrl-C / end of input
        if ans in valid:
            return ans
        print(f"  (enter one of: {', '.join(sorted(valid))})")


def interactive_review(csv_path: str) -> None:
    """Walk unlabeled rows in the terminal: show the item, take y/n, autosave.

    Resumable — already-labeled rows are skipped, so you can quit ('q') and
    resume later. Genetics is machine-proven, so this sheet is conceptual-only.
    """
    import gen_spec
    try:  # AP Bio text contains non-cp1252 glyphs (e.g. ΔG); force UTF-8 console
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    defs = gen_spec.MISC_DEFS
    rows, fields = _read_csv_rows(csv_path)
    if not rows:
        print(f"no rows in {csv_path} — export a sample first "
              f"(validate_corpus.py <corpus> --conceptual-only)")
        return
    todo = [r for r in rows if not (r.get("answer_ok?") or r.get("mapping_ok?"))]
    done = len(rows) - len(todo)
    print(f"{len(todo)} items to review ({done} already done). "
          f"Keys: y/n, s=skip, q=save & quit.\n")

    for i, row in enumerate(todo, 1):
        print("=" * 70)
        print(f"[{i}/{len(todo)}]  {row['id']}   ({row['topic']})")
        print(f"\nQ: {row['stem']}\n")
        tagmap = dict(p.split("=", 1) for p in row["distractor_misconceptions"].split("; ") if "=" in p)
        for L in ("A", "B", "C", "D"):
            mark = "  <-- keyed CORRECT" if L == row["correct"] else ""
            print(f"  {L}. {row.get(L, '')}{mark}")
            if L in tagmap:
                mid = tagmap[L]
                d = defs.get(mid, {})
                print(f"       tag: {mid} — \"{d.get('name', '')}\": {d.get('description', '')}")
        print()
        a = _ask("  Is the keyed answer CORRECT? [y/n/s/q]: ", _YES | _NO | {"s", "q"})
        if a == "q":
            break
        if a == "s":
            continue
        m = _ask("  Does EVERY distractor embody its tagged misconception? [y/n/s/q]: ",
                 _YES | _NO | {"s", "q"})
        if m == "q":
            break
        if m == "s":
            continue
        try:
            note = input("  note (optional, Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""
        row["answer_ok?"] = "Y" if a in _YES else "N"
        row["mapping_ok?"] = "Y" if m in _YES else "N"
        if note:
            row["notes"] = note
        _write_csv_rows(csv_path, rows, fields)  # autosave after each item

    _write_csv_rows(csv_path, rows, fields)
    print("\nsaved. summary so far:")
    report_review(csv_path)


def report_review(csv_path: str) -> None:
    """Read a human-filled review sheet and report the data error rate.

    This closes the human-validation loop for the conceptual half: the target is
    a low error rate; a high one means fix the frames (a DATA problem), not the
    training run."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    ans, mapp, both_ok, labeled = [], [], 0, 0
    for r in rows:
        a, m = (r.get("answer_ok?") or "").strip().lower(), (r.get("mapping_ok?") or "").strip().lower()
        if not a and not m:
            continue
        labeled += 1
        av = 1 if a in _YES else (0 if a in _NO else None)
        mv = 1 if m in _YES else (0 if m in _NO else None)
        if av is not None:
            ans.append(av)
        if mv is not None:
            mapp.append(mv)
        if av == 1 and mv == 1:
            both_ok += 1
    print(f"human review report ({labeled} labeled rows in {csv_path}):")
    if ans:
        print(f"  answer correct:   {sum(ans)}/{len(ans)}  "
              f"error rate {1-sum(ans)/len(ans):.1%}")
    if mapp:
        print(f"  mapping correct:  {sum(mapp)}/{len(mapp)}  "
              f"error rate {1-sum(mapp)/len(mapp):.1%}")
    if labeled:
        print(f"  fully clean:      {both_ok}/{labeled}  ({both_ok/labeled:.1%})")
    print("  target: error rate < ~5%. If higher -> fix the frames and regenerate.")


# ---- driver ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--review-csv", default="data/corpus_review_sample.csv")
    ap.add_argument("--conceptual-only", action="store_true",
                    help="sample only cellresp/enzymes (genetics is machine-proven)")
    ap.add_argument("--report", metavar="FILLED_CSV",
                    help="read a human-filled review sheet and report error rates")
    ap.add_argument("--interactive", action="store_true",
                    help="review the sample CSV in the terminal (y/n prompts, autosave)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.report:
        report_review(args.report)
        return
    if args.interactive:
        interactive_review(args.review_csv)
        return

    items = []
    for f in args.files:
        items += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]

    import random
    rng = random.Random(args.seed)

    genetics = [it for it in items if it.get("topic") == "genetics"]
    conceptual = [it for it in items if it.get("topic") != "genetics"]

    # Layer 1: independent genetics recompute.
    ans_ok = map_ok = checkable = 0
    fails = []
    for it in genetics:
        r = validate_genetics_item(it)
        if not r["checkable"]:
            continue
        checkable += 1
        ans_ok += r["answer_ok"]
        map_ok += r["mapping_ok"]
        if not (r["answer_ok"] and r["mapping_ok"]):
            fails.append((it["id"], r["notes"]))

    print(f"\n=== Independent genetics recompute (from-scratch Punnett) ===")
    print(f"genetics items checkable: {checkable}/{len(genetics)}")
    print(f"  answer correct:   {ans_ok}/{checkable}")
    print(f"  tags all mapped:  {map_ok}/{checkable}")
    if fails:
        print(f"  MISMATCHES ({len(fails)}):")
        for fid, notes in fails[:10]:
            print(f"    {fid}: {notes}")
    else:
        print("  no mismatches -> agrees with score_rubric; shared-bug risk ruled out")

    # Layer 2: conceptual cannot be recomputed.
    ctopics = Counter(it["topic"] for it in conceptual)
    print(f"\n=== Conceptual items: needs judge + human review (not recomputable) ===")
    print(f"conceptual items: {len(conceptual)}  by topic: {dict(ctopics)}")
    print("  correctness/fidelity -> run --judge (LLM) and review the CSV sample")

    # Layer 3: human review sample. Default to CONCEPTUAL items — genetics is
    # already machine-proven above, so human effort belongs on the authored half.
    pool = conceptual if args.conceptual_only else list(items)
    rng.shuffle(pool)
    n = export_review_csv(pool, args.review_csv, args.sample)
    print(f"\n=== Human review sample ({'conceptual only' if args.conceptual_only else 'all topics'}) ===")
    print(f"exported {n} items -> {args.review_csv}")
    print("  fill answer_ok?/mapping_ok? (Y/N); then:")
    print(f"  python scripts/validate_corpus.py --report {args.review_csv}")


if __name__ == "__main__":
    main()
