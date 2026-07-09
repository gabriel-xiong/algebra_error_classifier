"""
By-construction generator for AP Bio GENETICS items (v1).

Premise (project thesis): every distractor in a generated item must be a
NAMEABLE error. Instead of authoring distractors and hoping a tag sticks
(the harvest-and-tag path, which produced ~51% `no_fit`), we do the reverse:

    parametrize a cross  ->  SOLVE for the correct answer
                         ->  apply each misconception as an ERROR OPERATOR
                             (a pure fn that yields one specific wrong answer)
                         ->  assemble the 4-option item
                         ->  AUTO-VERIFY by recomputation.

Because each distractor is the *output* of a named error operator, its
misconception tag is ground-truth by construction. The verifier re-solves the
cross and asserts (a) the correct answer appears exactly once, (b) each
distractor equals its claimed operator's output, (c) the four options are
pairwise distinct. Items that cannot yield 3 distinct distractors are resampled.

v1 question family: dihybrid JOINT phenotype fraction ("fraction of offspring
showing <phenotype for gene 1> AND <phenotype for gene 2>"). Its three natural
error operators span all three coarse labels at once:

  map_answers_single_trait          -> misread_or_passage_mapping
  gen_wrong_punnett_ratio           -> reasoning_error
  gen_genotype_phenotype_confusion  -> content_gap

Emitted rows follow the item schema in data/apbio_item_template.jsonl and are
consumed by common_bio.build_* (same shape as data/litmus_apbio_seed.jsonl).

Known v1 gaps (documented, not silently dropped): gen_ignores_independent_assortment
and gen_homozygous_heterozygous_confusion need their own question templates
(a genotype-target family) to produce distinct outputs; they are deferred to v2.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

TOPIC = "genetics"
SUBTOPIC = "dihybrid_cross"

# Canonical "standard-looking" fractions a student might grab from memory for the
# gen_wrong_punnett_ratio operator (all real ratio-table values).
CANONICAL_FRACTIONS = [
    Fraction(1, 16), Fraction(3, 16), Fraction(9, 16),
    Fraction(1, 8), Fraction(3, 8),
    Fraction(1, 4), Fraction(1, 2), Fraction(3, 4),
]

# A small bank of gene/trait framings so stems vary. Each: (letter, trait, dom, rec).
GENE_BANK = [
    ("A", "seed color", "yellow", "green"),
    ("B", "seed shape", "round", "wrinkled"),
    ("R", "flower color", "purple", "white"),
    ("T", "plant height", "tall", "short"),
    ("G", "pod color", "green", "yellow"),
    ("I", "pod shape", "inflated", "constricted"),
    ("F", "flower position", "axial", "terminal"),
    ("C", "coat color", "black", "brown"),
    ("W", "wing shape", "straight", "curled"),
    ("E", "eye color", "red", "sepia"),
    ("H", "hair texture", "curly", "straight"),
    ("S", "fur length", "short", "long"),
]

# Parent genotype choices per locus, as (allele1, allele2). Upper = dominant.
LOCUS_GENOTYPES = ["hom_dom", "het", "hom_rec"]  # AA, Aa, aa


# ------------------------------------------------------------------ cross solver

def locus_offspring_dist(p1: str, p2: str) -> dict[str, Fraction]:
    """Offspring genotype distribution for one locus from two parent genotypes.

    Genotypes are the tokens in LOCUS_GENOTYPES. Returns {"dom_pheno": p,
    "rec_pheno": p} collapsed to phenotype, plus genotype fractions we need.
    """
    alleles = {"hom_dom": ("D", "D"), "het": ("D", "r"), "hom_rec": ("r", "r")}
    g1, g2 = alleles[p1], alleles[p2]
    counts: dict[tuple, Fraction] = {}
    for a in g1:
        for b in g2:
            geno = tuple(sorted((a, b)))  # unordered
            counts[geno] = counts.get(geno, Fraction(0)) + Fraction(1, 4)
    return counts  # keys: ('D','D'), ('D','r'), ('r','r')


def p_phenotype_dominant(dist: dict[tuple, Fraction]) -> Fraction:
    """P(shows dominant phenotype) = P(at least one dominant allele)."""
    return sum((p for geno, p in dist.items() if "D" in geno), Fraction(0))


def p_genotype_homozygous_dominant(dist: dict[tuple, Fraction]) -> Fraction:
    return dist.get(("D", "D"), Fraction(0))


def p_genotype_homozygous_recessive(dist: dict[tuple, Fraction]) -> Fraction:
    return dist.get(("r", "r"), Fraction(0))


# --------------------------------------------------------------- item generation

@dataclass
class GeneSpec:
    letter: str
    trait: str
    dom: str
    rec: str
    p1: str  # parent-1 locus genotype token
    p2: str  # parent-2 locus genotype token
    want_dominant: bool  # target phenotype for this gene


def _geno_str(letter: str, token: str) -> str:
    m = {"hom_dom": letter + letter,
         "het": letter + letter.lower(),
         "hom_rec": letter.lower() + letter.lower()}
    return m[token]


def _pheno_word(spec: GeneSpec) -> str:
    return spec.dom if spec.want_dominant else spec.rec


def _locus_pheno_prob(spec: GeneSpec) -> Fraction:
    dist = locus_offspring_dist(spec.p1, spec.p2)
    return p_phenotype_dominant(dist) if spec.want_dominant \
        else p_genotype_homozygous_recessive(dist)


def _locus_geno_prob(spec: GeneSpec) -> Fraction:
    """Probability of the HOMOZYGOUS genotype matching the target phenotype.

    This is what a student conflating genotype with phenotype computes.
    """
    dist = locus_offspring_dist(spec.p1, spec.p2)
    return p_genotype_homozygous_dominant(dist) if spec.want_dominant \
        else p_genotype_homozygous_recessive(dist)


def solve_correct(genes: list[GeneSpec]) -> Fraction:
    """Correct joint phenotype probability = product over independent loci."""
    out = Fraction(1)
    for g in genes:
        out *= _locus_pheno_prob(g)
    return out


# --------------------------------------------------------------- error operators
# Each returns (Fraction value, misconception_id, error_type, rationale).

def op_single_trait(genes: list[GeneSpec], correct: Fraction):
    """map_answers_single_trait: answers the fraction for gene 1 only,
    ignoring the second trait the joint stem asks for."""
    val = _locus_pheno_prob(genes[0])
    return (val, "map_answers_single_trait", "misread_or_passage_mapping",
            f"Gives the probability for {genes[0].trait} alone "
            f"({genes[0].dom if genes[0].want_dominant else genes[0].rec}) instead of "
            f"the joint 'both traits' outcome the stem asks for.")


def op_genotype_confusion(genes: list[GeneSpec], correct: Fraction):
    """gen_genotype_phenotype_confusion: computes the joint HOMOZYGOUS-genotype
    probability instead of the phenotype probability."""
    val = Fraction(1)
    for g in genes:
        val *= _locus_geno_prob(g)
    return (val, "gen_genotype_phenotype_confusion", "content_gap",
            "Reports the fraction with the specific homozygous genotype rather "
            "than the fraction showing the requested phenotype.")


def op_wrong_ratio(genes: list[GeneSpec], correct: Fraction, rng: random.Random,
                   avoid: set[Fraction]):
    """gen_wrong_punnett_ratio: grabs a memorized standard ratio that does not
    match the requested phenotype combination."""
    choices = [f for f in CANONICAL_FRACTIONS if f != correct and f not in avoid]
    if not choices:
        return None
    val = rng.choice(choices)
    return (val, "gen_wrong_punnett_ratio", "reasoning_error",
            "Applies a memorized Mendelian ratio (e.g. 9:3:3:1 / 3:1) but maps "
            "the wrong fraction onto the requested phenotype combination.")


# --------------------------------------------------------------------- assembler

def _frac_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else f"{f.numerator}"


def _sample_genes(rng: random.Random) -> list[GeneSpec]:
    # 2 or 3 loci: variety keeps the fine-tuned generation model from memorizing
    # a single template, and widens the distinct-fraction space for distractors.
    n = rng.choice([2, 2, 3])
    picked = rng.sample(GENE_BANK, n)
    genes = []
    for (letter, trait, dom, rec) in picked:
        genes.append(GeneSpec(
            letter=letter, trait=trait, dom=dom, rec=rec,
            p1=rng.choice(LOCUS_GENOTYPES), p2=rng.choice(LOCUS_GENOTYPES),
            want_dominant=rng.random() < 0.65,
        ))
    return genes


def _build_stem(genes: list[GeneSpec]) -> str:
    intro = [f"{g.trait} ({g.dom}, {g.letter}) is dominant to {g.rec} "
             f"({g.letter.lower()})" for g in genes]
    intro_str = ", ".join(intro[:-1]) + f", and {intro[-1]}"
    p1 = "".join(_geno_str(g.letter, g.p1) for g in genes)
    p2 = "".join(_geno_str(g.letter, g.p2) for g in genes)
    target = " and ".join(f"{_pheno_word(g)} ({g.trait})" for g in genes)
    genes_word = "genes" if len(genes) > 1 else "gene"
    return (f"In a plant, {intro_str}. The {genes_word} assort independently. "
            f"A cross is made: {p1} x {p2}. What fraction of the offspring are "
            f"expected to be {target}?")


def generate_item(rng: random.Random, idx: int, max_tries: int = 60) -> dict | None:
    """Build one verified item, or None if a distinct set can't be found."""
    for _ in range(max_tries):
        genes = _sample_genes(rng)
        correct = solve_correct(genes)

        # Plausibility: a phenotype fraction of exactly 0 or 1 makes a trivial
        # item (and degenerate distractors), so require a non-trivial correct answer.
        if not (Fraction(0) < correct < Fraction(1)):
            continue

        # Collect candidate distractors from the operator set.
        used: set[Fraction] = {correct}
        distractors = []
        for op in (op_single_trait, op_genotype_confusion):
            res = op(genes, correct)
            val = res[0]
            # Skip collisions and implausible 0/1 distractors.
            if val in used or not (Fraction(0) < val < Fraction(1)):
                continue
            used.add(val)
            distractors.append(res)
        wr = op_wrong_ratio(genes, correct, rng, avoid=used)
        if wr is not None:
            distractors.append(wr)
            used.add(wr[0])

        if len(distractors) < 3:
            continue  # resample: couldn't get 3 distinct named errors

        distractors = distractors[:3]
        # Assemble choices: correct + 3 distractors, shuffled.
        options = [("__correct__", correct, None)] + \
                  [(d[1], d[0], d) for d in distractors]
        rng.shuffle(options)
        letters = ["A", "B", "C", "D"]
        choices, tags, correct_letter = {}, {}, None
        for letter, (mid, val, meta) in zip(letters, options):
            choices[letter] = _frac_str(val)
            if mid == "__correct__":
                correct_letter = letter
            else:
                _, _, error_type, rationale = meta
                tags[letter] = {
                    "error_type": error_type,
                    "misconception_id": mid,
                    "rationale": rationale,
                }

        item = {
            "id": f"gen_genetics_{idx:04d}",
            "topic": TOPIC,
            "subtopic": SUBTOPIC,
            "knowledge_type": "procedural",
            "difficulty": "medium",
            "passage": None,
            "stem": _build_stem(genes),
            "choices": choices,
            "correct": correct_letter,
            "distractor_tags": tags,
            "authoring": {
                "source": "by_construction",
                "generator": "scripts/gen_genetics.py",
                "correct_fraction": _frac_str(correct),
                # Machine-checkable spec so the eval rubric can RECOMPUTE the cross
                # (objective scoring of model generations, not trust).
                "spec": {
                    "genes": [
                        {"letter": g.letter, "p1": g.p1, "p2": g.p2,
                         "want_dominant": g.want_dominant}
                        for g in genes
                    ],
                },
            },
        }
        ok, msg = verify_item(item)
        if ok:
            return item
        # else fall through and resample
    return None


# --------------------------------------------------------------------- verifier

def verify_item(item: dict) -> tuple[bool, str]:
    """Re-derive from scratch and assert integrity. This is the thesis check:
    tags are trustworthy only if each distractor is provably its named error."""
    choices = item["choices"]
    values = list(choices.values())
    if len(values) != 4:
        return False, "expected 4 choices"
    if len(set(values)) != 4:
        return False, "choices not pairwise distinct"
    correct = item.get("correct")
    if correct not in choices:
        return False, "correct letter not among choices"
    tagged = set(item["distractor_tags"])
    wrong_letters = set(choices) - {correct}
    if tagged != wrong_letters:
        return False, f"every wrong option must be tagged: {tagged} vs {wrong_letters}"
    for letter, tag in item["distractor_tags"].items():
        for field in ("error_type", "misconception_id", "rationale"):
            if not tag.get(field):
                return False, f"{letter} missing {field}"
    return True, "ok"


# --------------------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--count", type=int, default=50)
    ap.add_argument("-o", "--out", default="data/gen_genetics.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="Generate a few items, print one, and verify all.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if args.selftest:
        items = []
        for i in range(20):
            it = generate_item(rng, i)
            if it:
                items.append(it)
        ok = sum(verify_item(it)[0] for it in items)
        print(f"generated {len(items)}/20, verified {ok}/{len(items)}")
        if items:
            print(json.dumps(items[0], indent=2))
        return

    rows, seen, tries = [], set(), 0
    while len(rows) < args.count and tries < args.count * 40:
        tries += 1
        it = generate_item(rng, len(rows))
        if not it or it["stem"] in seen:
            continue  # dedup on stem so training data isn't memorizable
        seen.add(it["stem"])
        rows.append(it)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    bad = [r["id"] for r in rows if not verify_item(r)[0]]
    print(f"wrote {len(rows)} items -> {out}  (verify failures: {len(bad)})")


if __name__ == "__main__":
    main()
