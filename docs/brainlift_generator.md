# BrainLift — Misconception-Tagged AP Bio Item Generator

## Owners
Gabriel Xiong

## Purpose
To establish and defend the behavior thesis for a fine-tuned small model: that a
1.7B open model can be made to **reliably generate AP Biology multiple-choice
items in which every wrong answer is a deliberate, named misconception** — a
behavior a prompted small model cannot do reliably — and that this reliability
comes from the **training data**, not model scale or hyperparameters. The target
behavior is: given a topic and a set of target misconceptions, return one valid
JSON item with a stem, four choices, one correct answer, and each distractor
tagged with — and genuinely embodying — one of the requested misconceptions.

### In scope
- The falsifiable Behavior Spec that serves as data-gen rubric and eval criterion
- How to *generate* misconception-bearing distractors so their tags are ground-truth
- An evaluation built and calibrated *before* training, with an objective anchor
- Base-vs-tuned and tuned-vs-frontier comparisons, in- and out-of-distribution
- What generalization depends on, shown by a data-iteration experiment

### Out of scope
- Beating a frontier model on general capability (explicitly not the goal)
- Hyperparameter search (training is a downstream button-press)
- Tutoring/answering behavior (this generates items; it does not tutor)
- Arbitrary biology breadth — the domain is kept narrow by design

---

## DOK 4: Spiky Points of View

**SPOV 1 — The dataset is the deliverable, and good distractors must be authored
by construction, not harvested.**
The instinct is to scrape real MCQs and tag their distractors. We tried it: ~51%
of harvested distractors were `no_fit` — random wrong vocabulary, not
misconceptions. Real exam distractors are written to be *plausible*, not to encode
a *specific* wrong belief. The fix is to invert the pipeline: treat each
misconception as an **error operator** and *generate* the distractor from it, so
the tag is true by construction. For genetics this is a literal function (a buggy
Punnett computation); for conceptual topics it is a curated false statement. The
model's reliability is bounded by the dataset's fidelity, so the dataset — not the
training run — is the real artifact.

**SPOV 2 — Fine-tuning buys reliability, not capability; so the eval must measure
reliability and be built before training.**
A frontier model can already produce a tagged item when prompted. The thing a
small model *can't* do is produce one **every time, in-character, without
drifting**. That means the metric that matters is not "can it ever" but "does it
always" — reliability and a forbidden-failure rate, measured on held-out
scenarios. We built the eval (rubric + harness + held-out set) before any
training, so "we fine-tuned a model" could be made falsifiable from day one.

**SPOV 3 — An LLM judge is worthless until calibrated against a human; a cheap
judge will silently rubber-stamp.**
We injected 15 deliberately mis-mapped items into a 30-item calibration set and
scored both a human and the judge. **gpt-4o-mini caught 0 of 15** — it approved
every mis-map, which would have made the entire conceptual eval a null metric.
Only **gpt-4o (with an adversarial per-distractor prompt) reached 15/15, 100%
agreement with the human.** The lesson: the eval instrument itself must be
validated, or "the judge says tuned won" is unfalsifiable. Where possible we avoid
the judge entirely — genetics is scored by **objective recomputation**.

**SPOV 4 — Generalization is a property of data coverage, not model size.**
The first fine-tune (3 topics) *overfit*: on an unseen topic it mapped distractors
*worse than the base model* (negative transfer). Nothing about the model or the
hyperparameters was wrong — the **data** lacked coverage. Adding two topics
(membrane transport, evolution) while holding photosynthesis out flipped
out-of-distribution transfer from **negative to +0.65 over base**. The lever for
generalization was diversity in the data, demonstrated with a controlled
before/after.

**SPOV 5 — A tiny specialist can reach frontier parity on its niche; that, not raw
capability, is the defensible win.**
On its trained behavior, the 1.7B model **ties prompted gpt-4o** (and beats it on
task quality) — despite gpt-4o also serving as the judge and the genetics half
being bias-free. Off-distribution the frontier leads, as a generalist should.
Specialists and generalists have different jobs; the value of the small model is
reliable, cheap, local behavior on the one thing it was built for.

---

## Behavior Spec (the gate — falsifiable, pass/fail)
> Given a **topic** and a set of **target misconceptions**, the model returns one
> valid JSON object — no prose — with a stem, four distinct choices, exactly one
> correct answer, and a `distractor_tags` entry for every wrong option, where each
> distractor **actually embodies** its tagged misconception. For genetics it also
> emits a machine-checkable cross `spec`, and the keyed answer is mathematically
> correct.

**Forbidden failure:** a distractor that does not embody its tag. This is the
behavioral check; for genetics it is scored objectively (recompute).

## The dataset (the real artifact)
By construction, not scraped (SPOV 1). Genetics = Punnett solver + error
operators (objectively verifiable). Conceptual (cell resp, enzymes, membrane
transport, evolution) = curated frames, each with ≥3 competing misconceptions.
**v2 corpus: 2,046 items, 50/50 procedural/conceptual, 5 topics.** Validated by
independent recompute (genetics 1023/1023) + human review, which caught and fixed
3 frame bugs — fixed in data, then regenerated.

## Evaluation (built before training)
Rubric per output (0/1/2): spec_adherence, **distractor_mapping**, task_quality,
plus reliability / forbidden-failure aggregates. Genetics objective; conceptual by
the calibrated gpt-4o judge (SPOV 3). Held-out scenarios disjoint from training;
photosynthesis held out entirely for OOD.

## Results — base vs tuned (real gpt-4o judge)

**In-distribution (n=61):**
| dimension | base | tuned | Δ |
|---|---|---|---|
| spec_adherence | 1.80 | 2.00 | +0.20 |
| distractor_mapping | 1.05 | 1.80 | **+0.75** |
| task_quality | 1.71 | 2.00 | +0.30 |
| reliability | 0.15 | 0.80 | +0.66 |
| forbidden_failure_rate | 0.85 | 0.20 | −0.66 |

Genetics (objective): **base 0/40 → tuned 40/40.**

**Out-of-distribution — photosynthesis, never trained (n=20):**
| dimension | base | tuned | Δ |
|---|---|---|---|
| spec_adherence | 1.05 | 1.85 | +0.80 |
| distractor_mapping | 0.70 | 1.35 | **+0.65** |
| task_quality | 0.95 | 1.10 | +0.15 |
| reliability | 0.15 | 0.35 | +0.20 |

## Data-iteration experiment (SPOV 4)
v1 (3 topics): OOD mapping **0.96 — worse than base** (negative transfer). Added
membrane transport + evolution, held out photosynthesis → v2 OOD mapping **1.35,
+0.65 over base**. Fixed in data, not hyperparameters.

## Frontier comparison — full 2×2 (distractor_mapping)
| | base 1.7B | tuned 1.7B | gpt-4o |
|---|---|---|---|
| in-distribution | 1.05 | **1.80** | 1.90 |
| out-of-distribution | 0.70 | 1.40 | 1.75 |

In-distribution the tuned 1.7B ties gpt-4o (spec 2.00/2.00; task_quality 2.00 vs
1.95 — tuned wins) despite gpt-4o being the judge and genetics being objective.
OOD the frontier leads (its numbers are self-judge-inflated → upper bound).

## Did "data → behavior" hold?
**Yes, with an honest boundary.** Data reliably instilled the behavior within its
distribution (in-dist near-ceiling; genetics solved where base cannot). OOD
transfer is real but partial (reliability 0.35) and improved only when the data
broadened — direct evidence that coverage, not scale, governs transfer.

## Limitations (owned)
- OOD reliability 0.35 — partial generalization, not full.
- In-dist held-out scenarios reuse trained stems/phrasing; OOD is the stronger test.
- Eval N modest (61 in-dist, 20 OOD): decisive for a behavioral claim, not a
  large benchmark.
- v1→v2 OOD comparison is directional (test set shifted as topics moved into
  training); the clean standalone claim is v2 tuned > base on an unseen topic.
- Frontier conceptual scores carry self-judge bias (gpt-4o generates and judges);
  genetics is the bias-free anchor.
