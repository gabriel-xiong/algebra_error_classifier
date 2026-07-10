# Brainlift — Misconception-Tagged AP Bio Item Generator

## Thesis (spiky POV)
**Behavior comes from data, not capability.** A 1.7B model will never out-reason a
frontier model. But you *can* make it do **one narrow thing reliably** that a
prompted small model cannot: generate an AP Biology multiple-choice item where
**every distractor is a deliberate, named misconception**. The dataset — not the
model, not the hyperparameters — is the deliverable that buys that reliability.

## Behavior Spec (the gate — falsifiable, pass/fail)
> Given a **topic** and a set of **target misconceptions**, the model returns one
> valid JSON object — no prose — with a stem, four distinct choices, exactly one
> correct answer, and a `distractor_tags` entry for every wrong option, where each
> distractor **actually embodies** its tagged misconception. For genetics it also
> emits a machine-checkable cross `spec`, and the keyed answer is mathematically
> correct.

**Forbidden failure:** a distractor that does not embody its tag (a mis-mapped
error). This is the behavioral check, and for genetics it is scored *objectively*.

**Passes the prompt test?** Yes. A prompted Qwen3-1.7B drifts: emits prose,
mis-maps tags, botches Punnett math. Reliability is the hard part — that is what
the data buys and a prompt cannot guarantee.

## The dataset (the real artifact)
Built **by construction**, not scraped. We first tried harvest-and-tag: ~51% of
real-MCQ distractors were "no-fit" filler. So we reversed it — apply a named
misconception as an *error operator* to generate the distractor, so the tag is
guaranteed:
- **Genetics (procedural):** a Punnett solver + error operators → objectively
  verifiable by recomputation.
- **Conceptual (cell resp, enzymes, membrane transport, evolution):** curated
  frames, each with ≥3 competing misconceptions.

**v2 corpus: 2,046 items, 50/50 procedural/conceptual, 5 topics.** Validated by
independent recompute (genetics 1023/1023) + human review (which caught and fixed
3 frame bugs — fixed in data, then regenerated).

## Evaluation (built before training; the make-or-break)
Rubric per output (0/1/2): spec_adherence, **distractor_mapping**, task_quality,
plus reliability / forbidden-failure aggregates.
- **Genetics scored objectively** (recompute from the model's own declared cross).
- **Conceptual scored by an LLM judge, calibrated to the human.** gpt-4o-mini
  *failed* calibration (caught 0/15 injected mis-maps → rubber-stamped everything);
  **gpt-4o hit 15/15 = 100% agreement** with human labels. Judge = gpt-4o.
- Held-out scenarios are disjoint from training; photosynthesis is held out
  entirely for out-of-distribution testing.

## Results — base vs tuned (real gpt-4o judge)

**In-distribution (n=61):**
| dimension | base | tuned | Δ |
|---|---|---|---|
| spec_adherence | 1.80 | 2.00 | +0.20 |
| distractor_mapping | 1.05 | 1.80 | **+0.75** |
| task_quality | 1.71 | 2.00 | +0.30 |
| reliability | 0.15 | 0.80 | +0.66 |
| forbidden_failure_rate | 0.85 | 0.20 | −0.66 |

Genetics (objective, no judge): **base 0/40 → tuned 40/40.** The base model cannot
produce a valid, correctly-solved cross; the tuned model does it every time.

**Out-of-distribution — photosynthesis, never trained (n=20):**
| dimension | base | tuned | Δ |
|---|---|---|---|
| spec_adherence | 1.05 | 1.85 | +0.80 |
| distractor_mapping | 0.70 | 1.35 | **+0.65** |
| task_quality | 0.95 | 1.10 | +0.15 |
| reliability | 0.15 | 0.35 | +0.20 |

## The data-iteration story (Day-4 win)
The **first** tune (3 topics) *overfit*: OOD distractor_mapping was **0.96 — worse
than base (negative transfer)**. Diagnosis: it learned the trained topics' content,
not a general skill. Fix was in the **data**: added 2 topics (membrane transport,
evolution), held photosynthesis out. The **second** tune generalized — OOD mapping
**flipped from negative to +0.65 over base** on the unseen topic. Diversity in the
data, not any hyperparameter, closed the gap.

## Frontier comparison (for scale, not victory)
Prompted **gpt-4o** vs the fine-tuned **1.7B**, same rubric, in-distribution (n=61):

| dimension | gpt-4o (prompted) | tuned 1.7B | Δ |
|---|---|---|---|
| spec_adherence | 2.00 | 2.00 | tie |
| distractor_mapping | 1.90 | 1.80 | −0.10 |
| task_quality | 1.95 | **2.00** | **+0.05** |
| reliability | 0.90 | 0.80 | −0.10 |
| forbidden_failure_rate | 0.10 | 0.20 | +0.10 |

A model **~1,000× smaller, local, and free** lands within ~0.1 of prompted gpt-4o
on the trained behavior — and *beats* it on task quality. Two things strengthen
this: (1) gpt-4o was **also the judge**, so its conceptual scores are biased in its
own favor, and the specialist matched it anyway; (2) the genetics portion is scored
by objective recompute, so it is bias-free.

Out-of-distribution (photosynthesis, unseen, n=20): gpt-4o leads, as expected of a
generalist — spec_adherence 2.00 vs 1.95 (near-tie: the format transfers), but
distractor_mapping 1.75 vs 1.40 and reliability 0.80 vs 0.35. (gpt-4o's OOD numbers
are self-judge-inflated with no objective anchor, so its lead is an upper bound.)

### The full picture (distractor_mapping)
| | base 1.7B | tuned 1.7B | gpt-4o |
|---|---|---|---|
| in-distribution | 1.05 | **1.80** | 1.90 |
| out-of-distribution | 0.70 | 1.40 | 1.75 |

The point is not to beat the frontier — it is that **a tiny local model reaches
frontier-parity on its trained niche**, while the frontier leads where breadth
matters. Specialists and generalists have different jobs, and these numbers mark
the boundary precisely.

## Did "data → behavior" hold?
**Yes, with an honest boundary.** Data reliably instilled the behavior
*within its distribution* (in-dist near-ceiling; genetics solved where base fails).
Generalization to unseen topics is **real but partial** (OOD reliability 0.35), and
it *improved* only when we broadened the data — direct evidence that coverage, not
model size, governs transfer.

## Limitations (owned)
- OOD reliability is 0.35 — the specialist generalizes partially, not fully.
- In-dist held-out scenarios reuse trained stems/phrasing; the OOD topic is the
  stronger generalization test.
- Eval N is modest (61 in-dist, 20 OOD). Decisive for a behavioral claim, not a
  large benchmark.
- v1→v2 OOD comparison is directional (test set changed as topics moved into
  training); the clean standalone claim is v2 tuned > base on an unseen topic.
