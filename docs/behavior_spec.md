# Behavior Spec — Misconception-Tagged AP Bio Item Generator

This is the project gate. It is simultaneously the data-generation rubric, the
eval criterion, and the thesis. Everything downstream serves it. It is written
to be **falsifiable**: a stranger can mark any single model output pass/fail.

## The behavior (one paragraph, pass/fail)

> Given a **topic** and a set of **target misconceptions** (by id, with
> definitions), the model returns **exactly one valid JSON object** — no prose,
> no markdown fences, before or after — describing an AP Biology multiple-choice
> item with: a `stem`, exactly **four** `choices` (A–D), exactly **one**
> `correct` option, and a `distractor_tags` entry for **every** wrong option.
> Each wrong option is tagged with **one of the requested misconceptions** and
> **actually embodies that misconception** (a knowledgeable reader agrees "yes,
> a student with that specific wrong belief would pick this"). For **genetics**
> items the object also includes a machine-checkable `spec` (the cross), and the
> keyed correct answer is **mathematically correct** for that cross.

## The one failure the spec forbids (behavioral check)

A **mis-mapped distractor**: a wrong option whose text does not embody the
misconception it is tagged with (or the answer key is wrong / not unique, or the
output is not a single valid JSON object). This is the exact failure that made
harvested real MCQs unusable (~51% `no_fit`). Our behavioral check is
`score_rubric.distractor_mapping` — and for genetics it is **objective**
(recomputed from the item's own declared cross), not a matter of opinion.

## Why this needs training, not prompting (the litmus test)

A well-prompted **small** base model (Qwen3-1.7B class) cannot do this
*reliably*. It drifts in predictable ways:
- emits prose / markdown fences around the JSON, or invalid JSON;
- writes distractors that are merely "wrong" rather than embodying the *named*
  misconception (tag/text mismatch);
- produces non-distinct or implausible options;
- gets the Punnett arithmetic wrong so the keyed answer is incorrect.

A frontier model can mostly do this when prompted; a 1.7B model cannot do it
*every time, in-character, without drifting*. Reliability is the hard part, and
reliability is what a dataset buys and a prompt cannot guarantee. That is the
defensible win: **behavior from data**, in a tiny local model — not "smarter
than a frontier model."

## Scope discipline

**One behavior, one context.** The behavior is *generate a misconception-tagged
MCQ from a spec*. The three topics (genetics, cellular respiration, enzymes) are
content **variety within that one behavior**, present so the model learns the
skill rather than memorizing one template — not three separate domains.

## How it is graded (maps to the eval rubric)

Per output, base vs. tuned, on held-out scenarios (see `data/eval_scenarios.jsonl`):

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Spec adherence** | violates (bad JSON, missing tags, wrong shape) | partial | valid JSON, 4 choices, 1 correct, all wrong options tagged |
| **Distractor mapping** | distractors don't embody their tags | some do | every distractor embodies its tagged misconception |
| **Task quality** | item is wrong/useless (bad key, duplicate options) | acceptable | genuinely good, answer correct (genetics: recomputed) |
| **Consistency** | behaves differently across similar specs | mostly stable | reliable on every scenario |

**Genetics** dimensions are scored **programmatically** (objective recompute).
**Conceptual** dimensions are scored by an **LLM-as-judge**, itself validated
against a **human-labeled calibration gold set** (judge–human agreement) so the
judge's verdicts are trustworthy. Required output: mean score per dimension,
**base vs. tuned**, on the same held-out scenarios, plus an error-analysis note
on where the tuned model still fails and whether it is a data problem.
