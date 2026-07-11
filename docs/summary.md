# Project Summary — Misconception-Tagged AP Bio Item Generator

## What this project does
Fine-tunes a small open model (**Qwen3-1.7B**, QLoRA) to do **one narrow thing
reliably**: given a topic and a set of target misconceptions, generate an AP
Biology multiple-choice question where **every wrong answer is a deliberate,
named misconception** — emitted as clean JSON with each distractor tagged. The
thesis: *behavior comes from data, not model size.* A prompted 1.7B cannot do
this reliably; a dataset can make it.

## How we built it
- **Dataset by construction, not scraping.** Harvesting real MCQs gave ~51%
  "no-fit" filler distractors, so we reversed it: apply each misconception as an
  *error operator* to generate the distractor, so the tag is guaranteed. Genetics
  uses a Punnett solver (objectively verifiable); conceptual topics use curated
  frames with ≥3 competing misconceptions. **Final: 2,046 items, 5 topics,
  50/50 procedural/conceptual.**
- **Eval built before training,** with a rubric (spec adherence, distractor
  mapping, task quality, reliability). Genetics scored **objectively by
  recomputation**; conceptual scored by an **LLM judge calibrated to human
  labels** (gpt-4o-mini failed — caught 0/15 injected errors; gpt-4o hit 15/15,
  so gpt-4o is the judge).
- **Data iteration:** v1 (3 topics) overfit — negative transfer to unseen topics.
  Added 2 topics → v2 generalized. Fixed in *data*, not hyperparameters.

## Results (distractor mapping, 0–2)
|                      | base 1.7B | tuned 1.7B | gpt-4o |
|----------------------|-----------|------------|--------|
| in-distribution      | 1.05      | **1.80**   | 1.90   |
| out-of-distribution  | 0.70      | 1.40       | 1.75   |

- **In-distribution:** tuned ties prompted gpt-4o (and beats it on task quality,
  2.00 vs 1.95) — despite gpt-4o also being the judge. Genetics: base **0/40 →
  tuned 40/40** (objective recompute).
- **Out-of-distribution:** frontier leads, as a generalist should; the specialist
  transfers partially (format fully, content mapping partly).
- **Data-iteration win:** OOD mapping flipped from **negative transfer (v1) to
  +0.65 over base (v2)** by adding topic diversity.

## Takeaway
A ~1,000× smaller, local, free model reaches **frontier-parity on its trained
niche**, and the frontier leads only where breadth matters. Behavior came from the
data — and the boundary of that behavior is exactly the boundary of the data's
coverage.

## Repo map (key files)
- `docs/behavior_spec.md` — the falsifiable behavior spec (the gate)
- `docs/brainlift_generator.md` — full thesis, results, limitations
- `scripts/gen_*.py` + `build_dataset.py` — by-construction dataset generators
- `scripts/score_rubric.py`, `judge.py`, `eval_generation.py` — the eval harness
- `scripts/validate_corpus.py` — independent validation + human-review loop
- `scripts/train_gen_sft.py` + `notebooks/train_gen_colab.ipynb` — QLoRA training
- `data/gen_train.jsonl`, `gen_sft_*.jsonl`, `eval_scenarios*.jsonl` — the dataset
