# QuestionGen — AI-generated AP Biology questions with meaningful wrong answers

A fine-tuned small language model that writes AP Biology multiple-choice
questions where **every wrong answer reflects a specific, well-known student
misconception** — not a random distractor. The model returns a clean, structured
JSON item (question, four options, answer key, and the misconception behind each
wrong choice), which is exactly what an adaptive learning or test-prep tool needs
to diagnose *why* a student got something wrong.

The interesting result: a **1.7B-parameter model** — small enough to run locally
and for free — was trained to do this reliably and performs on par with a much
larger frontier model on the task, at a tiny fraction of the size and cost.

## Highlights
- **Purpose-built dataset** of ~2,000 questions across 5 biology topics, generated
  *by construction* so every distractor provably maps to a named misconception.
- **Evaluation harness built before training**, scoring generations on four
  dimensions (below), with an objective, recomputable ground truth for genetics.
- **Large, measured gains over the base model** on every dimension.
- **A data-iteration case study:** diagnosed a generalization failure and fixed it
  by improving the *data* (adding topic coverage), not the training settings.
- Trained with **QLoRA (Unsloth)** on a single GPU; shipped with a live demo.

## How it's evaluated (four dimensions)
Every generated question is scored 0–2 on:

1. **Spec adherence** — is the output a single, valid, well-formed JSON item
   (four distinct options, exactly one correct answer, every wrong option tagged)?
2. **Distractor mapping** — does each wrong answer genuinely embody the
   misconception it's labeled with? *(the core metric)*
3. **Task quality** — is the biology correct: right answer keyed, distractors
   actually wrong and plausible?
4. **Consistency** — does the model behave reliably across similar prompts?

The eval was written *before* any training. **Genetics is scored objectively** by
recomputing the underlying Punnett cross (no subjective judgment), and conceptual
topics are scored by an LLM judge that was validated against human labels before
being trusted. Models are compared on held-out prompts the model never trained on.

## Results (base vs. fine-tuned, 0–2)
| Dimension | Base Qwen3-1.7B | Fine-tuned |
|---|---|---|
| Spec adherence | 1.80 | **2.00** |
| Distractor mapping | 1.05 | **1.80** |
| Task quality | 1.71 | **2.00** |
| Consistency (fully-correct rate) | 15% | **80%** |

- **Genetics answer correctness (objective): 0/40 → 40/40.** The base model never
  produces a valid, correctly-solved cross; the fine-tuned model does it every time.
- On a **topic held out of training entirely**, fine-tuning still improved
  misconception mapping over the base model — and improving the dataset's topic
  coverage is what made that generalization possible.

## How it works
Good exam distractors are *engineered*, not scraped — real question banks are full
of filler wrong answers that don't correspond to any coherent misconception. So the
training data is generated **by construction**: each misconception is turned into
an "error operator" that produces the exact wrong answer a student holding that
belief would choose, which guarantees the label is correct.

- **Genetics** uses a Punnett-square solver plus error operators, so items are
  verifiable by recomputation.
- **Conceptual topics** (cellular respiration, enzymes, membrane transport,
  evolution) use curated question frames, each pairing three competing misconceptions.

The dataset is then validated (independent recomputation for genetics, human
spot-checks for the rest) and split so evaluation always happens on unseen prompts.

## Tech stack
Python · Qwen3-1.7B · QLoRA / LoRA (Unsloth, PEFT) · Hugging Face Transformers &
Hub · Gradio (demo) · LLM-as-judge + programmatic scoring for evaluation.

## Quickstart
- **Full pipeline** (train → evaluate → demo): `notebooks/run_all_pipeline.ipynb`
  (open in Colab with a GPU runtime).
- **Demo + deploy only** (model already on the Hub): `notebooks/demo_and_deploy.ipynb`
  — runs a base-vs-tuned comparison and deploys a free Hugging Face Space.
- **Read more:** [`docs/summary.md`](docs/summary.md) (one-page overview),
  [`docs/brainlift_generator.md`](docs/brainlift_generator.md) (full write-up),
  [`docs/behavior_spec.md`](docs/behavior_spec.md) (the target behavior definition).

## Repository layout
```
scripts/
  gen_genetics.py                 procedural (Punnett) generator + error operators
  gen_cellresp/enzymes/membrane/evolution.py   conceptual topic generators
  conceptual_engine.py            shared frame-based generator
  build_dataset.py                assembles the corpus + train/eval split
  gen_spec.py                     the generation prompt (shared by train + eval)
  train_gen_sft.py                QLoRA fine-tuning (Unsloth)
  eval_generation.py              base-vs-tuned evaluation harness
  score_rubric.py / judge.py      rubric scorer + validated LLM judge
  validate_corpus.py              independent validation + human-review loop
  app.py                          Gradio inference demo
data/                             the dataset (items, train/val splits, eval sets)
docs/                             overview, full write-up, behavior spec, demo guide
notebooks/                        end-to-end pipeline + demo/deploy notebooks
```

## A note on secrets
API keys load from a git-ignored `.local/.env` (or environment variables) and are
never committed. Use the notebooks' `getpass` / `login()` prompts rather than
pasting keys into cells.

---
*This repository previously hosted an algebra error-classification project; those
earlier files (`docs/spec.md`, `docs/brainlift.md`, legacy `data/`) are retained.*
