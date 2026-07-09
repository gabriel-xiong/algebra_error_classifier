# Algebra Error-Type Classifier: Build Spec

A one-week project: fine-tune a small open model to classify why a student got a
multi-step linear equation wrong, into a fixed error taxonomy, with calibrated
abstention. The dataset is the deliverable; training is a downstream button-press.

This spec is the handoff for implementation. It states the thesis, the exact
behavior to build, the taxonomy, the data pipeline, the eval, and the day-by-day
arc, and it points to the four scripts included alongside it.

---

## 1. Thesis (why this project exists)

The point is not to beat a frontier model on capability. It is to make a small
model reliably do one thing that a well-prompted small base model cannot do
consistently: apply a fixed error taxonomy the same way every time, and abstain
when the signal is too thin. Three positions drive every design choice:

- **The hard part is the taxonomy, not the model.** An ungrounded label set
  produces inconsistent labels regardless of model size, so the taxonomy is
  grounded in the algebra-misconception literature (Booth et al. 2014).
- **Granularity is a tradeoff, not a maximization.** More labels reduce accuracy
  and inter-rater agreement (Liu et al. 2023: GPT-4 fell from 91.9% at 4 candidate
  labels to 39.8% at 100). The taxonomy is deliberately coarse: 7 labels.
- **The observable signal is thin, so there is a hard accuracy ceiling.** A
  careless slip and a stable misconception can produce the identical wrong answer
  (Norman 1981, Reason 1990). The design target is calibrated abstention, not
  maximum accuracy. Prior work on this exact task (McNichols, Zhang, Lan 2023)
  reaches mid-80s accuracy with shown work and still confuses two specific error
  types 41% of the time, because intent is not recoverable from the trace.

The full research grounding is in the BrainLift document. This spec is the build.

---

## 2. Behavior Spec (the falsifiable gate)

> Given a multi-step linear equation, the correct answer, and a student's incorrect
> answer (with shown work when available), the model returns exactly one label from
> the fixed 7-label taxonomy and a confidence value, with no prose before or after.
> When the visible information does not support a confident single label, it returns
> `abstain`. At inference, an example is treated as abstained when the model's
> calibrated confidence for its top label is below a set threshold.

A stranger can mark any output pass/fail against this. It is simultaneously the
data-generation rubric, the eval criterion, and the thesis.

---

## 3. Scope

- **Domain:** multi-step linear equations in one variable. This is wider than bare
  `ax + b = c` on purpose, so all seven labels actually fire. The generator must
  produce equations with parentheses (for distribution errors), negatives and
  subtraction (for sign errors), variables on both sides (for balance errors), and
  multiple terms (for conjoining/variable errors).
- **Out of scope:** quadratics, factoring, systems, word problems. Broadening past
  linear equations breaks the "one target, one context" rule and makes the taxonomy
  mushy.

---

## 4. Taxonomy (7 labels, fixed)

Grounded on Booth et al. (2014), trimmed to what is observable in linear-equation
work, kept coarse.

| Label | Definition |
|---|---|
| `equality_balance_error` | Operated on one side only, dropped the equals sign, or did not keep both sides balanced. |
| `negative_sign_error` | Dropped or mishandled a negative, or moved a term across the equals sign without flipping its sign. |
| `variable_error` | Combined unlike terms, conjoined a constant and variable (2 + 3x written as 5x), or combined variable terms with the wrong sign. |
| `operation_inverse_error` | Used the wrong inverse operation (divided when they should have multiplied). |
| `distribution_property_error` | Misapplied distribution or order of operations (distributed to only one term in parentheses). |
| `arithmetic_slip` | Correct procedure and correct operations, but a pure computation mistake (6 + 8 = 13). |
| `abstain` | The visible information does not support a confident single label: no work shown with a multiply-reachable answer, or shown work that fits two distinct error types equally. |

Do not add sub-categories before inter-rater agreement is measured (see 8.3).

---

## 5. Output schema

Every training example and every inference output uses the same shape. Decide
between two options before generating data, because it changes every example:

- **Option A (recommended for week one):** the model emits only the label token.
  Confidence is derived from the label-token probability, then calibrated with
  temperature scaling (see 7). Simplest to train and to grade.
- **Option B:** the model emits `label|confidence_bucket` where bucket is
  `high|medium|low`. Easier to read, but the bucket boundaries must themselves be
  learned and calibrated, which is more work.

Default to Option A. Abstention is then a threshold on calibrated confidence, not a
class the model is trained to emit for its own sake. Note: `abstain` still appears
as a training label for the genuinely ambiguous cases, but most abstention at
inference comes from low confidence on the six substantive labels.

---

## 6. Data pipeline (this is 80% of the outcome)

Two complementary strategies. Use forward injection as the backbone; optionally add
teacher distillation for realism.

### 6.1 Forward error injection (`generate_dataset.py`, included)

Generate a correct equation and its solution, then inject a specific, documented
error to produce the wrong work and wrong answer. The label is known because you
chose the error, and class balance is fully controlled.

- Implemented injectors: distribution, balance, variable (conjoining and
  wrong-sign combine), negative-sign, operation/inverse, arithmetic slip.
- Abstain examples are made by stripping the shown work from an injected error so
  the wrong answer is present but the process is not.
- Every injector is tied to a documented misconception, not an arbitrary
  corruption. That is the mitigation for the main weakness below.

Run (one deduplicated pool split into disjoint train/val/test — per-seed
generation leaks ~69% of test problems into train at these volumes):
```
python make_splits.py --train-n 6000 --val-n 800 --test-n 1000 --seed 0 --out-dir ../data
```
This writes `train.jsonl` (QLoRA), `val.jsonl` (calibration only), and
`test_holdout.jsonl` (primary base-vs-tuned eval), guaranteed to share no problems.

**Known weakness, state it in the BrainLift:** injected errors may not match the
real distribution of student errors, so the model can learn "what injected errors
look like." Mitigations: (1) ground injectors in Booth's misconceptions (done),
(2) optionally pass examples through a teacher for surface variety (6.2), (3) keep
the human-labeled `linear_equation_errors_testset.jsonl` as a reality check the
model never trains on.

### 6.2 Teacher distillation (optional, `teacher_distill.py`, TODO scaffold)

The assignment covers AI costs and recommends distilling from a frontier teacher.
Two uses:
- **Realism pass:** feed an injected example to a teacher and ask it to rewrite the
  student work in a more natural, varied way without changing the error or answer.
- **Hard cases:** ask a teacher to generate plausible student errors for a given
  equation, then verify each against the taxonomy with a second teacher call before
  keeping it. Verification is required; unverified teacher labels are noise.

The craft is in the generation prompt and the quality gate, not raw volume. Filter
hard: drop any example where the stated label does not match the injected/derived
error, where the answer equals the correct answer, or where the work is malformed.

### 6.3 Data format

JSONL, one object per line, matching the included test set:
```json
{"id": "...", "problem": "2(x + 3) = 14", "correct_answer": "x = 4",
 "student_answer": "x = 5.5", "student_work": "2x + 3 = 14; 2x = 11; x = 5.5",
 "label": "distribution_property_error"}
```
`student_work` is `null` for final-answer-only examples.

---

## 7. Training

- **Base model:** Qwen/Qwen3-1.7B (default; post-trained, chat via `enable_thinking=False`). Alternates: Qwen3-0.6B/4B, Llama
  3.2 1B/3B. Start from the Instruct variant.
- **Method:** QLoRA via Unsloth (roughly 2x faster, about 70% less VRAM, single
  GPU, fits a 24GB card at 1.7B).
- **Compute:** one A100/H100 via Modal, RunPod, or Colab.
- **Do not tune hyperparameters to fix a data problem.** Nine times out of ten a
  disappointing model is a data problem. Fix the data, retrain.

### 7.1 Calibration (the piece that makes abstention real)

Cross-entropy training makes models overconfident, so raw softmax is not a
calibrated probability (Guo et al. 2017). After SFT:
1. Fit a single temperature scalar `T` on the held-out `val.jsonl` by minimizing
   negative log-likelihood, dividing logits by `T` before softmax.
2. Choose the abstention threshold on the calibrated confidence using the same
   held-out set, trading off accuracy-on-answered vs abstention rate.
This is one post-hoc parameter and it is what turns raw scores into usable
confidence. Report calibration, not just accuracy (see 8).

---

## 8. Evaluation (build this BEFORE training)

Without an eval, "we fine-tuned a model" is unfalsifiable. The harness
(`run_baseline.py`, included) already scores three of the four required things.

### 8.1 Required metrics, base vs tuned, on the same held-out scenarios
- **Accuracy:** modal predicted label vs gold label.
- **Schema validity:** did the model output exactly one clean label, no prose.
- **Consistency:** same input, same label across N repeat runs.
- **Calibration (add for the tuned model):** Expected Calibration Error (ECE) and a
  reliability diagram. This is what proves SPOV 3. TODO: extend `run_baseline.py`
  with an ECE function once the model emits confidences.

### 8.2 What a win looks like
A tuned model that beats the base on **spec adherence (schema + consistency)** and
**robustness** is a win, per the assignment rubric, even if raw accuracy moves
little. Expect the base 1.7B to name plausible labels but drift on schema and
consistency and to force labels on the abstain cases. The headline is the delta on
reliability, framed as "behavior from data," not "smarter than the base."

### 8.3 Reliability gate before expanding the taxonomy
Double-code a subset by hand, target Cohen's kappa at or above 0.7. If any label's
per-label agreement is low, merge it upward rather than splitting. Do not add
sub-categories until this passes.

### 8.4 Stretch (in rough order)
1. DPO on preference pairs (on-spec vs off-spec) on top of SFT.
2. Adversarial eval: malformed equations, prompt-injection attempts to force a
   label on abstain cases. Report robustness under attack.
3. Composed behavior: hold two constraints at once (correct label AND correct
   abstention) and show data can encode both.

---

## 9. The litmus test (run this FIRST, day 1)

The whole project is justified only if a well-prompted base model cannot already do
this reliably. Do not assume it, measure it.

```
pip install transformers torch accelerate
python run_baseline.py --model Qwen/Qwen3-1.7B --data linear_equation_errors_testset.jsonl --runs 5
```

- **Pass signal (justifies the project):** low schema validity or low consistency,
  and forced labels on the abstain cases (ex09, ex10). This is the expected result.
- **Fail signal (rethink scope):** high accuracy AND high consistency AND high
  schema validity from the prompt alone. If so, the base model is too capable or
  the task is too easy; narrow the scope or move to Qwen3-0.6B.
- Frontier models likely have the capability, that is fine and expected. The test
  is about the small base model specifically, not about whether the task is
  possible in principle. Distillation works because the frontier is good.

---

## 10. One-week arc

| Day | Focus | Checkpoint |
|---|---|---|
| 1 | Setup, research, BrainLift, litmus test | Base model runs; litmus numbers on the board; target behavior known; SPOVs match behavior. |
| 2 | Behavior spec, eval harness, data-gen pipeline; 50 junk examples end to end | Full loop generate -> train -> eval runs. |
| 3 | v1 dataset, first real training run, first base-vs-tuned eval | Midweek gate: base-vs-tuned numbers exist. |
| 4 | Diagnose failure modes, fix in data not hyperparameters, retrain | One failure mode resolved via data iteration. |
| 5 | Final eval + error analysis, calibration, ship inference demo, write BrainLift, record demo | Final submission package ready. |

---

## 11. Final submission package
1. The dataset, published (the real artifact).
2. The model on the Hugging Face Hub plus a running inference demo.
3. Eval harness plus results table, base vs tuned, with the behavior metrics.
4. BrainLift: behavior thesis and whether data-to-behavior held, with evidence.
5. A 3 to 5 minute demo showing the tuned model doing what the base fails to do
   reliably (hold the schema, stay consistent, and abstain on the ambiguous cases).

---

## 12. Included files
- `algebra_error_classifier_spec.md` — this document.
- `test_holdout.jsonl` — ~500 synthetic held-out examples (seed 2) for quantitative base-vs-tuned eval
- `linear_equation_errors_testset.jsonl` / `testset.jsonl` — 12 hand-labeled examples for qualitative litmus
  test and as a never-train reality check. Covers all 7 labels. Verify ex08, ex11,
  ex12 against your own reading; they were marked medium confidence.
- `run_baseline.py` — the eval/litmus harness (accuracy, schema validity,
  consistency, confusion table). Self-test with `--selftest`. TODO: add ECE.
- `generate_dataset.py` — forward error-injection data generator, class-balanced,
  tested. TODO: `teacher_distill.py` for the optional realism pass.

---

## 13. Open decisions to make before day 2
- Output schema: Option A (label only, calibrated confidence) vs Option B (label +
  bucket). Recommendation: A.
- Whether to add the teacher-distillation realism pass, or ship on injected data
  alone for week one.
- Verify the flagged figures in the BrainLift (Knuth 58/29 split, Kuchemann level
  percentages, the "Correct Answer Trap" arXiv id) before citing them as fixed.
