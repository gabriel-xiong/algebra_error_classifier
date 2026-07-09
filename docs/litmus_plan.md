# Litmus Plan — gating the SLM pivot before any item bank or fine-tuning

Status: **draft / additive.** This document and its companion files
(`data/apbio_misconceptions.json`, `data/litmus_apbio_seed.jsonl`,
`scripts/litmus_tagging.py`, `scripts/litmus_generation.py`, `notebooks/litmus.ipynb`)
do not modify any existing script. They implement the **litmus test** that
`docs/mcat_pivot_spec.md` §7 and §9.3 explicitly require *before* committing to an
item bank or fine-tuning.

Read alongside `docs/mcat_pivot_spec.md` (the pivot spec) and `scripts/common_bio.py`
(the coarse v1 taxonomy this reconciles with).

> **There are now TWO litmuses.** Sections 1–9 cover the **tagging** litmus
> (`litmus_tagging.py`). **Section 10 covers the GENERATION litmus**
> (`litmus_generation.py`), which — per the mentor-endorsed pivot
> (`mcat_pivot_spec.md` §12) — is the **PRIMARY v1 task**: conditionally generating
> misconception-tagged items. The tagging litmus is reused there as the independent
> **tag-fidelity verifier**. If you only read one section for the current plan, read
> §10.

---

## 1. What the litmus tests

The pivot proposes: a small model (`Qwen/Qwen3-1.7B`) that **tags a student's
chosen wrong answer (distractor)** on an AP Bio MCQ to a **misconception**, and
**generalizes** that tagging to NEW items it never saw tagged — not a hand-built
lookup table.

Before building anything, we test whether a **prompted (zero-shot, NOT
fine-tuned)** model can *already* do this **consistently**. This is prompted-only:
no fine-tuning happens anywhere in the litmus.

The measured question:

> Given a passage/stem + all choices + the correct answer + the student's CHOSEN
> distractor, can the model predict the authored misconception (from a provided
> candidate list), and produce the **same** answer across repeated runs on item
> stems it never saw tagged?

**The bar is CONSISTENCY, not one-shot accuracy.** A prompt that lands the right
tag once but flips across runs is exactly the reliability gap the pivot would
fill. (Same discipline as `run_baseline.py`'s modal-consistency metric.)

## 2. The 2x2 interpretation (how the result decides the project)

| base `Qwen3-1.7B` consistent? | frontier consistent? | Verdict |
|---|---|---|
| **yes** | (any) | **No fine-tuning project.** A prompt already does it. |
| **no** | **yes** | **Distill** the frontier capability into a deployable SLM. |
| **no** | **no** | **Fine-tuning is justified** — *then* check the labels aren't ambiguous (see below). |

Reading aids the harness prints:
- **Low schema validity alone** (model can't emit a clean id) is a
  prompt-format/parsing gap, not a capability gap — fix the prompt, don't conclude
  "needs fine-tuning."
- **High consistency + low accuracy on a label** = the model is *confidently
  wrong*, which usually points to **label ambiguity** (two misconceptions fit the
  same distractor) rather than a capability deficit. That triggers the kappa /
  double-coding gate (`mcat_pivot_spec.md` §8) before any taxonomy split.

### Honest fallback
If a prompt does mid-grained tagging fine, the SLM's value is **not** a capability
story. It becomes a **deployment-efficiency** story: cheap, on-device / at-scale
tagging where calling a frontier API per attempt is too slow or too expensive.
That is a legitimate reason to ship a small model, but it must be stated as such —
not dressed up as a capability the small model uniquely has.

## 3. Granularity decision — why MID-GRAINED

Three granularities were possible; we deliberately chose the middle:

| Granularity | What it is | Why NOT (for the litmus) |
|---|---|---|
| **Coarse** (the ~3–5 cognitive-kind labels in `common_bio.py`: `content_gap` / `reasoning_error` / `misread_or_passage_mapping`) | Very few, broad buckets | A prompt may *already* do this reliably, so it can't discriminate the pivot decision. Too easy to be a gate. |
| **Fine** (Eedi-style, ~2,500 misconceptions) | One label per specific misconception, retrieval-scale | That is just **re-running the Eedi retrieval benchmark** — a different, much larger problem with its own literature and infra. Not what this project is deciding. |
| **Mid-grained** (this plan: ~30–50 AP Bio misconceptions scoped to a few topics) | Named misconceptions per topic | Hard enough that a prompt might fail (so the gate is meaningful) but bounded enough to author, tag, and evaluate honestly. |

`data/apbio_misconceptions.json` defines **46** misconceptions across
**evolution, genetics, cellular respiration, photosynthesis, membrane transport,
enzymes, and experimental design**.

## 4. Two-layer taxonomy: mid-grained misconception → coarse cognitive-kind

The mid-grained layer does **not contradict** `common_bio.py`; it sits *under* it.
Every misconception carries a `coarse` field that **maps up** to exactly one of
the v1 cognitive-kind labels the product's three-score model actually consumes:

```
mid-grained misconception  (SLM output, ~46 ids)   e.g. cr_o2_is_electron_donor
        │  maps up (deterministic, stored in the taxonomy)
        ▼
coarse cognitive-kind      (common_bio SUBSTANTIVE_LABELS)   content_gap
        │  routes to
        ▼
three-score product        (memory / performance)      (see mcat_pivot_spec §5)
```

`abstain` stays a **meta/decision** label in `common_bio` (predict-and-confirm),
not a misconception, so it is not in the mid-grained list. The harness validates
at load time that every `coarse` value is one of
`common_bio.SUBSTANTIVE_LABELS`, so the two layers can never silently drift apart.
It reports **both** mid-grained accuracy and, after mapping up, **coarse-label
accuracy** — so you can see whether the model gets the *kind* of error right even
when it misses the specific misconception.

## 5. Generalization target — IN-DISTRIBUTION misconceptions on NOVEL stems

The generalization we require (and the only one we claim) is:

> **The same ~46 misconceptions, applied to item stems the model never saw
> tagged.** Held-out eval items reuse the *training-set misconceptions*; they are
> new *questions*, not new *misconceptions*.

We do **NOT** ask the model to invent or predict *unseen* misconceptions. That is
a harder, separate problem and is out of scope for this gate. This mirrors the
synthetic-train / real-eval split in `mcat_pivot_spec.md` §10: novelty lives in
the item stems, the label space is fixed.

## 6. Deliverables in this drop

| File | What it is |
|---|---|
| `data/apbio_misconceptions.json` | The 46 mid-grained misconceptions (id, name, description, `coarse`, topic). **Authored / illustrative.** |
| `data/litmus_apbio_seed.jsonl` | 20 AP Bio MCQ items in the `apbio_item_template` schema, each wrong answer tagged to a misconception id (incl. experimental-design/passage stems). **AUTHORED PLACEHOLDERS.** |
| `scripts/litmus_tagging.py` | The prompted (zero-shot) tagging harness. |
| `notebooks/litmus.ipynb` | Colab GPU runner (base 1.7B + optional frontier). New notebook; `train_and_eval.ipynb` is untouched. |
| `docs/litmus_plan.md` | This document. |

### Data honesty flag
`data/litmus_apbio_seed.jsonl` is a set of **AUTHORED PLACEHOLDERS** written to
exercise the harness and shape the taxonomy. **The DEFENSIBLE run needs ~40–50
REAL tagged items.** Per the owner's decision (`mcat_pivot_spec.md` §10), **real
items are eval-only**; synthetic/authored items never count toward reported
accuracy. So the placeholder seed is enough to prove the harness runs and to
smoke-test the pipeline, but **not** enough to make the gating call.

## 7. The harness (`scripts/litmus_tagging.py`)

- **Task per example:** each item is expanded into one example per tagged wrong
  choice (20 items → **60** item×distractor examples). For each, the prompt gives
  the passage/stem, all choices, the correct answer, and the student's chosen
  distractor, plus the **full candidate misconception list** (constrained
  classification, not open generation). The gold tag is **never** in the prompt.
- **Backends (uniform `generate(system, user)` interface):**
  - `--model` (HF id, default `Qwen/Qwen3-1.7B`) via `model_utils.HFClassifier`.
  - `--selftest` → a local dummy tagger (reuses `DummyModel`'s spirit) so the
    pipeline runs with no GPU/download.
  - `--frontier {openai,anthropic,auto}` → optional zero-shot frontier arm via
    `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. **If the key or client library is
    missing it prints a clean skip message and exits 0** — it never crashes.
  - `--runs N` (consistency), `--max-examples`, `--temperature`,
    `--save-predictions`.
- **Metrics printed:** mid-grained tagging accuracy, coarse-label accuracy (after
  mapping up), **consistency** (modal agreement across `--runs`, like
  `run_baseline`), mid-grained + coarse confusion summaries, per-misconception
  support (n / accuracy / consistency), schema validity, and the 2x2
  interpretation guide.
- **Reuse:** imports `HFClassifier` from `model_utils`, `summarize_confusion` from
  `metrics`, and the coarse label set from `common_bio`; parsing follows
  `common.parse_label`'s contract (exact = clean, recovered substring/name = valid
  but not clean).

### Local smoke test (done)
Windows CPU can only run the dummy path — that is expected. Verified:

```
python scripts/litmus_tagging.py --selftest --data data/litmus_apbio_seed.jsonl --runs 3
```

It loads 46 misconceptions and 20 items → 60 examples and prints the full metrics
block. (The dummy is intentionally noisy so consistency/schema land below 100%.)

## 8. Exact Colab run steps (the real gate)

Open `notebooks/litmus.ipynb` on Colab, **Runtime → T4 GPU**, run top to bottom:

1. **Clone + install** — clones the repo, installs `transformers accelerate
   torch` (no Unsloth, no training).
2. **Sanity** — runs the `--selftest` head to confirm harness + data are present.
3. **Arm 1 (base 1.7B):**
   ```
   python scripts/litmus_tagging.py --model Qwen/Qwen3-1.7B \
     --data data/litmus_apbio_seed.jsonl --runs 5 \
     --save-predictions outputs/litmus/base_qwen3_1p7b.jsonl
   ```
4. **Arm 2 (frontier, optional):** set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
   in the key cell, then:
   ```
   python scripts/litmus_tagging.py --frontier auto \
     --data data/litmus_apbio_seed.jsonl --runs 5 \
     --save-predictions outputs/litmus/frontier.jsonl
   ```
5. **Read the 2x2** and record the verdict.

## 9. This gates the pivot

**Do not build the full item bank and do not fine-tune until the litmus results
are in.** The litmus decides whether there is a fine-tuning project at all:

- If the prompt already tags mid-grained misconceptions consistently → the story
  is deployment efficiency, not capability (§2 fallback).
- If only a frontier model can → the project is distillation.
- If neither can → fine-tuning is justified, *after* confirming the labels aren't
  ambiguous.

And the numbers above only become **defensible** once the placeholder seed is
replaced with ~40–50 **real** tagged items and both arms are run on a GPU (base
1.7B) / with an API key (frontier).

---

## 10. The GENERATION litmus (PRIMARY — `scripts/litmus_generation.py`)

Per the mentor-endorsed pivot (`mcat_pivot_spec.md` §12), the **primary v1 SLM
task is conditional generation of misconception-tagged items**, and this is the
litmus that gates it. It is prompted zero/few-shot only — **no fine-tuning**.

### 10.1 What it tests

> Given a spec `{topic, target misconception(s) per distractor, difficulty,
> format}`, can a *prompted* model emit a full, well-formed, correctly-**tagged**
> AP Bio item — reliably, without drifting off-schema or attaching invalid tags?

The thesis under test is **controllable, anti-drift structured generation** (not
calibrated abstention): generic AI questions drift and lose valid tags; the value
of a small model is producing reliably-schema'd, validly-tagged items on demand.

### 10.2 Generate-and-verify — the verifiers ARE the metric

This is **measurement, not a production pipeline** (`mcat_pivot_spec.md` §12.4).
For each generated item we run:

- **V1 — Structural validity (deterministic code):** well-formed schema, exactly
  one correct answer, N distractors, every distractor tagged with a real taxonomy
  id from `data/apbio_misconceptions.json`, no duplicate/degenerate choices.
  *This is the anti-drift metric.* Malformed JSON counts as invalid — that IS the
  drift signal.
- **Spec-adherence:** does the item's topic + target misconception(s) match the
  requested spec?
- **V3 — Tag-fidelity (the crux):** an **independent** tagger
  (`litmus_tagging.predict_tag`, reused — imported, not duplicated) re-reads each
  generated distractor and predicts a misconception; we check agreement with the
  generator's **claimed** tag.

**Deferred to prod/future (described, NOT built):** the V2 solvability
"solve-it-blind" solver arm, and the full rejection-sampling + scaled-human-review
pipeline (`mcat_pivot_spec.md` §12.4).

### 10.3 Metrics (per arm)

- **Validity rate (V1)** — anti-drift.
- **Spec-adherence rate.**
- **Tag-fidelity rate (V3)** — per-distractor verifier↔claim agreement; the crux.
- **Diversity / dedup rate** — unique stems (mode-collapse guard; reuses
  `generate_dataset`'s seen-set dedup pattern).
- **Yield** — `valid ∧ faithful ÷ generated`, the bottom-line usable rate.

Plus a per-topic validity/adherence breakdown and the top structural-invalidity
reasons (the concrete drift modes).

### 10.4 The 2x2 (distillation gate)

| base `Qwen3-1.7B` reliable? | frontier reliable? | Verdict |
|---|---|---|
| **yes** | (any) | A prompt suffices → SLM value is **deployment efficiency** only. |
| **no (drifts)** | **yes** | **Distill** the frontier teacher into a deployable small generator. ← the pivot. |
| **no** | **no** | The schema / taxonomy / task needs rework before training. |

### 10.5 Independence + human anchor (non-negotiable)

- The tag-fidelity verifier must be a **DIFFERENT model** than the generator
  (frontier, or a separate HF tagger). The harness **enforces** this: with no
  independent verifier configured it reports tag-fidelity as **N/A** rather than
  letting the generator self-grade.
- A **~30–50-item human-labeled sample** must anchor verifier↔human agreement
  before the tag-fidelity number is trusted. Described here; a small sample, not a
  pipeline (`mcat_pivot_spec.md` §12.6).

### 10.6 Arms & flags

- `--selftest` — dummy generator (emits schema-ish JSON with realistic drift) +
  dummy verifier (agrees ~70%); runs with no GPU/download.
- `--model <hf id>` (default `Qwen/Qwen3-1.7B`) — base generator via
  `HFClassifier.generate`.
- `--frontier {openai,anthropic,auto}` — prompted frontier **generator** (teacher);
  skips cleanly with no key/lib.
- `--verifier-frontier {...}` / `--verifier-model <hf id>` — the **independent**
  tag-fidelity verifier (must differ from the generator).
- `--num-specs`, `--max-examples`, `--runs` (generations per spec), `--tagger-runs`.

### 10.7 Local smoke test (done)

Windows CPU runs the dummy path only:

```
python scripts/litmus_generation.py --selftest --num-specs 5 --runs 1
```

Verified: it builds specs across topics, generates dummy items, runs V1 /
spec-adherence / V3, and prints validity / spec-adherence / tag-fidelity /
diversity / yield with the drift-reason breakdown and the 2x2 guide.

### 10.8 Exact Colab run steps (the real gate)

In `notebooks/litmus.ipynb` (Runtime → T4 GPU), the generation section runs:

1. **Base 1.7B generator + frontier verifier** (independent):
   ```
   python scripts/litmus_generation.py --model Qwen/Qwen3-1.7B \
     --verifier-frontier auto --num-specs 20 --runs 2 \
     --save-predictions outputs/litmus/gen_base_1p7b.jsonl
   ```
   (If no frontier key: use `--verifier-model` with a *different* HF tagger, or
   accept tag-fidelity = N/A for that run.)
2. **Frontier generator (teacher) + independent verifier:**
   ```
   python scripts/litmus_generation.py --frontier auto \
     --verifier-model Qwen/Qwen3-1.7B --num-specs 20 \
     --save-predictions outputs/litmus/gen_frontier.jsonl
   ```
3. **Read the 2x2** (§10.4) and record the verdict.

### 10.9 This gates the pivot (generation edition)

**Do not build the full item bank and do not fine-tune until the generation
litmus results are in.** Defensible numbers require a **GPU** (base 1.7B), an
optional **frontier API key** (teacher and/or independent verifier), and the
**~30–50-item human-anchored fidelity sample**. The shipped specs are AUTHORED
PLACEHOLDERS built from the taxonomy.
