# MCAT Pivot: AP Bio Error-Typing Model Layer — Design Spec

Status: **draft / proposal**. This document is additive. It does not change any
existing algebra code. The algebra error-type classifier stays intact as the
working reference implementation; this spec describes how that same machinery is
re-pointed at a new content domain (AP Biology, as a bounded proxy for MCAT bio)
and re-framed as one layer of a larger MCAT study tool.

Read alongside `docs/spec.md` (the algebra build spec this pivot inherits from)
and `docs/brainlift.md` (research grounding). Two new draft artifacts ship with
this spec: `scripts/common_bio.py` (taxonomy/prompt draft) and
`data/apbio_item_template.jsonl` (schema examples).

---

## 0. TL;DR of the pivot

The algebra project fine-tuned a small model (Qwen3-1.7B, QLoRA) to name **why** a
student got a linear equation wrong, from a fixed 7-label taxonomy, with
calibrated abstention. The thesis was *behavior from data*: schema-faithfulness,
consistency, calibrated abstention — not raw capability.

The pivot keeps that thesis and that machinery, and changes three things:

1. **Content domain:** linear equations → AP Bio multiple-choice items (cellular
   respiration, genetics, membrane transport, experimental design), a bounded
   tractable stand-in for MCAT bio.
2. **Input signal (v1):** worked algebra steps → **the item + the student's chosen
   distractor**. In MCQ settings the primary observable is *which wrong answer
   they picked*, and distractor choice is itself a **model-layer behavioral
   signal** — captured at answer time with zero extra infrastructure — that
   carries diagnostic information (the MC-DINA distractor→misconception line of
   work, de la Torre, beats correct/incorrect-only scoring by ~29%). Response
   **timing** is explicitly *deferred* to Phase 2+ because it requires an
   instrumented UI collecting real student data that does not exist yet (see §2.2,
   §7, §9.4).
3. **Role:** a standalone classifier → the **error-typing model layer** of a
   study tool that surfaces three live scores (memory / performance / readiness)
   and runs a **predict-and-confirm** loop with the student.

**Two owner decisions are resolved in this revision (see §10):**
- **Item bank = synthetic-train / real-eval** (the algebra project's existing
  pattern): pull real AP Bio / MCAT-style questions, generate synthetic *training*
  items seeded on those real items, and reserve the real questions **exclusively**
  as the eval set.
- **Behavioral signal = distractor choice for v1; timing deferred.** The v1 engine
  is centered on distractor choice alone and supports `content_gap`,
  `reasoning_error`, and `misread_or_passage_mapping`. `careless_slip` is **dropped
  from the v1 taxonomy** because a slip and a content gap can select the same
  distractor; it returns only once real timing / repeated-attempt data exists.

> **UPDATE — mentor-endorsed pivot (see new §12): the PRIMARY v1 SLM task is now
> CONDITIONAL GENERATION of misconception-tagged items, not just tagging existing
> ones.** The SLM takes a spec `{topic, target misconception(s) per distractor,
> difficulty, format}` and emits a full tagged item. This trades the
> calibrated-abstention thesis for a **controllable, anti-drift structured-generation
> thesis**. The distractor→misconception TAGGING work above is reframed as the
> independent **tag-fidelity verifier** for generated items; the two compose into a
> **generate-and-verify** loop. All decisions in this §0 (synthetic-train/real-eval,
> distractor-choice signal, timing deferred, the mid-grained taxonomy) still hold —
> §12 is additive and consistent with them.

Everything the algebra project already does well maps onto this: the taxonomy is
still the hard part, calibration/abstention becomes "defer to the student when
unsure," and the eval harness's schema/consistency/ECE metrics gain one new
headline metric — **override rate**.

---

## 1. Why pivot (theses from the BrainLift)

Every design choice below is anchored to one of the owner's strategic points of
view (SPOVs). They are restated here so a reviewer can check each design element
against its justification.

- **SPOV1 — WHAT vs WHY.** Current tools record *what* a student missed ("missed a
  question in topic X"), collapsing very different failures. A miss from misreading
  the passage, a miss from a reasoning error, and a miss from a genuine content gap
  carry completely different information and imply completely different remediation.
  Personalization requires capturing error **type** at the moment it happens.

- **SPOV2 — students can't self-diagnose.** Kruger-Dunning double-burden: the
  students who err most are worst at detecting it. Andrews et al.: <10% of med
  students adapt their reasoning after a wrong answer; 84% default to generic
  ability self-criticism ("I'm bad at this") rather than diagnosing the error.
  **Conclusion: error classification must live at the MODEL LAYER, driven by
  behavioral signals, not self-report.**

- **SPOV3 — readiness signal comes too late.** Students do ~240 hrs over ~3 months
  before any trustworthy readiness signal appears. The product needs a **live**
  readiness signal early (within ~6 weeks). Artino: 95-98% of med students
  overestimate their performance, so the signal must be **externally calibrated**,
  not self-assessed.

- **SPOV4 — recall ≠ application.** Content review and standalone practice
  questions don't automatically close the recall→application gap. Declarative
  knowledge ("knowing what") and procedural knowledge ("knowing how") are distinct
  outcomes and must be **measured separately.**

The algebra project already proved out the enabling capability: a small model can
be trained to apply a fixed error taxonomy consistently and abstain when the
signal is thin. That is exactly the primitive SPOV1 and SPOV2 require. The pivot
is: point it at bio, feed it distractor + timing, and wrap it in the three-score
product.

---

## 2. The reframing: SLM as the error-typing model layer

### 2.1 Behavior spec (the falsifiable gate, pivoted — v1)

> Given an AP Bio multiple-choice item (passage if present, stem, all answer
> choices, the correct answer, and the specific distractor the student selected),
> the model returns exactly one error-type label from the fixed v1 taxonomy
> (`content_gap`, `reasoning_error`, `misread_or_passage_mapping`, `abstain`) and a
> calibrated confidence, with no prose. When its top label is low-confidence or two
> error types tie, it **abstains into predict-and-confirm**: it surfaces its top-2
> candidates and asks the student to confirm or adjust (structured selection, never
> open generation).

Note the input in v1 is **distractor choice only** — no timing. As in the algebra
spec, a stranger can mark any output pass/fail against this. It is simultaneously
the data-authoring rubric, the eval criterion, and the thesis.

### 2.2 Two behavioral signals, NOT equally available

The central architectural point of this revision. There are two model-layer
behavioral signals (both are behavioral, i.e. observed-from-what-the-student-did,
not self-report), and they differ sharply in availability:

| Signal | Availability | Role |
|---|---|---|
| **Distractor choice** | **Available now.** Captured at answer time, zero infrastructure. This is the exact signal the MC-DINA distractor→misconception literature (de la Torre; ~29% gain over correct/incorrect-only) is built on. | The **v1 engine.** Fully a behavioral (not self-report) signal, so it satisfies SPOV2 on its own (see §2.4). |
| **Response timing** | **Not available.** Requires an instrumented UI collecting real student response times, which does not exist yet. | A **deferred Phase-2+ enhancement**, contingent on real data (§7, §9.4). |

### 2.3 What the SLM does vs. what a (future) timing model does

The two are **complementary**, not competing:

| Question | Best answered by | Why |
|---|---|---|
| *"Why is this specific distractor attractive? What content gap / reasoning error / misread does choosing it reveal?"* | **The SLM (v1)** | Semantic. Requires reading the passage/stem, understanding the biology, and mapping a wrong choice to an error-type profile. This is the algebra model's core competency, re-pointed, and it is what distractor choice supports well. |
| *"Was this a careless slip or a genuine error?"* | **A future timing model (Phase 2+)** | A slip and a genuine content gap can select the **same** distractor, so semantics alone cannot separate them. Response timing discriminates them (San Pedro's slip detector reaches A'=0.82 from behavioral features) — but only once real timing data exists. |
| *"How confident should we be, and when do we ask the student?"* | **Calibration layer (shared)** | The algebra project's temperature scaling + abstain threshold, re-used verbatim in spirit. |

**v1 combination:**

1. The SLM consumes item + chosen distractor and emits a **distribution over the
   four v1 error types** (calibrated log-probs, exactly like `score_labels` today).
2. The **calibration layer** decides the action: emit the top label if confident;
   otherwise enter **predict-and-confirm** with the top-2.

**Phase-2+ combination (only when real timing exists):** insert a **timing gate**
between steps 1 and 2 that re-weights toward a (then-live) `careless_slip` label on
abnormally fast responses. The `apply_timing_gate` helper in `common_bio.py` is a
labeled, currently-inert sketch of this; it is a no-op in v1.

### 2.4 SPOV2 is satisfied in v1 without timing

SPOV2 says error classification must happen at the **model layer, from behavioral
signals, not self-report**. Distractor choice *is* that behavioral, model-layer
signal — it is what the student did, read by the model, with no self-diagnosis
required. So v1 satisfies SPOV2 as stated. Student self-report enters only as a
**secondary confirmation** through predict-and-confirm, and even there it is
structured selection (never open generation), which is exactly the form the
evidence supports (Yerushalmi/Cohen 0.73 vs 0.24; Johnson-Mayer d=1.2 vs -0.06).
Timing, when it arrives, *sharpens* SPOV2 (adds the slip/effort split); it is not
required to satisfy it.

**Honesty flag (see also §9):** whether the SLM meaningfully beats a plain
distractor→misconception lookup table is an open empirical question. The algebra
project's own litmus test (does a well-prompted base model already do this?) must
be repeated here before claiming the SLM earns its keep. For a fixed item bank
with pre-tagged distractors the lookup is a *strong* baseline; the SLM earns its
keep mainly on **new/untagged** items (see §9.3).

---

## 3. New error-type taxonomy (AP Bio / MCAT)

Grounded in the BrainLift SPOVs. Where the algebra taxonomy was carved along
*algebraic operations* (Booth et al.), this taxonomy is carved along the
**declarative/procedural/comprehension** axes the SPOVs demand, because the
product needs to route each error type to a different remediation and to different
scores.

### 3.1 v1 taxonomy (4 labels — inferable from distractor choice alone)

| Label | Kind | Definition | Grounded in |
|---|---|---|---|
| `content_gap` | Declarative | Student lacks the underlying fact/concept the item tests (couldn't recall or never learned it). The distractor reflects a wrong/absent fact, not a reasoning slip. Routes to the **memory score**. | SPOV4 (declarative outcome); SPOV1 (a genuine content gap is distinct info). |
| `reasoning_error` | Procedural | Student has the facts but misapplied them — wrong inference, mis-integrated two concepts, chose a distractor that is a classic *application* trap. Routes to the **performance score**. | SPOV4 (procedural outcome distinct from declarative); MC-DINA distractor→misconception mapping. |
| `misread_or_passage_mapping` | Comprehension | Student mapped the passage/stem to the wrong quantity, missed a qualifier ("EXCEPT", "NOT", "increases"), or mis-read an experimental figure/axis. The bio content may be known; the failure is passage→question mapping. | SPOV1 (misreading is distinct from a content gap); MCAT is passage-heavy, so this is a first-class failure mode. |
| `abstain` | Meta | The chosen distractor does not support a confident single label, or two error types tie. Triggers **predict-and-confirm** deferral to the student. | SPOV2 (students overestimate; externally calibrated deferral); Artino 95-98% overestimate. |

### 3.2 Deferred label (Phase 2+, requires timing / repeated-attempt data)

| Label | Kind | Why deferred |
|---|---|---|
| `careless_slip` | Behavioral (timing) | A careless slip and a genuine `content_gap` can select the **exact same distractor**, so a slip is **not separable from a single distractor choice**. It needs response timing (too-fast on a familiar topic) or repeated-attempt signal (got it right on retry) to identify. Emitting it in v1 would mean guessing, which contradicts the honest-measurement thesis. It becomes a live label in Phase 2 once real instrumented timing exists (San Pedro A'=0.82 slip detector as the target). |

Until then, an item that is *really* a slip will surface in v1 as its
best-semantic-fit label (often `content_gap`) or, when genuinely ambiguous, as
`abstain` → predict-and-confirm — which is the honest behavior, not a silent
mislabel.

### 3.3 How this differs from the algebra taxonomy, and why

- **Algebra:** six *operation-specific* substantive labels
  (`distribution_property_error`, `negative_sign_error`, …) + `abstain`. Carved by
  *what algebraic move went wrong*, observable from **shown work**.
- **AP Bio (v1):** three *cognitive-kind* substantive labels + `abstain`. Carved by
  *what kind of knowledge failed*, observable from **distractor choice** because
  there is no worked trace in MCQ. (A fourth, behavioral `careless_slip`, is
  deferred until timing exists — §3.2.)

The change is forced by the pivot's inputs and product goals. The algebra
taxonomy could afford to be operation-specific because the worked steps expose the
operation. In MCQ, the only v1 signal is the chosen distractor, so the taxonomy
must be coarse enough to be reliably inferable from that one choice, and must line
up with the three scores (declarative→memory, procedural→performance,
comprehension→performance/diagnostic routing). This mirrors the algebra spec's own
"granularity is a tradeoff, not a maximization" principle (Liu et al. 2023:
GPT-4 fell 91.9%→39.8% going from 4 to 100 labels): start at **4 labels**
(3 substantive + abstain), add `careless_slip` only when timing supports it, and
otherwise split further only after inter-rater agreement (Cohen's kappa ≥ 0.7) is
measured, exactly the gate `docs/spec.md` §8.3 already prescribes.

**Note — content-specific sub-tags are metadata, not labels.** Per-distractor
rationale and a specific misconception name (e.g. "confuses osmosis direction")
are stored as fields on the item (see §4), so the coarse 4-label output stays
reliable while richer content lives in the item bank for analytics and
remediation copy.

---

## 4. AP Bio item + distractor-tag schema

The core new data artifact. Every wrong answer option is **pre-mapped** to an
error-type profile at authoring time. This must be built in from the start;
retrofitting distractor tags onto an untagged bank is the expensive path. See
`data/apbio_item_template.jsonl` for filled examples.

### 4.1 Item schema (authoring time — the item bank)

```json
{
  "id": "apbio_cellresp_0001",
  "topic": "cellular_respiration",
  "subtopic": "electron_transport_chain",
  "mcat_skill": 1,
  "knowledge_type": "declarative",
  "difficulty": "medium",
  "passage": null,
  "stem": "In the electron transport chain, what is the direct role of oxygen?",
  "choices": {
    "A": "Final electron acceptor",
    "B": "Donates electrons to NADH",
    "C": "Phosphorylates ADP directly",
    "D": "Catalyzes the citric acid cycle"
  },
  "correct": "A",
  "distractor_tags": {
    "B": {
      "error_type": "content_gap",
      "misconception": "Reverses electron donor/acceptor roles",
      "rationale": "Choosing B means the student does not hold the fact that O2 is the terminal acceptor; they have the concept inverted."
    },
    "C": {
      "error_type": "reasoning_error",
      "misconception": "Conflates the ETC with ATP synthase's phosphorylation step",
      "rationale": "The student knows phosphorylation happens nearby but mis-attributes the mechanism — an application/integration error, not a missing fact."
    },
    "D": {
      "error_type": "content_gap",
      "misconception": "Associates O2 with an unrelated pathway stage",
      "rationale": "Selecting a citric-acid-cycle role reflects an absent link between O2 and the ETC."
    }
  },
  "authoring": {
    "source": "authored_draft",
    "validated_by": null,
    "notes": "TEMPLATE / illustrative. Distractor tags are drafts, not expert-reviewed."
  }
}
```

Field reference:

- `mcat_skill` (1-4): MCAT Scientific Inquiry and Reasoning Skills. Skill 1 =
  knowledge of scientific concepts (declarative-leaning); Skills 2-4 = reasoning,
  data-based reasoning, experimental design (procedural-leaning). Used to seed
  `knowledge_type` and to route to memory vs. performance scores.
- `knowledge_type`: `declarative` | `procedural` — which score this item primarily
  feeds (SPOV4).
- `passage`: nullable. Present for experimental-design / data-interpretation items
  (see the template's genetics-cross and membrane-transport-experiment items).
- `distractor_tags`: **every** non-correct choice carries `error_type` (from the
  taxonomy in §3), a specific `misconception` string, and a human-readable
  `rationale`. This is the semantic ground truth the SLM is trained/evaluated
  against.
- `distractor_tags` note: in v1 every `error_type` is one of the **4 v1 labels**
  (no `careless_slip` — §3.2).
- `authoring`: provenance, aligned to the synthetic-train / real-eval split
  (Decision 1). `source` ∈ {`real_seed` (a pulled real item used to seed synthetic
  training), `synthetic_seeded` (generated, seeded on a real item's real
  distractors — training only), `real_eval` (real item held out **exclusively** for
  eval, never trained on), `imported` (external tagged bank, e.g. Eedi)};
  `validated_by` records the reviewer; `seed_item_id` (on `synthetic_seeded` rows)
  points back to the `real_seed` item it was derived from. **Only `real_eval` rows
  are gold for scoring**; synthetic rows are training material and are never used to
  report accuracy.

### 4.2 Attempt schema (inference time — where behavioral signals attach)

At attempt time, the item is joined with what the student did. In **v1 the only
required field is `chosen`** (the distractor). The timing/exposure fields below are
**Phase-2+ and optional** — they are shown here so the schema is forward-compatible,
but v1 neither collects nor consumes them, and we will not fabricate them (§9.4).

```json
{
  "attempt_id": "att_000123",
  "item_id": "apbio_cellresp_0001",
  "student_id": "stu_042",
  "chosen": "C",
  "correct": false,

  "// Phase-2+ (optional, requires instrumented UI; absent in v1):": null,
  "response_time_ms": null,
  "expected_time_ms": null,
  "prior_exposure": null,
  "timestamp": "2026-07-08T15:04:00Z"
}
```

- `chosen` (**v1, required**): the selected option key; join to
  `distractor_tags[chosen]` for the authored semantic tag.
- `response_time_ms` + `expected_time_ms` (**Phase 2+**): when a real UI provides
  them, the ratio drives the timing gate that separates `careless_slip` from an
  effortful error (§2.3). `null` / absent in v1.
- `prior_exposure` (**Phase 2+**): how much the student has seen this topic —
  helps disambiguate `content_gap` (never learned) from `careless_slip` (knew it,
  slipped) once repeated-attempt data exists.

The model's job **in v1**: given item + `distractor_tags[chosen]`, predict the
error type (over the 4 v1 labels) with calibrated confidence, and either commit or
enter predict-and-confirm. Timing joins this pipeline only in Phase 2+.

---

## 5. The three-score model

Three separate scores, surfaced **live** (SPOV3), computed separately for
declarative and procedural knowledge (SPOV4), and updated by error-type signal
(SPOV1).

### 5.1 Definitions

- **Memory score (declarative retention).** Rolling estimate of recall on
  `knowledge_type: declarative` items, decayed over time (spaced-repetition style)
  and **penalized specifically by `content_gap` errors**: a `content_gap` miss on a
  previously-"known" fact drops it and re-queues the fact. (Phase 2+: once
  `careless_slip` is separable via timing, a slip barely dents memory — in v1 an
  unrecognized slip may look like a small `content_gap` hit, an accepted v1
  limitation, §9.6.)
- **Performance score (procedural/application).** Rolling estimate on
  `knowledge_type: procedural` items (MCAT Skills 2-4), driven by
  `reasoning_error` and `misread_or_passage_mapping` rates. This is the
  recall→application gap made measurable (SPOV4).
- **Readiness score.** A calibrated composite predicting exam-day performance,
  blending memory + performance + coverage (breadth of topics seen) + trend.
  Surfaced from week ~1, not month ~3 (SPOV3), with an explicit confidence band
  that narrows as more attempts accrue. Externally calibrated (§6) so it does not
  inherit the student's 95-98% overestimation (Artino / SPOV3).

### 5.2 Live update loop

Each attempt: SLM+timing produce an error type → the error type updates the
matching score(s) → readiness recomputes → the recommendation engine reads the
error-type *mix*, not just the raw score. Example recommendation the mix enables:

> "Your electrochem/ETC **recall is fine** (memory score high, few `content_gap`
> misses). Your misses are **passage-to-question mapping**
> (`misread_or_passage_mapping`). Do targeted passage-mapping drills, not more
> content review."

That recommendation is impossible under a WHAT-only tool (SPOV1); it requires the
error-type layer — and it is fully supported in **v1 from distractor choice
alone**, no timing needed. (Phase 2+ enriches it with "...under time pressure"
once timing exists.)

---

## 6. Reuse map (old component → new role)

The pivot is deliberately high-reuse. Most algebra scripts change *content and
input fields*, not *structure*.

| Algebra component | New role in the pivot | Carryover |
|---|---|---|
| `common.py` (LABELS, TAXONOMY_TEXT, SYSTEM_PROMPT, build_user_prompt, parse_label, load/write_jsonl) | `common_bio.py`: new **4-label v1 taxonomy** (+ `careless_slip` in `DEFERRED_LABELS`), new SYSTEM_PROMPT, `build_user_prompt` over **item + distractor** (timing optional, Phase-2+). `parse_label`, `load_jsonl`, `write_jsonl`, `format_chat` copy over near-verbatim. | **Structure unchanged**; only the taxonomy text and prompt body change. |
| `generate_dataset.py` (forward error injection) | AP Bio **synthetic training-item generator, seeded on real items** (Decision 1). It ingests real pulled items and produces synthetic variants for *training only*, **seeding each synthetic distractor from the real item's actual distractors** so the injected wrong answers keep real pedagogical validity. This is the direct analogue of "each injector tied to a documented misconception" — here, "each synthetic distractor grounded in a real, student-chosen distractor." | Concept + mechanism carry (forward generation into JSONL); the seed is now real items, and output is train-only. |
| `make_splits.py` (disjoint, dedup, balanced splits) | Enforces the **synthetic-train / real-eval boundary**: synthetic (seeded) items → train/val only; real items → held-out eval only, never trained on. Dedup at the **item** level and keep any item's variants/attempts on one side (no leakage). | **Near-verbatim**, plus item-level grouping and the hard train=synthetic / eval=real split. |
| `prepare_sft.py` / `train_sft.py` (Unsloth QLoRA) | Unchanged pipeline; new chat-formatted data from `common_bio.build_sft_messages`. Same base model (Qwen3-1.7B). | **Unchanged.** |
| `calibrate.py` (temperature scaling + abstain threshold) | Confidence calibration for **error-type prediction**; the abstain threshold becomes the **predict-and-confirm trigger**. | **Unchanged algorithm.** Directly satisfies SPOV3's "externally calibrated." |
| `model_utils.py` (`HFClassifier.score_labels`, `apply_abstention`) | `score_labels` over the **4 v1 bio labels** → the error-type distribution. `apply_abstention` → `apply_predict_and_confirm` (below threshold or top-2 tie ⇒ surface top-2 to student). | **Structure unchanged.** The optional timing-gate re-weighting is **deferred to Phase 2+** (`common_bio.apply_timing_gate` is a labeled, inert sketch), not part of the v1 path. |
| `run_baseline.py` (accuracy, schema validity, consistency, ECE) | Error-type accuracy **evaluated on the real held-out items** (Decision 1) vs. authored tags + schema + consistency + ECE, **plus a new headline metric: override rate** (how often the student adjusts the model's top prediction in predict-and-confirm). | Metrics carry; add override-rate; eval set = real items. |
| `mine_hard_negatives.py` (base model mis-scores) | Real items/distractors the base model **mis-types** or is overconfident on → prioritize for the synthetic-generation seed set and upweighting. | **Unchanged logic.** |
| `real_data.py` + `docs/real_data.md` (normalize real dumps, keyword-map free text) | **Two jobs now:** (a) import the **real AP Bio / MCAT-style items** that seed synthetic training and serve as the exclusive eval set (Decision 1); (b) import **real distractor-tagged banks** (Eedi already tags every distractor to a misconception) and map their misconception text onto the 4-label taxonomy. | **Unchanged pattern**; central to the synthetic-train / real-eval split. |
| `metrics.py` (ECE, reliability diagram, confusion) | Same, plus override-rate and per-error-type confusion. | **Near-verbatim.** |
| Abstain-review export (`build_abstain_review`) | Predict-and-confirm review export: sample low-confidence/tie items for expert QA before shipping. | **Unchanged pattern.** |

**Generation additions (primary v1 task — see §12).** With generation now the
primary SLM task, the reuse map gains:

| Component | New role for generation |
|---|---|
| `scripts/litmus_tagging.py` (`predict_tag`, the tagger) | **Reframed as the independent TAG-FIDELITY VERIFIER** (V3): re-reads a generated distractor and predicts its misconception; agreement with the generator's claimed tag is the crux metric. Same code, new job. |
| `scripts/litmus_generation.py` (new) | The **generation litmus**: builds specs from the taxonomy, prompts a model to emit tagged items, and runs the V1/spec-adherence/V3 verifiers that ARE the metric. |
| `generate_dataset.py` (validators / dedup / `_try_add_example` seen-set) | Pattern reused for **structural validation + stem dedup** of generated items (deterministic V1 checks + diversity rate). |
| `run_baseline.py` (summarize patterns) | Metric-summary style reused by the generation harness (per-arm rates + interpretation guide). |
| `prepare_sft`/`train_sft` (QLoRA) | Unchanged pipeline; when generation is fine-tuned, SFT targets become `spec -> tagged-item JSON` pairs instead of `item -> label`. |

**Carries over unchanged:** the training pipeline (`prepare_sft`, `train_sft`),
the calibration algorithm, the QLoRA/Unsloth setup, the base model choice, the
JSONL I/O, `parse_label`, `format_chat`, and the "dataset is the deliverable /
training is a button-press" philosophy.

---

## 7. Phased roadmap

Each phase is shippable and falsifiable on its own, and each maps to specific
SPOVs.

**Phase 0 (updated, PRIMARY) — Conditional generation of tagged items + generate-and-verify (see §12).**
The mentor-endorsed primary v1 SLM task. Concretely: gate with the **generation
litmus** (`scripts/litmus_generation.py`) — does a prompted base 1.7B produce
reliably-schema'd, validly-tagged items, or does it drift where a frontier model
doesn't? The distractor-typing model below (`litmus_tagging.py`) is reused as the
independent **tag-fidelity verifier**. If the litmus shows base-small drifts and a
frontier teacher does not, the deliverable is a **distilled small generator**.
This phase gates whether there is a fine-tuning project at all; do not build the
full item bank or train until its numbers are in.

**Phase 1 — Distractor-choice error-typing, synthetic-train / real-eval (SPOV1, SPOV2).**
The v1 engine. Concretely:
- **Data (Decision 1):** pull real AP Bio / MCAT-style items across 3-4 topics;
  generate **synthetic training items seeded on those real items** (distractors
  seeded from the real items' actual distractors — the pedagogical-validity
  mitigation, §9.1/§9.2); reserve the **real items exclusively as the eval set**.
- **Model:** SLM error-typing on **distractor choice alone**, over the 4 v1 labels
  (`content_gap`, `reasoning_error`, `misread_or_passage_mapping`, `abstain`), with
  calibrated abstention → predict-and-confirm.
- **Eval:** base-vs-tuned on the **real** held-out items —
  schema/consistency/accuracy/ECE. **Litmus test first** (does a well-prompted base
  model already do this from distractor choice?), exactly as `docs/spec.md` §9
  demands.
This is the pure re-point of the algebra project and proves the semantic layer
works in bio, with **no dependency on timing data we don't have**.

**Phase 2 — Add behavioral timing + re-introduce `careless_slip` (SPOV2 sharpened) — CONTINGENT ON REAL DATA.**
**Gated on having a real instrumented UI producing genuine response times.** Only
then: attach `response_time_ms` + `expected_time_ms`; activate the timing gate
(`apply_timing_gate`); add `careless_slip` as a live label and split it from
effortful `reasoning_error`/`content_gap`; validate against a labeled subset
(target A' in San Pedro's ~0.82 range). **We will not fabricate or simulate timing
to unlock this phase early (§9.4).** Repeated-attempt signal (right on retry) is an
alternative slip signal that can precede full timing.

**Phase 3 — Three-score model + recommendation (SPOV3, SPOV4).**
Compute memory / performance / readiness live from the error-type stream; build
the recommendation engine that reads the error-type *mix*. Deliverable: a live
dashboard-grade signal within ~6 weeks of simulated study, externally calibrated.
Works on the v1 (distractor-only) error stream; timing-derived `careless_slip`
enriches it if/when Phase 2 lands.

**Phase 4 — Predict-and-confirm UI + override-rate validation (SPOV2).**
Ship the structured predict/confirm loop (model shows top error mode; student
confirms or picks from a short list — never open generation). Track **override
rate** as the validation metric: does the prediction layer add accuracy, or just
anchor students? Grounding for the *structured* choice: Yerushalmi/Cohen
structured-rubric self-diagnosis 0.73 vs 0.24 open; Johnson-Mayer
selection-from-list d=1.2 vs open generation d≈-0.06.

---

## 8. Evaluation (build before training, as in algebra)

Base-vs-tuned on the **real held-out items** (Decision 1: eval set is real,
never-trained-on items), per `docs/spec.md` §8:

- **Error-type accuracy:** predicted label vs. authored `distractor_tags[chosen].error_type`.
- **Schema validity:** exactly one clean label, no prose (`parse_label` reused).
- **Consistency:** same item+distractor+timing ⇒ same label across N runs.
- **Calibration (ECE + reliability diagram):** the tuned model's confidence must
  track accuracy — this is what makes readiness "externally calibrated" (SPOV3).
- **Override rate (new headline):** in predict-and-confirm, fraction of
  predictions the student adjusts. Interpreted carefully (see §9): a low override
  rate could mean the model is right *or* that students are anchoring. Pair it
  with a held-out expert-labeled slice to separate the two.

**Reliability gate before splitting the taxonomy:** double-code an item subset by
hand, target Cohen's kappa ≥ 0.7 on the 4 v1 labels; merge upward if any label is
low. Do not add sub-labels (or re-introduce `careless_slip` without timing) before
this passes (same rule as algebra §8.3).

---

## 9. Risks & open questions (honesty flags)

Written in the candid spirit of `docs/spec.md`'s known-weakness sections.

### 9.1 Valid distractor-tagged bio items — RESOLVED (Decision 1), with a caveat
The algebra injectors worked because an algebraic misconception deterministically
produces a specific wrong number — the label was *known because we chose the
error*. Bio is not deterministic that way: a distractor's "attractiveness" and the
misconception it reveals are pedagogical judgments. **Decision (owner):**
**synthetic-train / real-eval** — pull real AP Bio / MCAT-style items, generate
synthetic *training* items seeded on them, and use the real items **exclusively as
the eval set** (the exact pattern the algebra project already uses,
`docs/real_data.md`). This resolves the "where does the bank come from" question.
Residual caveats to manage: real MCAT items are copyrighted (use AP Bio /
open-licensed / Eedi-style banks, not scraped MCAT); the real eval set will be
small at first; and license terms (e.g. Eedi CC BY-NC) must be honored.

### 9.2 Can synthetic bio distractors be pedagogically valid? — mitigated by seeding
This is the hard part flagged in the first draft, and it drove Decision 1's
mechanism. For algebra, forward injection guaranteed the error→answer link. For
bio, a *freely* synthesized distractor might be implausible (no student would pick
it) or mislabeled (the misconception assigned isn't the one it triggers).
**Mitigation (core to Decision 1): seed synthetic distractors from the real
items' actual distractors.** Because real distractors were written by item authors
and actually chosen by students, seeding on them carries real pedagogical validity
into the synthetic training set instead of inventing wrong answers from scratch.
Synthetic items remain `authored_draft`, never gold; **gold lives only in the real
eval set**, so a bad synthetic distractor can hurt training but can never inflate
reported eval numbers.

### 9.3 Does an SLM beat a distractor-lookup table?
For a **fixed, pre-tagged** item bank, the error type of a chosen distractor is
*already stored in the item* — a lookup is trivially "100% accurate" against its
own tags. The SLM only earns its keep when: (a) items are **new/untagged** (it
generalizes the tagging to real eval items it never trained on — which is exactly
what synthetic-train / real-eval measures), or (b) we want a *distribution* over
error types rather than one authored tag. **We must run the algebra-style litmus
test and state honestly where the SLM adds value over the lookup.** If it doesn't
generalize to real items, the honest answer may be "lookup for tagged items, SLM
only for cold-start/new-item tagging."

### 9.4 Timing data — DEFERRED, and we will NOT fabricate it
The timing gate and slip detector need **real** response-time distributions to set
thresholds and validate A'. We have none, and building the instrumented UI is out
of v1 scope. **Decision (owner): timing is a Phase-2+ enhancement, contingent on
real instrumented data.** Critically: **we will not fabricate or simulate response
times and report the result as a finding.** Doing so would manufacture a
`careless_slip` signal that looks like measurement but is really an assumption,
directly undermining the project's honest-measurement thesis (the same discipline
behind the algebra spec's "don't tune hyperparameters to fix a data problem" and
its abstain-review gate). v1 therefore ships **without** timing and **without**
`careless_slip`; both arrive together when real data does. Sourcing options for
later: instrument our own practice UI (cleanest; chicken-and-egg), partner/buy
logged data, or bootstrap from public step-timing datasets (PSLC DataShop has
timestamps; MCQ timing datasets are rarer).

### 9.5 Anchoring risk in predict-and-confirm
Showing the model's guess first may bias the student to accept it (the confirm
becomes rubber-stamping), inflating apparent accuracy while adding no diagnostic
value. This is why **override rate is the validation metric, not just accuracy**,
and why the confirm is *structured selection from a short list*, not a single
yes/no. Mitigation to test: occasionally withhold the prediction (blind confirm)
and compare, to measure the anchoring delta.

### 9.6 One distractor is thin signal — and slip is the clearest casualty
One distractor choice may map to more than one error type (a student could pick C
from a reasoning error *or* a content gap; more sharply, a **careless slip and a
content gap can select the identical distractor**). This is precisely why
`careless_slip` is **dropped from v1** (§3.2): with only distractor choice, it is
not separable, and forcing it would be guessing. In v1 such cases surface as the
best-semantic-fit label or as `abstain` → predict-and-confirm — the honest
behavior. This caps standalone v1 accuracy (the same "intent isn't recoverable
from the trace" ceiling the algebra spec names), and the eval must report that
ceiling honestly rather than chase it with a bigger model or a fabricated signal.

### 9.7 Generation drift and tag validity (the new primary risk)
Generic AI-generated MCQs **drift**: they lose the required schema, produce
degenerate/duplicate choices, mis-assign the correct answer, or attach a
misconception tag that the distractor does not actually embody. This is *the*
risk the generation thesis is about, so we **measure it directly**: V1 structural
validity is a deterministic anti-drift gate, and V3 tag-fidelity checks the tag
against an independent read. Low validity/fidelity is not hidden — it is the
signal that justifies (or refutes) the distilled small generator.

### 9.8 Verifier independence and human anchoring (generate-and-verify honesty)
The tag-fidelity verifier must be a **different model** than the generator (a
frontier model or the separately-trained tagger); a generator grading its own
output is circular and inflates fidelity. The harness enforces this (it reports
tag-fidelity as **N/A** rather than let the generator self-grade). Because the
verifier is itself a model, its trust must be **anchored by a small human-labeled
sample (~30–50 items)** reporting verifier↔human agreement. That sample is a
described requirement, **not** a pipeline — see §12.6.

### 9.9 Measurement now vs. production pipeline later
We are **not** building a production rejection-sampling / data-cleaning pipeline
in v1. We are building the **measurement** that shows generation beats baseline
(§12.3). The solvability "solve-it-blind" solver arm (V2) and the scaled
rejection-sampling + scaled-human-review pipeline are explicitly **prod / future
work** (§12.4). Conflating the measurement with the pipeline would over-scope v1.

---

## 10. Decisions

### 10.1 Resolved by the owner
1. **Item bank strategy (§9.1, §9.2) — RESOLVED: synthetic-train / real-eval.**
   Pull real AP Bio / MCAT-style items; generate synthetic *training* items seeded
   on them (distractors seeded from the real items' real distractors, for
   pedagogical validity); use the real items **exclusively** as the eval set.
   `generate_dataset.py` becomes a real-seeded synthesizer; `real_data.py` handles
   both the seed import and the eval set.
2. **Behavioral signal / timing (§2.2, §9.4) — RESOLVED: distractor choice for v1;
   timing deferred.** v1 error-typing runs on distractor choice alone (a
   model-layer behavioral signal that satisfies SPOV2). Timing is a Phase-2+
   enhancement contingent on real instrumented data, and **will not be fabricated
   or simulated.** Consequence: `careless_slip` is **dropped from the v1 taxonomy**
   and returns only with real timing / repeated-attempt data.

### 10.2 Still open (recommendations noted)
3. **SLM vs. lookup scope (§9.3):** commit to running the litmus test and accepting
   its verdict on where the SLM is actually justified (likely generalizing tags to
   real/new items + distribution output), rather than assuming it. *Recommendation:
   run it in Phase 1 before over-investing in the SLM.*
4. **Domain confirmation:** AP Bio as the MCAT proxy — confirm the 3-4 seed topics
   (proposed: cellular respiration, classical/molecular genetics, membrane
   transport, + one experimental-design/data topic) and the real-item source
   (open-licensed AP Bio banks / Eedi-style, **not** copyrighted MCAT items)
   before pulling the seed set.

---

## 11. What explicitly does NOT change

- The base model, QLoRA/Unsloth training path, and "data is the deliverable"
  philosophy.
- The calibration algorithm (temperature scaling + threshold).
- The eval philosophy (base-vs-tuned; behavior over capability; litmus test first;
  kappa gate before splitting the taxonomy).
- The abstain→defer primitive (now predict-and-confirm).
- The existing algebra project, which remains the intact, working reference
  implementation this pivot is measured against.

---

## 12. Primary v1 task update (mentor-endorsed): conditional generation of tagged items

This section supersedes the "what the SLM *does*" framing of §2 for v1. It is
additive and consistent with every §10 decision (synthetic-train/real-eval,
distractor-choice signal, timing deferred, mid-grained taxonomy). The tagging
engine of §2 is **not discarded** — it is reused as the verifier (§12.5).

### 12.1 The task: conditional spec → tagged item (NOT free generation)

The primary v1 SLM performs **conditional generation**:

> **Input:** a spec `{topic, target misconception(s) per distractor, difficulty,
> format}`.
> **Output:** a full, well-formed AP Bio item in the `apbio_item_template` schema —
> stem (and passage if the format calls for it), N answer choices, exactly one
> correct answer, and a `distractor_tags` map assigning every wrong choice to a
> misconception id from `data/apbio_misconceptions.json`.

It is **conditional, not free**: the recommendation layer (§5) needs items that
target a *specific* misconception a student is struggling with, and conditioning
on the target is also what makes eval **checkable** (we know what the item was
supposed to be, so we can verify it). Free "make me a bio question" generation is
neither what the product needs nor measurable.

### 12.2 Thesis shift: from calibrated abstention to controllable, anti-drift generation

The algebra project's thesis was **calibrated abstention** — schema-faithfulness,
consistency, knowing when to defer. The generation task trades that for a
**controllable, anti-drift structured-generation thesis**:

> Generic AI-generated questions **drift** — they lose the schema and, worse, lose
> *valid tags* (the distractor no longer embodies the misconception it claims). A
> fine-tuned small model can produce **reliably-schema'd, validly-tagged** items on
> demand for a requested misconception. The behavior we train and measure is
> *controllability + reliability of the tag*, not calibrated deferral.

This is still **"behavior from data, not raw capability"** — the same north star —
but the *behavior* is different (produce a valid tagged item to spec, every time)
and so are the metrics (§12.7).

### 12.3 The generate-and-verify loop

The tagging work and the generation work **compose**:

```
   spec {topic, target misconception(s), difficulty, format}
        │
        ▼
  ┌───────────────┐     generated item (stem, choices, correct, distractor_tags)
  │  GENERATOR    │ ───────────────────────────────────────────────┐
  │  (the SLM)    │                                                 │
  └───────────────┘                                                 ▼
        ▲                                          ┌────────────────────────────┐
        │                                          │ V1 structural validity     │  deterministic
        │                                          │  (schema, 1 correct, tags) │  [anti-drift]
        │                                          ├────────────────────────────┤
        │                                          │ spec-adherence             │  topic + targets
        │                                          ├────────────────────────────┤
        │                                          │ V3 tag-fidelity            │  INDEPENDENT tagger
        │                                          │  (litmus_tagging.predict_  │  re-reads distractor,
        │                                          │   tag, a DIFFERENT model)  │  agrees with claim? [crux]
        │                                          └────────────────────────────┘
        │                                                          │
        └──────────────────  metrics: validity / fidelity / yield ◄┘
```

The verifier (V3) is the **tagger from §2**, used as an *independent read* on the
generator's output. That is the whole point of building the tagging litmus first:
it is now the instrument that checks whether generated tags are real.

### 12.4 Two USES: measurement now vs. production pipeline later

The same generate-and-verify primitive is used two ways, and v1 builds only the
first:

1. **Measurement (v1, built now — `scripts/litmus_generation.py`).** Run the
   verifiers *as the metric* to answer: does generation beat baseline? Which arm
   (base 1.7B vs prompted frontier) produces valid, faithfully-tagged items? This
   is a measurement harness, **not** a data factory.
2. **Production pipeline (future, described only).** The same verifiers wrapped in
   **rejection sampling** (keep only valid∧faithful items) + **scaled human
   review** to mass-produce a training/serving item bank. This is explicitly
   **out of v1 scope**.

**Explicitly deferred to prod/future (documented, NOT built for v1):**
- **V2 solvability ("solve-it-blind").** A solver model attempts the item without
  the answer key to confirm exactly one defensible correct answer. Deferred.
- The full **rejection-sampling + scaled-human** data pipeline. Deferred.

### 12.5 Relationship to the tagging work

`scripts/litmus_tagging.py` (the distractor→misconception tagger) is **reframed as
the independent tag-fidelity verifier**. `predict_tag(model, item, chosen,
misconceptions)` is the reusable entry point the generation harness calls. Nothing
about the tagging design changes; it simply gets a second job. Generate-and-verify
= *generator proposes, an independent tagger disposes.*

### 12.6 Non-negotiable: a small human-anchored fidelity sample

The tag-fidelity metric is only as trustworthy as the verifier. Therefore a
**small human-labeled sample of ~30–50 generated items** must be double-checked by
a person, and we report **verifier↔human agreement** on that sample. This anchors
the automated fidelity number to human judgment. It is a **small sample to
describe and collect, not a pipeline to build** — the same discipline as the
algebra project's abstain-review export (§6).

### 12.7 Metrics (per arm)

Computed by `scripts/litmus_generation.py` for each generator arm:

- **Validity rate (V1):** fraction of generated items that are structurally valid
  (well-formed schema, exactly one correct, N distractors, every distractor tagged
  with a real taxonomy id, no duplicate/degenerate choices). *The anti-drift metric.*
- **Spec-adherence rate:** fraction whose topic + target misconception(s) match the
  spec.
- **Tag-fidelity rate (V3):** fraction of generated distractors whose independent
  verifier tag agrees with the generator's claimed tag. *The crux metric.*
- **Diversity / dedup rate:** fraction of unique (non-duplicate) stems — guards
  against mode collapse.
- **Yield:** `valid ∧ faithful ÷ generated` — the bottom-line usable-item rate.

### 12.8 Distillation framing (the arms)

- **Base Qwen3-1.7B (candidate SLM):** expected to **drift** — malformed JSON,
  broken schema, invalid/hallucinated tags. If so, it fails validity/fidelity.
- **Prompted frontier model (the teacher / bar to beat):** expected to be decent at
  producing valid, faithfully-tagged items zero/few-shot. It sets the target.
- **Fine-tuned small model (the deliverable):** a deployable model **distilled from
  the frontier teacher** that generates valid tagged items reliably where base-small
  can't. Its value proposition is **controllability + reliability + cost/scale**
  (cheap, on-device / at-scale generation), not raw capability the frontier lacks.

**Decision gate:** if base-small already generates valid, faithful items → no
capability project (value is deployment efficiency only). If base-small drifts but
the frontier is reliable → **distill** into the small generator (the pivot). If
neither is reliable → the schema/taxonomy/task needs rework before training.

### 12.9 Independence + honesty caveats (restated)

- The V3 verifier is a **different model** than the generator — never self-grading
  (the harness reports N/A instead of faking it).
- The ~30–50-item **human sample anchors** verifier trust (§12.6).
- The generation litmus ships with **authored PLACEHOLDER specs**; defensible
  numbers require a **GPU** (base 1.7B), an optional **frontier API key** (teacher
  arm and/or frontier verifier), and the human-anchored fidelity sample.
