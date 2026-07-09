# Real-data evaluation harness

Synthetic data is used for training *volume*, but injected errors may not match the
real distribution of student mistakes (the known weakness in `spec.md`). This harness
lets you evaluate the base and fine-tuned models on a small set of **real** student
errors mapped into our 7-label taxonomy — the "does this generalize off synthetic?"
check.

Pattern: **synthetic train / real eval.**

## Where to get real data

| Source | What it is | License | Fit |
|--------|-----------|---------|-----|
| [Eedi – Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data) | 1,868 real diagnostic MCQs; each distractor tagged to a specific misconception (~2,586 categories) | CC BY-NC 4.0 (cite, non-commercial) | Real misconception signal; MCQ only, no worked steps. Filter to linear-equation items. |
| [PSLC DataShop](https://pslcdatashop.web.cmu.edu/) — Cognitive Tutor Algebra ("Algebra I 2005‑2006", Bridge to Algebra, KDD Cup 2010) | Real step-by-step algebra tutor logs; step outcomes incl. `BUG` = diagnosed misconception | Account required; per-dataset terms | Closest to our schema (worked steps). Needs reshaping from transaction logs. |
| [PRM800K](https://github.com/openai/prm800k) | 800k step-level correctness labels on MATH solutions; `neutral` label ≈ our `abstain` | MIT | Precedent for step-level + ambiguity, not a direct fit (LLM solutions, competition math). |

Because none matches our exact schema, the workflow is: **harvest real problem stems +
real wrong answers, then hand-map a few hundred into the taxonomy.**

## Schema

Each row (raw dump) can use varied field names — `real_data.py` normalizes them.
Recognized aliases:

- problem: `problem`, `question`, `question_text`, `prompt`, `stem`, `equation`
- correct_answer: `correct_answer`, `correct`, `answer`, `correct_option`, `solution`
- student_answer: `student_answer`, `distractor`, `chosen`, `response`, `selected_answer`
- student_work: `student_work`, `work`, `steps`, `working`, `rationale`, `reasoning`
- label (explicit taxonomy label): `label`, `gold`, `gold_label`, `taxonomy_label`
- misconception (free text, gets mapped): `misconception`, `misconception_name`, `bug`, `bug_message`, `diagnosis`
- id: `id`, `question_id`, `row_id`, `uid`

See `data/real_eval_template.jsonl` for examples. Provide **either** an explicit `label`
**or** a free-text `misconception` (mapped via keywords). Rows whose misconception maps
to nothing are defaulted to `abstain` and flagged `"needs_review": true` for a human.

## Workflow

```bash
# 1. Normalize a raw real dump into the taxonomy schema (reports unmapped rows).
python scripts/real_data.py --in data/real_raw.jsonl --out data/real_eval.jsonl

# 2. Hand-review rows flagged needs_review; set a correct label or keep abstain.

# 3. Evaluate base vs tuned on the real set (add --normalize-real to skip step 1).
python scripts/run_baseline.py --model Qwen/Qwen3-1.7B \
    --data data/real_eval.jsonl --score-labels --calibration outputs/calibration.json
python scripts/run_baseline.py --model Qwen/Qwen3-1.7B --adapter outputs/lora \
    --data data/real_eval.jsonl --score-labels --calibration outputs/calibration.json
```

The keyword mapping in `real_data.py` (`MISCONCEPTION_KEYWORDS`) is a **starting point**,
not ground truth — always review before reporting numbers.

---

## AP Bio / MCAT pivot — real EVAL-ONLY item pool

For the MCAT pivot the item bank is **synthetic-train / real-eval**
(`docs/mcat_pivot_spec.md` §9.1, §10.1): real, openly-licensed, exam-style bio MCQs
are reserved **exclusively** as the held-out eval set and are **never trained on**.
`scripts/fetch_real_bio_items.py` acquires that pool and normalizes it into the AP Bio
item schema (`data/apbio_item_template.jsonl`), mirroring the conventions here
(field aliases, best-effort taxonomy mapping, provenance).

### Sources used (openly licensed — we do NOT scrape College Board AP or AAMC MCAT)

| Source (HF) | What it is | License | Fit |
|-------------|-----------|---------|-----|
| [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) — `high_school_biology` + `college_biology` | Clean 4-option single-best-answer MCQs with an answer key | **MIT** | Closest to AP-Bio content; the priority source |
| [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) (biology-filtered) | Crowdsourced science MCQs: correct answer + 3 distractors + a **support passage** | **CC BY-NC 3.0** (non-commercial; cite) | Adds passage-bearing items; filtered here to biology-relevant stems |
| [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) — `ARC-Challenge` + `ARC-Easy` (biology-filtered) | Grade-school science MCQs with answer key | **CC BY-SA 4.0** (share-alike; cite) | Lowest-priority fallback; filtered to biology-relevant stems |

Confirm licenses upstream before any redistribution; **CC BY-NC 3.0 (SciQ) is
non-commercial** and **CC BY-SA 4.0 (ARC) is share-alike** — honor both if this pool
is ever published.

### What the acquired pool is (and is NOT)

`data/real_bio_eval_raw.jsonl` is a **RAW, UNTAGGED eval pool**. Each row follows the
apbio schema (`id`, `topic`, `passage`, `stem`, `choices` as an A/B/C/D dict, `correct`
letter, `correct_answer` text, `distractor_tags`, `provenance: "real_eval"`, `source`,
`source_license`, `authoring`). Every **wrong** option carries
`{"tag": null, "needs_tagging": true}` because these static items have a correct answer
but **NO student-chosen distractor** and **NO misconception tags**.

`topic` is a **best-effort** keyword map onto the 7 taxonomy topics in
`data/apbio_misconceptions.json`; stems that don't match land on `"unmapped"` for a
human to review or drop (use `--require-mapped-topic` / `--topics` to filter).

### Reality flags (read before using as eval)

- **UNTAGGED.** No per-distractor misconception, no student choice, no timing. This is a
  raw candidate pool, not a gold eval set.
- **Distractor-choice error-typing needs a student-selected distractor**, which these
  static items don't have. So for eval the protocol is to treat **each wrong option as a
  potential chosen distractor to be tagged**.
- **Real = eval only.** These items must never enter training (spec §9/§10).

### NEXT STEP (required before this is a usable gold eval set)

1. **Draft** per-distractor misconception tags (frontier-assisted): for each wrong
   option, assign an `error_type` from the v1 taxonomy (`content_gap`,
   `reasoning_error`, `misread_or_passage_mapping`) + a `misconception` id from
   `data/apbio_misconceptions.json`, treating each wrong option as a potential chosen
   distractor.
2. **HUMAN-verify ~40–50** of these tagged items (prioritize the topic-mapped rows) to
   form the **gold eval set** — the same discipline as the abstain-review /
   double-coding gate (`docs/mcat_pivot_spec.md` §8, kappa ≥ 0.7).
3. Only then evaluate base-vs-tuned models against `distractor_tags[chosen].error_type`.

### Frontier-assisted tag drafting (`scripts/draft_tags.py`)

Step 1 above is implemented by `scripts/draft_tags.py`. It filters the raw pool
to the AP-Bio-mapped core (topic in the 7 taxonomy topics, i.e. NOT `unmapped`),
stratified-samples ~80 items across those topics (per-topic cap so small topics
like enzymes / experimental_design are represented while genetics / evolution
stay largest), and for each **wrong** option asks a frontier model to draft a
best-fit misconception `id`, a one-line `rationale`, a `confidence`, and a
`no_fit` flag (real distractors do not always map — that is expected, useful
coverage signal). It writes two side-by-side outputs:

- `data/real_bio_eval_drafted.jsonl` — raw schema, each wrong option's
  `distractor_tags` entry filled with `{tag, rationale, confidence, no_fit,
  needs_review: true, tag_source: "frontier_draft"}`; correct option stays
  untagged; `provenance` stays `real_eval`.
- `data/real_bio_eval_worksheet.csv` — one row per item×wrong-distractor with the
  draft plus BLANK `confirmed(Y/N)` / `corrected_misconception_id` / `notes`
  columns for the human (owner) to fill in. The verified rows (~40–50) become the
  gold eval set.

The API key is loaded from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or a gitignored
`.local/.env` / `.env` (provider auto-detected) and is **never printed, logged,
or committed**. With no key the script still writes an empty-draft worksheet and
prints a "set the key and re-run" message. `--selftest` runs an offline dummy
drafter to verify the plumbing without a key or network.

```bash
python scripts/draft_tags.py --selftest        # offline plumbing smoke test
python scripts/draft_tags.py --n 80 --seed 0    # real drafts (needs a key)
```

### Workflow

```bash
# Acquire + normalize (all three sources, ~a few hundred items):
python scripts/fetch_real_bio_items.py --out data/real_bio_eval_raw.jsonl

# Only MMLU, mapped-topic items only, capped:
python scripts/fetch_real_bio_items.py --sources mmlu --require-mapped-topic --max-items 200

# Filter to specific taxonomy topics:
python scripts/fetch_real_bio_items.py --topics genetics evolution membrane_transport
```

If the HF `datasets` download is constrained, the script reports the failing source and
continues with whatever succeeded — it **never fabricates items**.
