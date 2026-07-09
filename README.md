# Algebra Error-Type Classifier

Fine-tune a small open model to classify why a student got a multi-step linear equation wrong, into a fixed 7-label taxonomy, with calibrated abstention.

Start with `docs/spec.md` for the full build spec and `docs/brainlift.md` for research grounding.

## Folder layout

```
algebra_error_classifier/
  README.md
  configs/experiment.yaml       experiment defaults
  docs/
    spec.md
    brainlift.md
  data/
    train.jsonl                 synthetic training (seed 0, never use for eval)
    val.jsonl                   synthetic validation (seed 1, calibration only)
    test_holdout.jsonl          synthetic held-out test (~500, seed 2) — main base vs tuned comparison
    testset.jsonl               12 hand-labeled spot-check (never train on this)
    train_sft.jsonl             chat format for Unsloth
    val_sft.jsonl
  notebooks/
    train_and_eval.ipynb        Colab GPU workflow (start here for training)
  scripts/
    common.py                   shared taxonomy + prompts
    metrics.py                    ECE + reliability diagram
    model_utils.py                HF backend + label scoring
    generate_dataset.py           forward error-injection generator
    prepare_sft.py                JSONL -> chat SFT format
    run_baseline.py               litmus + eval harness
    calibrate.py                  temperature scaling on val
    train_sft.py                  Unsloth QLoRA training
    run_local_prep.sh             generate data locally (CPU)
  outputs/                      lora adapter, calibration, eval plots
```

## Quick start

### Step 0 — local setup (CPU)

```bash
pip install -r requirements.txt
bash scripts/run_local_prep.sh
```

This generates `train.jsonl`, `val.jsonl`, `test_holdout.jsonl`, and the SFT chat files.

### Data splits

All three synthetic splits are carved from **one deduplicated pool** by
`make_splits.py`, so they share **zero problems** (generating per-seed instead
leaks ~69% of test problems into train). Regenerate with:

```bash
python scripts/make_splits.py --train-n 6000 --val-n 800 --test-n 1000 --seed 0 --out-dir data
```

| File | Size | Use |
|------|------|-----|
| `train.jsonl` | ~6,000 | QLoRA training |
| `val.jsonl` | ~800 | Calibration only (never report as "test accuracy") |
| `test_holdout.jsonl` | ~1,000 | **Primary eval** — compare base vs fine-tuned |
| `testset.jsonl` | 12 | Qualitative litmus / abstain spot-check |

Splits are disjoint and label-balanced; sizes vary by a few rows due to
per-label rounding during partitioning.

### Step 1 — baseline on held-out test (GPU)

```bash
python scripts/run_baseline.py \
  --model Qwen/Qwen3-1.7B \
  --data data/test_holdout.jsonl \
  --runs 1 \
  --out-dir outputs/eval/base_holdout
```

Optional quick litmus on hand-labeled cases:

```bash
python scripts/run_baseline.py \
  --model Qwen/Qwen3-1.7B \
  --data data/testset.jsonl \
  --runs 5 \
  --out-dir outputs/eval/base_litmus
```

### Step 2 — train + calibrate + eval (GPU: use Colab notebook)

Open `notebooks/train_and_eval.ipynb` in Google Colab:

1. Set `REPO` to your project path (upload to Drive or clone from GitHub)
2. Runtime -> **GPU**
3. Run all cells

The notebook runs: base litmus -> Unsloth QLoRA -> calibration -> tuned eval with ECE.

Or run manually on a GPU machine:

```bash
pip install -r requirements-colab.txt
python scripts/train_sft.py --model Qwen/Qwen3-1.7B \
  --train data/train_sft.jsonl --val data/val_sft.jsonl --out outputs/lora
python scripts/calibrate.py --model Qwen/Qwen3-1.7B \
  --adapter outputs/lora --data data/val.jsonl --out outputs/calibration.json
python scripts/run_baseline.py --model Qwen/Qwen3-1.7B \
  --adapter outputs/lora --data data/test_holdout.jsonl \
  --calibration outputs/calibration.json --runs 1 \
  --out-dir outputs/eval/tuned_holdout
# Optional ECE on a subset (score_labels is slow on 500 examples):
python scripts/run_baseline.py --model Qwen/Qwen3-1.7B \
  --adapter outputs/lora --data data/test_holdout.jsonl \
  --calibration outputs/calibration.json --score-labels --max-examples 100 \
  --out-dir outputs/eval/tuned_holdout_ece
```

### Optional — hard-negative mining (GPU)

Concentrate training on the cases the base model actually fails (the differentiation
thesis): score a pool with the base model, keep the wrong / overconfident-on-abstain
cases, and upweight them into an augmented training set.

```bash
python scripts/mine_hard_negatives.py --model Qwen/Qwen3-1.7B \
  --pool data/train.jsonl --out data/hard_negatives.jsonl \
  --augment-train data/train.jsonl --weight 3 --out-augmented data/train_hardaug.jsonl
python scripts/prepare_sft.py --data data/train_hardaug.jsonl --out data/train_hardaug_sft.jsonl
# then train_sft.py --train data/train_hardaug_sft.jsonl ...
```

### Optional — real-data eval (GPU)

Evaluate on real student errors (Eedi / PSLC DataShop) mapped into the taxonomy.
See `docs/real_data.md` for sourcing and the schema.

```bash
python scripts/run_baseline.py --model Qwen/Qwen3-1.7B \
  --data data/real_eval.jsonl --normalize-real --score-labels \
  --calibration outputs/calibration.json
```

## What each script does

| Script | Where to run | Purpose |
|---|---|---|
| `generate_dataset.py` | Local CPU | Synthetic labeled data (~35% abstain, edge-case enriched) |
| `make_splits.py` | Local CPU | Disjoint train/val/test_holdout from one deduped pool |
| `prepare_sft.py` | Local CPU | Convert to chat format for Unsloth |
| `mine_hard_negatives.py` | GPU | Find + upweight cases the base model fails |
| `real_data.py` | Local CPU | Normalize real dumps (Eedi/DataShop) into the taxonomy |
| `run_baseline.py` | GPU | Litmus + eval (accuracy, schema, consistency, ECE); `--normalize-real` for real data |
| `train_sft.py` | Colab GPU | QLoRA fine-tuning |
| `calibrate.py` | Colab GPU | Temperature scaling + abstain threshold |

## Outputs to keep for submission

- `data/train.jsonl` — the dataset artifact
- `outputs/lora/` — fine-tuned adapter
- `outputs/calibration.json` — temperature + abstain threshold
- `outputs/eval/tuned/reliability.png` — calibration plot
- Base vs tuned metrics from `run_baseline.py`

## Honesty flags

- Injected errors may not match real student mistakes; `testset.jsonl` (12 hand-labeled) is the qualitative reality check. `test_holdout.jsonl` is the quantitative base-vs-tuned comparison.
- Some BrainLift figures are flagged as unverified — confirm before citing.
