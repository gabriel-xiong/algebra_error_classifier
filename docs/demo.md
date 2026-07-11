# Demo & Hosting Guide

Everything to record the demo video and host a persistent inference demo.

---

## Step 0 — Push the MERGED model to the Hub (do this once, in Colab)
The Space and the interactive demo load with plain `transformers`, so push a
merged model (not just the LoRA adapter):

```python
# model, tok are in memory after training; if not, reload:
# from unsloth import FastLanguageModel
# model, tok = FastLanguageModel.from_pretrained('outputs/gen_lora', max_seq_length=1536, load_in_4bit=True)
from huggingface_hub import login
login()   # paste your hf_... WRITE token into the hidden box
model.push_to_hub_merged('gabriel-xiong/apbio-item-generator-qwen3-1.7b',
                         tok, save_method='merged_16bit')
```

---

## A) Video demo — run in Colab (on the GPU)

### A1. The money shot: base vs tuned on the SAME prompt
```python
import json, sys; sys.path.insert(0, 'scripts')
import gen_spec
from unsloth import FastLanguageModel

sc = json.loads(open('data/eval_scenarios_ood.jsonl').readline())   # an unseen topic
system, user = gen_spec.build_generation_prompt(sc['topic'], sc['misconception_ids'])
msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

def run(path):
    m, t = FastLanguageModel.from_pretrained(path, max_seq_length=1536, load_in_4bit=True)
    FastLanguageModel.for_inference(m)
    txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = t(txt, return_tensors='pt').to(m.device)
    out = m.generate(**ids, max_new_tokens=512, do_sample=False, pad_token_id=t.eos_token_id)
    return t.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

print("=== BASE Qwen3-1.7B ===\n", run('Qwen/Qwen3-1.7B'))
print("\n=== TUNED ===\n", run('outputs/gen_lora'))
```
On camera: base drifts (prose / invalid JSON / mis-tagged distractors); tuned emits
one clean tagged JSON item. Then narrate the 2×2 results table from the brainlift.

### A2. Interactive UI with a public share link
```python
!pip install -q gradio
!cd /content/algebra_error_classifier && sed -i 's/demo.launch()/demo.launch(share=True)/' scripts/app.py
!cd /content/algebra_error_classifier && python scripts/app.py
```
Prints a public `https://…gradio.live` URL (lasts ~72h). Click it, pick a topic +
misconceptions, hit Generate on screen. Set `MODEL_ID` in `scripts/app.py` to your
merged Hub repo first (or point it at a local merged dir).

---

## B) Persistent demo — Hugging Face Space

1. huggingface.co → **New → Space** → SDK **Gradio** → name it (e.g. `apbio-item-generator`).
2. Add three files to the Space repo:

**`app.py`** — copy from `scripts/app.py` in this repo. Set `MODEL_ID` to your merged model.

**`requirements.txt`**:
```
transformers
torch
gradio
```

**`data/apbio_misconceptions.json`** — copy from this repo (the app reads it for the topic/misconception menu).

3. The Space auto-builds and gives a live URL. Free CPU works for a 1.7B model
   (a few seconds per generation); choose a small GPU Space for snappier demos.

---

## Order of operations
1. Colab: `push_to_hub_merged(...)` → model on the Hub ✅
2. Record: A1 money-shot cell, then A2 Gradio share link ✅
3. Create the Space (3 files above), paste its URL into the submission ✅

## Submission checklist
- [ ] Dataset — published (this GitHub repo, `data/`)
- [ ] Model — on HF Hub (merged repo above)
- [ ] Running demo — HF Space URL
- [ ] Eval harness + results — `scripts/eval_generation.py`, tables in `docs/brainlift_generator.md`
- [ ] Brainlift — `docs/brainlift_generator.md`
- [ ] 3–5 min demo video
