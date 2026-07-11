"""
Gradio inference demo for the misconception-tagged AP Bio item generator.

Runs as a Hugging Face Space (or locally / in Colab with share=True). Loads the
MERGED model from the Hub with plain transformers (no Unsloth needed).

Space setup: create a Space (SDK: Gradio), and add three files:
  - app.py                     (this file)
  - requirements.txt           (transformers, torch, gradio)
  - data/apbio_misconceptions.json  (the misconception definitions)
Set MODEL_ID below to your merged-model repo.
"""

import json
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your fine-tuned repo. If it is a LoRA ADAPTER (repo name ends in -lora), leave
# ADAPTER=True so the base is loaded and the adapter applied on top (works on a
# free CPU Space). If you pushed a MERGED model, set ADAPTER=False.
MODEL_ID = "gabriel-xiong/apbio-item-generator-qwen3-1.7b-lora"
ADAPTER = True
BASE_ID = "Qwen/Qwen3-1.7B"

SYSTEM_PROMPT = (
    "You are an AP Biology item writer. You generate multiple-choice questions in "
    "which every wrong answer is a deliberate, named misconception. You output "
    "ONLY a single JSON object and nothing else: no prose, no explanation, no "
    "markdown code fences, before or after the JSON."
)

MISC = {m["id"]: m for m in json.loads(
    Path("data/apbio_misconceptions.json").read_text(encoding="utf-8"))["misconceptions"]}
TOPICS = sorted({m["topic"] for m in MISC.values()})

print("loading model…")
if ADAPTER:
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype="auto",
                                                 device_map="auto")
    model = PeftModel.from_pretrained(model, MODEL_ID)
else:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto",
                                                 device_map="auto")
model.eval()


def _prompt(topic, ids):
    lines = "\n".join(f'  - {i} ("{MISC[i]["name"]}"): {MISC[i]["description"]}'
                      for i in ids)
    note = ""
    if topic == "genetics":
        note = ('\n- Because this is genetics, also include a "spec" field with the '
                "cross, and make the keyed answer the correct fraction.")
    user = (f"Write one AP Biology multiple-choice item on the topic: {topic}.\n\n"
            f"Embed EXACTLY these misconceptions, one per wrong answer choice:\n\n"
            f"{lines}\n\nOutput a single JSON object with fields stem, choices "
            f'(A-D), correct, and distractor_tags (each wrong letter -> '
            f'{{\"misconception_id\": ...}}).{note}\n\nOutput only the JSON object.')
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate(topic, ids, temperature):
    if not ids:
        return "Pick some misconceptions first.", ""
    ids = ids[:3]
    msgs = _prompt(topic, ids)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=512,
                         do_sample=temperature > 0, temperature=temperature or None,
                         pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    item = _extract_json(raw)
    if not item:
        return "⚠️ model did not return valid JSON:\n\n" + raw, raw
    # Pretty render: question, choices with the misconception tag on each distractor.
    lines = [f"**{item.get('stem','')}**", ""]
    tags = item.get("distractor_tags", {})
    for L, c in item.get("choices", {}).items():
        if L == item.get("correct"):
            lines.append(f"- **{L}. {c}**  ✅ correct")
        else:
            mid = tags.get(L, {}).get("misconception_id", "?") if isinstance(tags.get(L), dict) else tags.get(L, "?")
            lines.append(f"- {L}. {c}  — *{mid}*")
    return "\n".join(lines), json.dumps(item, indent=2)


def misc_choices(topic):
    ids = sorted(i for i, m in MISC.items() if m["topic"] == topic)
    return gr.update(choices=ids, value=ids[:3])


with gr.Blocks(title="AP Bio Misconception-Tagged Item Generator") as demo:
    gr.Markdown("# AP Bio Misconception-Tagged Item Generator\n"
                "A fine-tuned Qwen3-1.7B that writes MCQs where **every distractor "
                "is a named misconception**. Pick a topic and up to 3 misconceptions.")
    with gr.Row():
        topic = gr.Dropdown(TOPICS, value=TOPICS[0], label="Topic")
        temp = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
    ids = gr.CheckboxGroup(label="Misconceptions to embed (pick 3)")
    btn = gr.Button("Generate item", variant="primary")
    out_md = gr.Markdown(label="Item")
    out_json = gr.Code(label="Raw JSON", language="json")

    topic.change(misc_choices, topic, ids)
    demo.load(misc_choices, topic, ids)
    btn.click(generate, [topic, ids, temp], [out_md, out_json])

if __name__ == "__main__":
    demo.launch()  # add share=True to get a public link from Colab/local
