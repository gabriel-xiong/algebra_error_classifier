"""
Persistent, free demo on Modal (free-tier credits; scales to zero when idle).

HF free Spaces no longer host Gradio apps (they require PRO), so this deploys the
same interactive demo on Modal instead — a permanent public URL, free for low
traffic. Self-contained: fetches the misconception list from GitHub at startup,
loads base Qwen3-1.7B + your LoRA adapter, and serves a Gradio UI.

Setup (once, in a terminal):
    pip install modal
    modal token new            # opens a browser to link your free Modal account

Deploy:
    modal deploy scripts/modal_app.py
    # prints a permanent https URL (e.g. https://<you>--apbio-item-generator-ui.modal.run)

Run locally for a quick check:
    modal serve scripts/modal_app.py
"""

import modal

MODEL_ID = "gabriel-xiong/apbio-item-generator-qwen3-1.7b-lora"  # your adapter (or merged)
ADAPTER = True                                                    # False if MODEL_ID is merged
BASE_ID = "Qwen/Qwen3-1.7B"
MISC_URL = ("https://raw.githubusercontent.com/gabriel-xiong/QuestionGen/main/"
            "data/apbio_misconceptions.json")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "torch", "gradio", "peft", "huggingface_hub", "requests")
)
app = modal.App("apbio-item-generator")

SYSTEM_PROMPT = (
    "You are an AP Biology item writer. You generate multiple-choice questions in "
    "which every wrong answer is a deliberate, named misconception. You output "
    "ONLY a single JSON object and nothing else: no prose, no explanation, no "
    "markdown code fences, before or after the JSON."
)


@app.function(image=image, cpu=2.0, memory=16384, timeout=900,
              scaledown_window=300)  # stays warm 5 min after use, then scales to zero (free)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def ui():
    import json, re, requests
    import gradio as gr
    import torch
    from fastapi import FastAPI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MISC = {m["id"]: m for m in requests.get(MISC_URL, timeout=30).json()["misconceptions"]}
    TOPICS = sorted({m["topic"] for m in MISC.values()})

    if ADAPTER:
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(BASE_ID)
        model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype="auto")
        model = PeftModel.from_pretrained(model, MODEL_ID)
    else:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    model.eval()

    def build_prompt(topic, ids):
        lines = "\n".join(f'  - {i} ("{MISC[i]["name"]}"): {MISC[i]["description"]}' for i in ids)
        note = ('\n- Because this is genetics, also include a "spec" field with the cross.'
                if topic == "genetics" else "")
        user = (f"Write one AP Biology multiple-choice item on the topic: {topic}.\n\n"
                f"Embed EXACTLY these misconceptions, one per wrong answer choice:\n\n{lines}\n\n"
                f'Output a single JSON object with fields stem, choices (A-D), correct, and '
                f'distractor_tags (each wrong letter -> {{"misconception_id": ...}}).{note}\n\n'
                "Output only the JSON object.")
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def generate(topic, ids, temperature):
        if not ids:
            return "Pick some misconceptions first.", ""
        ids = ids[:3]
        try:
            text = tok.apply_chat_template(build_prompt(topic, ids), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(build_prompt(topic, ids), tokenize=False,
                                           add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=512, do_sample=temperature > 0,
                             temperature=temperature or None, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return "⚠️ no valid JSON:\n\n" + raw, raw
        item = json.loads(m.group(0))
        rows = [f"**{item.get('stem','')}**", ""]
        tags = item.get("distractor_tags", {})
        for L, c in item.get("choices", {}).items():
            if L == item.get("correct"):
                rows.append(f"- **{L}. {c}**  ✅ correct")
            else:
                t = tags.get(L, {})
                mid = t.get("misconception_id", "?") if isinstance(t, dict) else t
                rows.append(f"- {L}. {c}  — *{mid}*")
        return "\n".join(rows), json.dumps(item, indent=2)

    def misc_choices(topic):
        ids = sorted(i for i, m in MISC.items() if m["topic"] == topic)
        return gr.update(choices=ids, value=ids[:3])

    with gr.Blocks(title="AP Bio Misconception-Tagged Item Generator") as demo:
        gr.Markdown("# AP Bio Misconception-Tagged Item Generator\n"
                    "A fine-tuned Qwen3-1.7B that writes MCQs where **every distractor is a "
                    "named misconception**. Pick a topic and up to 3 misconceptions.")
        with gr.Row():
            topic = gr.Dropdown(TOPICS, value=TOPICS[0], label="Topic")
            temp = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
        ids = gr.CheckboxGroup(label="Misconceptions to embed (pick 3)")
        btn = gr.Button("Generate item", variant="primary")
        out_md = gr.Markdown()
        out_json = gr.Code(label="Raw JSON", language="json")
        topic.change(misc_choices, topic, ids)
        demo.load(misc_choices, topic, ids)
        btn.click(generate, [topic, ids, temp], [out_md, out_json])

    web = FastAPI()
    return gr.mount_gradio_app(web, demo, path="/")
