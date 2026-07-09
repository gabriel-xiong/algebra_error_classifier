"""Model loading, label scoring, and generation helpers."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch

from common import LABELS, SYSTEM_PROMPT, build_user_prompt, format_chat, parse_label


class DummyModel:
    """Fake classifier for testing the scoring pipeline only."""

    def __init__(self, data):
        self._answers = {example["id"]: example["label"] for example in data}
        self._counter = 0

    def generate(self, system, user):
        match = re.search(r"Problem: (.+)", user)
        self._counter += 1
        import random

        rng = random.Random(hash(match.group(1)) + self._counter)
        pick = rng.choice(LABELS)
        roll = rng.random()
        if roll < 0.15:
            return f"The student made a {pick}."
        if roll < 0.25:
            return "not_a_real_label"
        return pick

    def score_labels(self, system, user, temperature=1.0):
        raw = self.generate(system, user)
        label, _ = parse_label(raw)
        scores = {item: -20.0 for item in LABELS}
        if label:
            scores[label] = 0.0
        return scores, label or LABELS[0], 0.5


class HFClassifier:
    """Hugging Face backend with generation and label log-prob scoring."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        adapter_path: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return format_chat(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
        )

    def generate(self, system: str, user: str) -> str:
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
        )
        generated = output[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.inference_mode()
    def score_labels(self, system: str, user: str, temperature: float = 1.0) -> tuple[dict, str, float]:
        """Return label log-scores, top label, and calibrated top confidence."""
        prompt = self._build_prompt(system, user)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        temp = max(temperature, 1e-5)

        prefill = self.model(input_ids=prompt_ids, use_cache=True)
        first_logits = prefill.logits[:, -1, :]
        base_past = prefill.past_key_values

        log_scores: dict[str, float] = {}
        for label in LABELS:
            label_ids = self.tokenizer(label, add_special_tokens=False).input_ids
            total = 0.0
            past = base_past
            logits = first_logits
            for idx, token_id in enumerate(label_ids):
                log_probs = torch.log_softmax(logits / temp, dim=-1)
                total += log_probs[0, token_id].item()
                if idx + 1 < len(label_ids):
                    next_token = torch.tensor([[token_id]], device=prompt_ids.device)
                    step = self.model(
                        input_ids=next_token,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = step.past_key_values
                    logits = step.logits[:, -1, :]
            log_scores[label] = total

        max_log = max(log_scores.values())
        probs = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        norm = sum(probs.values())
        probs = {label: value / norm for label, value in probs.items()}
        top_label = max(probs, key=probs.get)
        return log_scores, top_label, probs[top_label]


def load_calibration(path: str | Path | None) -> dict:
    if not path:
        return {"temperature": 1.0, "abstain_threshold": None}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def apply_abstention(pred_label: str, confidence: float, calibration: dict) -> str:
    threshold = calibration.get("abstain_threshold")
    if threshold is None:
        return pred_label
    if confidence < threshold and pred_label != "abstain":
        return "abstain"
    return pred_label
