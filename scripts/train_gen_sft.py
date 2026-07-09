"""
QLoRA SFT for the GENERATION behavior (docs/behavior_spec.md), via Unsloth.

Teaches a small base model to turn a generation spec (topic + target
misconceptions) into one valid JSON MCQ. Mirrors train_sft.py (the algebra
classifier trainer) but adds COMPLETION-ONLY masking: loss is computed only on
the assistant JSON, not the prompt, so the model learns to EMIT the item rather
than echo the instructions — which matters for reliable structured output.

Run on Colab / RunPod (single GPU). Data comes from build_dataset.py:
  python train_gen_sft.py \
      --train ../data/gen_sft_train.jsonl \
      --val   ../data/gen_sft_val.jsonl \
      --model Qwen/Qwen3-1.7B --out ../outputs/gen_lora

Then evaluate base vs tuned:
  python eval_generation.py --base hf:Qwen/Qwen3-1.7B --tuned hf:../outputs/gen_lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bio import format_chat, load_jsonl  # noqa: E402


def load_messages(path):
    return [row["messages"] for row in load_jsonl(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", default="../data/gen_sft_train.jsonl")
    ap.add_argument("--val", default="../data/gen_sft_val.jsonl")
    ap.add_argument("--out", default="../outputs/gen_lora")
    ap.add_argument("--max-seq-length", type=int, default=1536)  # long trihybrid stems + JSON
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if "Instruct" in args.model:
        raise SystemExit("Qwen/Qwen3-1.7B-Instruct does not exist; use Qwen/Qwen3-1.7B "
                         "(it already carries a chat template).")

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "Unsloth stack not installed. In Colab:\n"
            "  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'\n"
            "  pip install trl datasets") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    def to_ds(path):
        texts = [format_chat(tokenizer, m) for m in load_messages(path)]
        return Dataset.from_dict({"text": texts})

    train_ds, val_ds = to_ds(args.train), to_ds(args.val)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ta_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, logging_steps=10, save_strategy="no",
        seed=args.seed, report_to="none", fp16=not use_bf16, bf16=use_bf16)
    try:
        targs = TrainingArguments(**ta_kwargs, eval_strategy="epoch")
    except TypeError:
        targs = TrainingArguments(**ta_kwargs, evaluation_strategy="epoch")

    kw = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=targs,
              dataset_text_field="text", max_seq_length=args.max_seq_length)
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **kw)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **kw)

    # Completion-only: mask everything up to and including the assistant header,
    # so loss falls only on the generated JSON. Qwen3 chat markers below.
    try:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n")
    except Exception as exc:  # helper/version mismatch -> fall back to full-text SFT
        print(f"[warn] train_on_responses_only unavailable ({exc}); "
              "training on full text instead.")

    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"base_model": args.model, "train": args.train, "val": args.val,
                   "epochs": args.epochs, "lr": args.lr,
                   "completion_only": True}, fh, indent=2)
    print(f"Saved LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()
