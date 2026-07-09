"""
QLoRA supervised fine-tuning via Unsloth.

Designed for Colab / RunPod with a single GPU. Run after generating train/val JSONL.

Example:
  python train_sft.py --train ../data/train_sft.jsonl --val ../data/val_sft.jsonl --out ../outputs/lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import format_chat, load_jsonl


def load_sft_dataset(path):
    rows = load_jsonl(path)
    return [row["messages"] for row in rows]


def formatting_func(tokenizer, examples):
    texts = []
    for messages in examples:
        texts.append(format_chat(tokenizer, messages))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train", default="../data/train_sft.jsonl")
    parser.add_argument("--val", default="../data/val_sft.jsonl")
    parser.add_argument("--out", default="../outputs/lora")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if "1.7B-Instruct" in args.model:
        raise SystemExit(
            "Invalid model id: Qwen/Qwen3-1.7B-Instruct does not exist on Hugging Face.\n"
            "Use: Qwen/Qwen3-1.7B"
        )

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "Unsloth stack not installed. In Colab run:\n"
            "  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'\n"
            "  pip install trl datasets"
        ) from exc

    train_messages = load_sft_dataset(args.train)
    val_messages = load_sft_dataset(args.val)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    train_texts = formatting_func(tokenizer, train_messages)
    val_texts = formatting_func(tokenizer, val_messages)
    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ta_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        # Unsloth reloads TRL; mid-run checkpoints fail to pickle SFTConfig.
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        fp16=not use_bf16,
        bf16=use_bf16,
    )
    try:
        training_args = TrainingArguments(**ta_kwargs, eval_strategy="epoch")
    except TypeError:
        training_args = TrainingArguments(**ta_kwargs, evaluation_strategy="epoch")

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "base_model": args.model,
        "train": args.train,
        "val": args.val,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
    }
    with open(out_dir / "train_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Saved LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()
