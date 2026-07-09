"""
Convert JSONL examples into chat-format JSONL for SFT / Unsloth.

Example:
  python prepare_sft.py --data ../data/train.jsonl --out ../data/train_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import build_sft_messages, load_jsonl, write_jsonl


def to_sft_rows(examples):
    rows = []
    for example in examples:
        rows.append(
            {
                "id": example["id"],
                "messages": build_sft_messages(example),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    examples = load_jsonl(args.data)
    rows = to_sft_rows(examples)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} SFT examples to {args.out}")


if __name__ == "__main__":
    main()
