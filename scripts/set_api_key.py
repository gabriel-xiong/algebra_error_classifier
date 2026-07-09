"""
Write an API key into the gitignored .local/.env that draft_tags.py / judge.py
already read. The secret is never printed, and .local/ is force-added to
.gitignore so the key cannot be committed.

Usage (run in YOUR terminal, not through the assistant):

  # from an env var already in your shell (no key in history):
  python scripts/set_api_key.py --provider openai --from-env OPENAI_API_KEY

  # paste directly (key stays local; shown masked back to you):
  python scripts/set_api_key.py --provider anthropic --key sk-ant-...

  # or pipe it in:
  echo "sk-..." | python scripts/set_api_key.py --provider openai --stdin

Provider must be "openai" or "anthropic"; it sets OPENAI_API_KEY or
ANTHROPIC_API_KEY accordingly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV_PATH = REPO / ".local" / ".env"
GITIGNORE = REPO / ".gitignore"
VAR = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _ensure_gitignored() -> None:
    line = ".local/"
    existing = GITIGNORE.read_text(encoding="utf-8").splitlines() if GITIGNORE.is_file() else []
    if line not in existing and ".local" not in existing:
        with open(GITIGNORE, "a", encoding="utf-8") as fh:
            fh.write(("" if not existing or existing[-1] == "" else "\n") + line + "\n")


def _read_env() -> dict:
    if not ENV_PATH.is_file():
        return {}
    out = {}
    for ln in ENV_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _mask(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]} (len {len(secret)})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--key", help="paste the key directly")
    src.add_argument("--from-env", help="name of an env var holding the key")
    src.add_argument("--stdin", action="store_true", help="read the key from stdin")
    args = ap.parse_args()

    if args.key:
        secret = args.key.strip()
    elif args.from_env:
        secret = (os.environ.get(args.from_env) or "").strip()
        if not secret:
            sys.exit(f"env var {args.from_env} is empty or unset")
    else:
        secret = sys.stdin.readline().strip()
    if not secret:
        sys.exit("no key provided")

    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored()
    env = _read_env()
    env[VAR[args.provider]] = secret
    with open(ENV_PATH, "w", encoding="utf-8") as fh:
        for k, v in env.items():
            fh.write(f"{k}={v}\n")

    print(f"wrote {VAR[args.provider]}={_mask(secret)} -> {ENV_PATH}")
    print(".local/ is gitignored; the key is not printed in full and not committed.")


if __name__ == "__main__":
    main()
