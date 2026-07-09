#!/usr/bin/env python3
"""Local, zero-install review tool for the AP-Bio misconception gold eval set.

Owner workflow
--------------
The frontier model drafted misconception tags for every WRONG distractor in the
real eval pool (``data/real_bio_eval_drafted.jsonl``). This script serves a small
local web UI so the owner can, per distractor:

  * CONFIRM the drafted misconception tag,
  * CORRECT it (searchable dropdown over the 46 misconceptions), or
  * DROP it (no fit / not a usable distractor),

with a free-text note and an item-level "include in gold set" toggle.

Every change is autosaved immediately to ``data/real_bio_eval_reviewed.jsonl`` so a
crash or refresh loses nothing (the UI restores from it on load). The Export button
writes ``data/real_bio_eval_gold.jsonl`` (finalized gold set).

Design constraints (deliberate):
  * Python STANDARD LIBRARY ONLY -- no pip installs. Runs on Windows / Python 3.14.
  * Frontend is vanilla HTML/CSS/JS served from ``scripts/templates/`` -- no CDNs,
    fully offline.

Run
---
    python scripts/review_server.py

Opens http://127.0.0.1:8000 in the default browser. Use ``--port`` to change the
port and ``--no-browser`` to skip auto-opening (used by the smoke test).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --------------------------------------------------------------------- paths

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA = ROOT / "data"
TEMPLATES = SCRIPT_DIR / "templates"

DRAFTED_PATH = DATA / "real_bio_eval_drafted.jsonl"
MISCONCEPTIONS_PATH = DATA / "apbio_misconceptions.json"
REVIEWED_PATH = DATA / "real_bio_eval_reviewed.jsonl"
GOLD_PATH = DATA / "real_bio_eval_gold.jsonl"

# Decisions a reviewer can record for a single distractor.
DECISION_CONFIRM = "confirm"
DECISION_CORRECT = "correct"
DECISION_DROP = "drop"
REVIEWED_DECISIONS = {DECISION_CONFIRM, DECISION_CORRECT, DECISION_DROP}

_LOCK = threading.Lock()


# --------------------------------------------------------------------- data io

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Write JSONL to a temp file then atomically replace, so a crash mid-write
    never corrupts the saved review progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_misconceptions() -> list[dict]:
    with open(MISCONCEPTIONS_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("misconceptions", [])


# --------------------------------------------------------------------- state

class ReviewStore:
    """In-memory review state, mirrored to ``real_bio_eval_reviewed.jsonl``.

    One record per item id::

        {"id", "include", "include_touched",
         "distractors": {"A": {"decision", "tag", "notes"}, ...}}
    """

    def __init__(self) -> None:
        self.items = load_jsonl(DRAFTED_PATH)
        self.misconceptions = load_misconceptions()
        self.mis_by_id = {m["id"]: m for m in self.misconceptions}
        self.reviews: dict[str, dict] = {}
        for rec in load_jsonl(REVIEWED_PATH):
            if "id" in rec:
                self.reviews[rec["id"]] = rec

    # ----- helpers

    def _wrong_letters(self, item: dict) -> list[str]:
        correct = item.get("correct")
        return [k for k in sorted(item.get("choices", {})) if k != correct]

    def _mis_name(self, tag: str | None) -> str | None:
        if not tag:
            return None
        m = self.mis_by_id.get(tag)
        return m["name"] if m else tag

    # ----- persistence

    def _persist(self) -> None:
        rows = [self.reviews[item["id"]] for item in self.items if item["id"] in self.reviews]
        write_jsonl_atomic(REVIEWED_PATH, rows)

    def save_item_review(self, record: dict) -> None:
        item_id = record.get("id")
        if not item_id:
            raise ValueError("save payload missing 'id'")
        clean = {
            "id": item_id,
            "include": bool(record.get("include", True)),
            "include_touched": bool(record.get("include_touched", False)),
            "distractors": {},
        }
        for letter, d in (record.get("distractors") or {}).items():
            clean["distractors"][letter] = {
                "decision": d.get("decision"),
                "tag": d.get("tag"),
                "notes": d.get("notes", ""),
            }
        with _LOCK:
            self.reviews[item_id] = clean
            self._persist()

    # ----- projections for the frontend

    def build_payload(self) -> dict:
        items = []
        for item in self.items:
            item_id = item["id"]
            review = self.reviews.get(item_id, {})
            rdist = review.get("distractors", {})
            distractors = []
            for letter in self._wrong_letters(item):
                draft = item.get("distractor_tags", {}).get(letter, {})
                rev = rdist.get(letter, {})
                decision = rev.get("decision")
                distractors.append({
                    "letter": letter,
                    "text": item.get("choices", {}).get(letter, ""),
                    "draft_tag": draft.get("tag"),
                    "draft_tag_name": self._mis_name(draft.get("tag")),
                    "rationale": draft.get("rationale", ""),
                    "confidence": draft.get("confidence"),
                    "no_fit": bool(draft.get("no_fit")),
                    "needs_review": bool(draft.get("needs_review")),
                    "tag_source": draft.get("tag_source"),
                    "review": {
                        "decision": decision,
                        "tag": rev.get("tag"),
                        "tag_name": self._mis_name(rev.get("tag")),
                        "notes": rev.get("notes", ""),
                        "reviewed": decision in REVIEWED_DECISIONS,
                    },
                })
            items.append({
                "id": item_id,
                "topic": item.get("topic"),
                "passage": item.get("passage"),
                "stem": item.get("stem"),
                "choices": item.get("choices", {}),
                "correct": item.get("correct"),
                "correct_answer": item.get("correct_answer"),
                "include": review.get("include", True),
                "include_touched": review.get("include_touched", False),
                "distractors": distractors,
            })
        return {
            "items": items,
            "misconceptions": self.misconceptions,
            "totals": {
                "items": len(items),
                "distractors": sum(len(i["distractors"]) for i in items),
            },
        }

    # ----- export

    def export_gold(self) -> dict:
        gold_rows = []
        gold_distractors = 0
        for item in self.items:
            item_id = item["id"]
            review = self.reviews.get(item_id)
            if not review:
                continue
            if not review.get("include", True):
                continue
            rdist = review.get("distractors", {})
            final_tags = {}
            for letter in self._wrong_letters(item):
                rev = rdist.get(letter, {})
                decision = rev.get("decision")
                if decision == DECISION_CONFIRM:
                    tag = item.get("distractor_tags", {}).get(letter, {}).get("tag")
                elif decision == DECISION_CORRECT:
                    tag = rev.get("tag")
                else:
                    continue  # dropped or unreviewed -> omit from gold
                if not tag:
                    continue
                draft = item.get("distractor_tags", {}).get(letter, {})
                entry = {
                    "tag": tag,
                    "rationale": draft.get("rationale", ""),
                    "confidence": draft.get("confidence"),
                    "no_fit": False,
                    "needs_review": False,
                    "tag_source": "human_verified",
                }
                notes = (rev.get("notes") or "").strip()
                if notes:
                    entry["review_notes"] = notes
                final_tags[letter] = entry
            if not final_tags:
                continue  # no confirmed/corrected distractor -> item not gold
            row = dict(item)
            row["distractor_tags"] = final_tags
            row["provenance"] = "real_eval"
            gold_rows.append(row)
            gold_distractors += len(final_tags)
        with _LOCK:
            write_jsonl_atomic(GOLD_PATH, gold_rows)
        return {
            "items": len(gold_rows),
            "distractors": gold_distractors,
            "path": str(GOLD_PATH),
        }


# --------------------------------------------------------------------- http

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    store: ReviewStore  # injected on the server instance

    def log_message(self, fmt, *args):  # keep the console quiet
        return

    # ----- helpers

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404, "Not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", STATIC_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # ----- routes

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_file(TEMPLATES / "review.html")
        elif path == "/api/data":
            self._send_json(self.server.store.build_payload())
        elif path.startswith("/static/"):
            name = os.path.basename(path)  # prevent traversal
            self._send_file(TEMPLATES / name)
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/save":
                payload = self._read_body()
                self.server.store.save_item_review(payload)
                self._send_json({"ok": True})
            elif path == "/api/export":
                result = self.server.store.export_gold()
                print(f"[export] gold set: {result['items']} items / "
                      f"{result['distractors']} distractors -> {result['path']}")
                self._send_json({"ok": True, **result})
            else:
                self.send_error(404, "Not found")
        except Exception as exc:  # surface errors to the client instead of 500-crashing
            self._send_json({"ok": False, "error": str(exc)}, status=400)


# --------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open the browser (used by smoke tests).")
    args = parser.parse_args()

    if not DRAFTED_PATH.exists():
        raise SystemExit(f"Drafted eval file not found: {DRAFTED_PATH}")
    if not MISCONCEPTIONS_PATH.exists():
        raise SystemExit(f"Misconception taxonomy not found: {MISCONCEPTIONS_PATH}")

    store = ReviewStore()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = store

    url = f"http://{args.host}:{args.port}"
    print(f"Review tool serving at {url}")
    print(f"  drafted : {DRAFTED_PATH}")
    print(f"  reviewed: {REVIEWED_PATH}  (autosaved)")
    print(f"  gold    : {GOLD_PATH}  (on export)")
    print(f"  items   : {len(store.items)}  distractors: "
          f"{sum(len(store._wrong_letters(i)) for i in store.items)}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
