# Gold-set review tool (`scripts/review_server.py`)

A local, zero-install web UI for the owner to review and finalize the
frontier-drafted misconception tags into the **gold eval set**.

The frontier model drafted a misconception tag for every wrong distractor in the
real eval pool (`data/real_bio_eval_drafted.jsonl` — 80 items / 240 wrong
distractors). This tool lets the owner confirm, correct, or drop each draft,
then export a clean, human-verified gold file.

## Run

```bash
python scripts/review_server.py
```

It starts a local server at **http://127.0.0.1:8000** and tries to open your
browser automatically. Stop it with `Ctrl+C`.

- Standard-library only — no `pip install`, works offline (no CDNs/frameworks).
- Options: `--port 8000` to change the port, `--host`, and `--no-browser`
  (skip auto-opening the browser; used by smoke tests).

## What you do per distractor

For each **wrong** distractor the UI shows the drafted misconception (id + name),
the model's rationale, its confidence, and a **no-fit** badge when the model
couldn't place it. You then pick one:

- **Confirm** — accept the drafted tag.
- **Correct** — searchable typeahead over all 46 misconceptions (grouped by
  topic, showing id + name + description); pick the right one.
- **No fit / drop** — drop this distractor from the gold set.
- **notes** — optional free text.

Each item also has an **include in gold set** toggle (defaults on; auto-unchecks
if you drop all of its distractors — you can override it).

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `y` | Confirm the active distractor |
| `n` | Drop / no-fit the active distractor |
| `j` / `k` (or `Down`/`Up`) | Move between distractors |
| `Left` / `Right` | Previous / next item |
| `/` | Focus the "correct" search box |
| `i` | Toggle "include in gold set" |

A sticky header shows two progress bars (distractors reviewed / 240, items
complete / 80) and a filter to show **all**, **unreviewed only**, or items that
**have a no-fit draft**.

## Files (all under `data/`, git-ignored — never committed)

- **Autosave:** every change is written immediately to
  `data/real_bio_eval_reviewed.jsonl`. A crash or refresh loses nothing — reload
  restores your state (the tool merges saved progress on load).
- **Export:** the **Export gold** button writes `data/real_bio_eval_gold.jsonl` —
  only items marked *include* **and** with at least one confirmed/corrected
  (non-dropped) distractor. Dropped distractors are omitted; kept distractors get
  `tag_source: "human_verified"`, `no_fit: false`, `needs_review: false`;
  `provenance` stays `"real_eval"`. The server prints how many items/distractors
  made the gold set.
