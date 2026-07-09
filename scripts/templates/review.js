"use strict";

/* ------------------------------------------------------------------ state */

const state = {
  items: [],
  misconceptions: [],
  misById: {},
  topics: [],          // topic order for grouping
  totals: { items: 0, distractors: 0 },
  filter: "all",
  currentId: null,     // item id currently shown
  active: 0,           // index of active distractor within current item
  saveTimers: {},      // per-item debounce timers for notes
};

const $ = (sel) => document.querySelector(sel);

/* ------------------------------------------------------------------ load */

async function boot() {
  const res = await fetch("/api/data");
  const data = await res.json();
  state.items = data.items;
  state.misconceptions = data.misconceptions;
  state.totals = data.totals;
  state.misById = {};
  const seen = new Set();
  state.topics = [];
  for (const m of state.misconceptions) {
    state.misById[m.id] = m;
    if (!seen.has(m.topic)) { seen.add(m.topic); state.topics.push(m.topic); }
  }
  if (state.items.length) state.currentId = state.items[0].id;
  wireGlobalUI();
  render();
  updateProgress();
}

/* ------------------------------------------------------------- filtering */

function itemMatchesFilter(item) {
  if (state.filter === "unreviewed") {
    return item.distractors.some((d) => !d.review.reviewed);
  }
  if (state.filter === "nofit") {
    return item.distractors.some((d) => d.no_fit);
  }
  return true;
}

function visibleItems() {
  const vis = state.items.filter(itemMatchesFilter);
  return vis.length ? vis : state.items; // never strand the user on empty
}

function currentItem() {
  return state.items.find((it) => it.id === state.currentId) || state.items[0];
}

/* ------------------------------------------------------------- rendering */

function render() {
  const item = currentItem();
  const view = $("#itemView");
  if (!item) { view.innerHTML = '<div class="empty">No items.</div>'; return; }

  const vis = visibleItems();
  let pos = vis.findIndex((it) => it.id === item.id);
  if (pos < 0) pos = 0;
  $("#itemCounter").textContent = `Item ${pos + 1} / ${vis.length}`;

  const correct = item.correct;
  const choicesHtml = Object.keys(item.choices).sort().map((k) => {
    const isCorrect = k === correct;
    return `<div class="choice ${isCorrect ? "correct" : "wrong"}">
      <span class="letter">${k}</span>
      <span class="ctext">${esc(item.choices[k])}</span>
      ${isCorrect ? '<span class="tick">correct answer</span>' : ""}
    </div>`;
  }).join("");

  const passageHtml = item.passage
    ? `<div class="passage"><div class="passage-label">Passage</div>${esc(item.passage)}</div>`
    : "";

  const included = item.include;
  view.innerHTML = `
    <div class="item-card">
      <div class="item-head">
        <div class="item-meta">
          <span class="topic-badge">${esc(item.topic || "\u2014")}</span>
          <span class="item-id">${esc(item.id)}</span>
          <label class="include-toggle ${included ? "" : "excluded"}">
            <input type="checkbox" id="includeChk" ${included ? "checked" : ""}/>
            include in gold set
          </label>
        </div>
        ${passageHtml}
        <div class="stem">${esc(item.stem || "")}</div>
        <div class="choices">${choicesHtml}</div>
      </div>
      <div class="distractors-wrap">
        <div class="distractors-title">Wrong distractors \u2014 review each</div>
        <div id="dcards"></div>
      </div>
    </div>`;

  $("#includeChk").addEventListener("change", (e) => {
    item.include = e.target.checked;
    item.include_touched = true;
    render();
    saveItem(item);
  });

  const wrap = $("#dcards");
  item.distractors.forEach((d, idx) => wrap.appendChild(renderDistractor(item, d, idx)));

  if (state.active >= item.distractors.length) state.active = 0;
  highlightActive();
}

function renderDistractor(item, d, idx) {
  const card = document.createElement("div");
  const dec = d.review.decision;
  card.className = "dcard" + (dec ? " d-" + dec : "");
  card.dataset.idx = idx;

  const badges = [];
  if (d.no_fit) badges.push('<span class="badge badge-nofit">draft: no fit</span>');
  if (d.needs_review) badges.push('<span class="badge badge-review">needs review</span>');
  if (typeof d.confidence === "number") badges.push(`<span class="badge badge-conf">conf ${d.confidence.toFixed(2)}</span>`);
  if (dec) badges.push(`<span class="badge badge-status ${dec}">${decLabel(dec)}</span>`);

  let draftLine;
  if (d.draft_tag) {
    draftLine = `<span class="tag-id">${esc(d.draft_tag)}</span> &middot; <span class="tag-name">${esc(d.draft_tag_name || "")}</span>`;
  } else {
    draftLine = '<span class="tag-null">no drafted tag (model marked no-fit)</span>';
  }

  const chosenCorrect = dec === "correct" && d.review.tag;
  const chosenLabel = chosenCorrect ? `${d.review.tag} \u2014 ${d.review.tag_name || ""}` : "";

  card.innerHTML = `
    <div class="dcard-head">
      <span class="dchoice-letter">${d.letter}</span>
      <span class="dchoice-text">${esc(d.text)}</span>
      <span class="badges">${badges.join("")}</span>
    </div>
    <div class="draft-tag">Draft: ${draftLine}</div>
    <div class="rationale">${esc(d.rationale || "")}</div>
    <div class="dcontrols">
      <button class="dbtn btn-confirm ${dec === "confirm" ? "sel-confirm" : ""}" data-act="confirm" ${d.draft_tag ? "" : "disabled title='no drafted tag to confirm'"}>Confirm</button>
      <div class="typeahead">
        <input type="text" class="ta-input ${chosenCorrect ? "chosen" : ""}" placeholder="Correct &rarr; search misconception\u2026" value="${esc(chosenLabel)}" autocomplete="off"/>
        <div class="ta-list"></div>
      </div>
      <button class="dbtn btn-drop ${dec === "drop" ? "sel-drop" : ""}" data-act="drop">No fit / drop</button>
    </div>
    <div class="notes">
      <label>notes</label>
      <textarea class="note-input" placeholder="optional note\u2026">${esc(d.review.notes || "")}</textarea>
    </div>`;

  card.addEventListener("mousedown", () => { state.active = idx; highlightActive(); });

  card.querySelector('[data-act="confirm"]').addEventListener("click", () => setDecision(item, idx, "confirm"));
  card.querySelector('[data-act="drop"]').addEventListener("click", () => setDecision(item, idx, "drop"));

  const note = card.querySelector(".note-input");
  note.addEventListener("input", () => {
    d.review.notes = note.value;
    debouncedSave(item);
  });

  setupTypeahead(card, item, idx, d);
  return card;
}

function decLabel(dec) {
  return dec === "confirm" ? "confirmed" : dec === "correct" ? "corrected" : "dropped";
}

/* ------------------------------------------------------------- typeahead */

function setupTypeahead(card, item, idx, d) {
  const input = card.querySelector(".ta-input");
  const list = card.querySelector(".ta-list");
  let hl = -1;
  let opts = [];

  function build(query) {
    const q = query.trim().toLowerCase();
    list.innerHTML = "";
    opts = [];
    hl = -1;
    for (const topic of state.topics) {
      const matches = state.misconceptions.filter((m) => m.topic === topic && (
        !q || m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q) ||
        (m.description || "").toLowerCase().includes(q) || topic.toLowerCase().includes(q)
      ));
      if (!matches.length) continue;
      const g = document.createElement("div");
      g.className = "ta-group";
      g.textContent = topic.replace(/_/g, " ");
      list.appendChild(g);
      for (const m of matches) {
        const o = document.createElement("div");
        o.className = "ta-opt";
        o.innerHTML = `<span class="oid">${m.id}</span> <span class="oname">${esc(m.name)}</span><span class="odesc">${esc(m.description || "")}</span>`;
        o.addEventListener("mousedown", (e) => { e.preventDefault(); choose(m); });
        list.appendChild(o);
        opts.push(o);
      }
    }
    if (!opts.length) list.innerHTML = '<div class="ta-opt" style="color:var(--muted)">no match</div>';
  }

  function open() { build(""); list.classList.add("open"); }
  function close() { list.classList.remove("open"); }

  function choose(m) {
    input.value = `${m.id} \u2014 ${m.name}`;
    input.classList.add("chosen");
    close();
    setDecision(item, idx, "correct", m.id);
  }

  input.addEventListener("focus", () => { state.active = idx; highlightActive(); open(); });
  input.addEventListener("input", () => { build(input.value); list.classList.add("open"); });
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); hl = Math.min(hl + 1, opts.length - 1); paint(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); hl = Math.max(hl - 1, 0); paint(); }
    else if (e.key === "Enter") {
      e.preventDefault();
      if (hl >= 0 && opts[hl]) opts[hl].dispatchEvent(new MouseEvent("mousedown"));
    } else if (e.key === "Escape") { close(); input.blur(); }
  });
  function paint() {
    opts.forEach((o, i) => o.classList.toggle("hl", i === hl));
    if (hl >= 0 && opts[hl]) opts[hl].scrollIntoView({ block: "nearest" });
  }

  input._openTypeahead = () => { input.focus(); };
}

/* --------------------------------------------------------------- actions */

function setDecision(item, idx, decision, tag) {
  const d = item.distractors[idx];
  d.review.decision = decision;
  if (decision === "confirm") {
    d.review.tag = d.draft_tag;
    d.review.tag_name = d.draft_tag_name;
  } else if (decision === "correct") {
    d.review.tag = tag;
    d.review.tag_name = (state.misById[tag] || {}).name || tag;
  } else { // drop
    d.review.tag = null;
    d.review.tag_name = null;
  }
  d.review.reviewed = true;

  if (!item.include_touched) {
    item.include = !item.distractors.every((x) => x.review.decision === "drop");
  }
  state.active = idx;
  render();
  updateProgress();
  saveItem(item);
}

function toggleActiveInclude() {
  const item = currentItem();
  item.include = !item.include;
  item.include_touched = true;
  render();
  saveItem(item);
}

/* --------------------------------------------------------------- persist */

function reviewPayload(item) {
  const distractors = {};
  for (const d of item.distractors) {
    distractors[d.letter] = {
      decision: d.review.decision,
      tag: d.review.tag,
      notes: d.review.notes || "",
    };
  }
  return {
    id: item.id,
    include: item.include,
    include_touched: item.include_touched,
    distractors,
  };
}

async function saveItem(item) {
  setSaveStatus("saving");
  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reviewPayload(item)),
    });
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || "save failed");
    setSaveStatus("saved");
  } catch (err) {
    setSaveStatus("error");
    toast("Save failed: " + err.message);
  }
}

function debouncedSave(item) {
  setSaveStatus("saving");
  clearTimeout(state.saveTimers[item.id]);
  state.saveTimers[item.id] = setTimeout(() => saveItem(item), 400);
}

function setSaveStatus(s) {
  const el = $("#saveStatus");
  el.className = "save-status" + (s === "saving" ? " saving" : s === "error" ? " error" : "");
  el.textContent = s === "saving" ? "saving\u2026" : s === "error" ? "save error" : "saved";
}

/* -------------------------------------------------------------- progress */

function updateProgress() {
  let distDone = 0, itemsDone = 0;
  for (const it of state.items) {
    const done = it.distractors.filter((d) => d.review.reviewed).length;
    distDone += done;
    if (done === it.distractors.length && it.distractors.length) itemsDone += 1;
  }
  const dt = state.totals.distractors || 1;
  const nt = state.totals.items || 1;
  $("#distStat").textContent = `${distDone} / ${state.totals.distractors}`;
  $("#itemStat").textContent = `${itemsDone} / ${state.totals.items}`;
  $("#distBar").style.width = (100 * distDone / dt) + "%";
  $("#itemBar").style.width = (100 * itemsDone / nt) + "%";
}

/* ------------------------------------------------------------ navigation */

function go(delta) {
  const vis = visibleItems();
  let pos = vis.findIndex((it) => it.id === state.currentId);
  if (pos < 0) pos = 0;
  pos = Math.min(Math.max(pos + delta, 0), vis.length - 1);
  state.currentId = vis[pos].id;
  state.active = 0;
  render();
}

function highlightActive() {
  document.querySelectorAll(".dcard").forEach((c, i) => c.classList.toggle("active", i === state.active));
  const active = document.querySelector(".dcard.active");
  if (active) active.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

/* --------------------------------------------------------------- export */

async function doExport() {
  setSaveStatus("saving");
  try {
    const res = await fetch("/api/export", { method: "POST" });
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || "export failed");
    setSaveStatus("saved");
    toast(`Gold set written: ${j.items} items / ${j.distractors} distractors`);
  } catch (err) {
    setSaveStatus("error");
    toast("Export failed: " + err.message);
  }
}

/* -------------------------------------------------------------- keyboard */

function wireGlobalUI() {
  $("#prevBtn").addEventListener("click", () => go(-1));
  $("#nextBtn").addEventListener("click", () => go(1));
  $("#exportBtn").addEventListener("click", doExport);
  $("#filterSelect").addEventListener("change", (e) => {
    state.filter = e.target.value;
    const vis = visibleItems();
    if (!vis.some((it) => it.id === state.currentId)) state.currentId = vis[0].id;
    state.active = 0;
    render();
  });

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    const typing = tag === "input" || tag === "textarea" || tag === "select";

    if (e.key === "/" && !typing) {
      e.preventDefault();
      focusActiveSearch();
      return;
    }
    if (typing) return; // don't hijack keys while editing text

    const item = currentItem();
    switch (e.key) {
      case "y": setDecision(item, state.active, "confirm"); break;
      case "n": setDecision(item, state.active, "drop"); break;
      case "j": case "ArrowDown":
        e.preventDefault();
        state.active = Math.min(state.active + 1, item.distractors.length - 1);
        highlightActive();
        break;
      case "k": case "ArrowUp":
        e.preventDefault();
        state.active = Math.max(state.active - 1, 0);
        highlightActive();
        break;
      case "ArrowLeft": e.preventDefault(); go(-1); break;
      case "ArrowRight": e.preventDefault(); go(1); break;
      case "i": toggleActiveInclude(); break;
    }
  });
}

function focusActiveSearch() {
  const card = document.querySelectorAll(".dcard")[state.active];
  if (card) { const input = card.querySelector(".ta-input"); if (input) input.focus(); }
}

/* ----------------------------------------------------------------- utils */

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3200);
}

boot();
