# Distractor Adjudication Rubric + Taxonomy Consolidation Proposal

Status: **PROPOSAL / DRAFT — pending owner sign-off.** This document is additive.
It does **not** modify `data/apbio_misconceptions.json`, `scripts/common_bio.py`,
or any other data/script. The consolidated taxonomy it argues for ships as a
separate, non-destructive file: `data/apbio_misconceptions_v2.proposal.json`.
Nothing here is applied to the live taxonomy until an owner approves it.

Read alongside `docs/mcat_pivot_spec.md` (§3 taxonomy, §12 generate-and-verify)
and `docs/litmus_plan.md` (§4 two-layer mapping). It operationalizes the
distractor→misconception tagging that the litmus (`scripts/litmus_tagging.py`) and
the generation verifier (V3) both depend on.

---

## Part A — The adjudication rubric (apply PER wrong distractor)

A human rater (and later a verifier model) applies this to **one wrong distractor
at a time**: a specific incorrect option that a student did (or would) choose on a
specific item. The unit is the `(item, chosen_distractor)` pair, never the item as
a whole.

### A.1 The core decision

```
                 ┌─────────────────────────────────────────────────────────┐
                 │ Look at THIS wrong distractor on THIS stem.               │
                 └─────────────────────────────────────────────────────────┘
                                          │
        Step 1 ── Can I name the SPECIFIC wrong belief that makes
                  THIS distractor attractive?
                  (Not "it's wrong" — WHY would a student be pulled to it?)
                                          │
              ┌───────────────────────────┴───────────────────────────┐
             NO / "they just guessed or                              YES
             didn't know the term"                                     │
                      │                                                │
                      ▼                              Step 2 ── Does that named belief match a
              verdict = content_gap                  (consolidated) misconception in the taxonomy?
              (a FIRST-CLASS distractor-level                        │
               verdict — NOT a misconception)         ┌──────────────┼───────────────┐
                                                      YES         YES, but          NO — but it IS a real,
                                                       │          not listed        coherent, reproducible
                                                       ▼             │               belief (see A.2)
                                              verdict = tag it        │                     │
                                              with that misconception │                     ▼
                                                       │              └────────────►  verdict = taxonomy_gap
                                                       │                              (candidate) + describe it
                                                       ▼
                       Step 3 ── Sanity check: is this actually just a wrong vocabulary word or a
                                 random plausible option with NO underlying model behind it?
                                 If yes → downgrade to content_gap.
```

Three terminal verdicts:

1. **`content_gap`** — a **first-class, distractor-level verdict**, not a
   misconception. The student lacks the fact/term the item tests; the wrong choice
   reflects an absent or fuzzy fact, not a coherent wrong theory. This is the honest
   default whenever you cannot name the attractive wrong belief.
2. **`<misconception id>`** — the named belief matches a (consolidated) taxonomy
   entry. Tag it with that id.
3. **`taxonomy_gap (candidate)`** — you *can* name a real, coherent, reproducible
   wrong belief, but the taxonomy has no id for it. Record the belief in prose so
   the taxonomy owner can decide whether to add it. Do **not** force-fit it into the
   nearest existing id.

> **Why `content_gap` is first-class, not a fallback bucket.** Per
> `mcat_pivot_spec.md` §3.1, `content_gap` (declarative) is a legitimate coarse
> outcome that routes to the **memory score**. Most harvested wrong answers are
> genuinely "they didn't know it," and calling that out is a *result*, not a
> failure to tag (see A.5).

### A.2 Supporting heuristic 1 — predictability / reproducibility

The line between a **misconception** and a **content_gap** is whether the wrong
choice is *shared and reproducible*:

- **Misconception** → a specific, **predictable** wrong belief that many students
  independently hold and that reliably attracts them to the same option ("if they
  think evolution is goal-directed, they will pick the 'progress toward perfection'
  option"). It is *reproducible across students*.
- **content_gap** → the miss is **scattered**. Different students who don't know
  the fact spread across whichever distractors happen to sound plausible; there is
  no single belief driving them to one option. Nothing predicts *which* wrong
  answer they land on.

Test to apply: *"If I gave this item to 100 students who get it wrong, would they
cluster on THIS distractor for a REASON I can state, or scatter?"* Cluster-for-a-
reason → misconception. Scatter → content_gap.

### A.3 Supporting heuristic 2 — diagnosticity

Ask: **did we learn HOW this student thinks, or only THAT they don't know it?**

- If tagging the distractor tells you something *actionable and specific* about the
  student's mental model (and would drive a targeted remediation), it is a
  **misconception**. ("They invert osmosis direction" → drill tonicity.)
- If all you learned is "this fact isn't in their head," it is a **content_gap**.
  ("They didn't recognize the term 'selective permeability'" → re-teach the term.)

A misconception is diagnostic (it reveals a model); a content_gap is merely
detective (it reveals an absence).

### A.4 The confidence / timing axis (and predict-and-confirm)

The draft tags carry a `confidence` field, and timing is a deferred Phase-2+ signal
(`common_bio.apply_timing_gate`, currently inert). The axis nonetheless guides
adjudication:

| Signal | Leans `content_gap` | Leans misconception |
|---|---|---|
| Confidence in the wrong answer | low-confidence, tentative, abstain-worthy | **confident-but-wrong** |
| Response speed (Phase 2+) | slow / hesitant (searching for a half-known fact) | **fast** (the wrong belief is fluent and automatic) |
| Behavior | would likely abstain or guess | commits to the wrong model |

**How it ties to predict-and-confirm** (`mcat_pivot_spec.md` §2, §4.2): a *confident,
fast, wrong* answer is the signature of a real misconception — the student is sure,
so the system should **predict the misconception and confirm** it with the student
(surface the top-2 and let them select). A *low-confidence, slow* answer is the
signature of a content_gap — there is no coherent model to confirm, so the honest
action is to treat it as a knowledge gap and re-queue the fact, not to fabricate a
misconception. In v1 (no real timing) we lean on confidence + semantics only, and
when they conflict or two misconceptions tie, the honest verdict is `content_gap`
or `abstain → predict-and-confirm`, never a guessed misconception.

### A.5 Finding — harvested distractors are largely "filler"

Across the 240 drafted distractors in `data/real_bio_eval_drafted.jsonl`, **50.8%
are `no_fit`** (no misconception applies). This is not a tagging failure — it is a
property of the *source*. The items are harvested from MMLU, SciQ, and ARC, whose
wrong options were written to be **plausible-but-clearly-wrong test distractors**,
not to embody a specific student misconception. Many are simply wrong vocabulary
words ("periodic permeability", "moderate permeability"), off-topic plausible nouns
("atom", "organism"), or content the taxonomy doesn't cover at all (immune system,
endocrine regulation, RNA processing).

**Consequence for the project:** misconception-bearing distractors mostly have to be
**authored/generated on purpose** (which is exactly why `mcat_pivot_spec.md` §12
makes conditional *generation* of misconception-tagged items the primary v1 task).
Real *harvested* eval items therefore skew heavily toward `content_gap`, and the
eval must report that honestly rather than inflate misconception coverage. A high
`no_fit` / `content_gap` rate on harvested items is the expected, correct result —
not a bug to tune away.

### A.6 Worked examples (real items from `real_bio_eval_drafted.jsonl`)

Verdicts below use the **consolidated (v2 proposal)** ids. Where the frontier draft
tag differs, that is called out — several examples show the rubric *correcting* a
draft.

---

**Example 1 — clear misconception (reasoning_error → `evo_teleology`).**
Item `mmlu_hsbio_test_0307`, topic evolution.

> **Stem:** "Which statement about natural selection is most correct?"
> **Distractor C:** "Adaptations beneficial at one time should generally be
> beneficial during all other times as well." (correct answer: D)

- Step 1: Can I name the attractive belief? **Yes** — the student thinks adaptation
  is *directional and permanent*, i.e. evolution moves toward a fixed "better" state
  regardless of context. That is textbook teleology.
- Step 2: Matches a consolidated misconception? **Yes → `evo_teleology`** (v1
  `evo_goal_directed_progress`).
- Predictability/diagnosticity: many students share the "evolution = progress" model
  and will cluster on this option; tagging it drives a specific remediation
  (context-dependence of fitness). High confidence + fast would confirm it.
- **Verdict: `evo_teleology` (misconception).**

---

**Example 2 — clear misconception (reasoning_error → `gen_cross_computation_error`).**
Item `arc_arcchallenge_test_00110`, topic genetics.

> **Stem:** "In pea plants, the trait for round seeds is dominant over wrinkled. If
> a pure dominant round-seed plant is crossed with a wrinkled-seed plant, what can
> be predicted about the offspring?"
> **Distractor C:** "Each offspring plant will have some round and some wrinkled
> seeds." (correct answer: A — all round)

- Step 1: Attractive belief? **Yes** — the student mis-works the cross, expecting
  both phenotypes to segregate within the F1 rather than a uniform dominant F1.
- Step 2: Matches? **Yes → `gen_cross_computation_error`** (this consolidates v1
  `gen_wrong_punnett_ratio` with the other cross-math tags a rater can't separate
  from one numeric answer).
- Note: this is the merge in action — the v1 draft tagged C `gen_wrong_punnett_ratio`
  and D `gen_wrong_punnett_ratio`, but B `gen_recessive_disappears`; under v2 the
  numeric/ratio slips collapse to one id, which is the point of the consolidation.
- **Verdict: `gen_cross_computation_error` (misconception).**

---

**Example 3 — misconception in the comprehension family (`map_confuses_variable_role`).**
Item `arc_arcchallenge_test_00193`, topic experimental_design.

> **Stem:** "…Each plant receives the same amount of water but different amounts of
> sunlight. The students measure the number of fruits on each plant every day. What
> is the dependent variable?"
> **Distractor C:** "amount of sunlight." (correct answer: D — number of fruits)

- Step 1: Attractive belief? **Yes** — the student swaps the independent variable
  (sunlight, the thing manipulated) for the dependent variable (fruits, the thing
  measured).
- Step 2: Matches? **Yes → `map_confuses_variable_role`** (retained unchanged; well
  supported at n=6). Coarse = `misread_or_passage_mapping`.
- This is a *reproducible* role confusion, not a content gap: the student can read
  the passage but maps the roles backward — high diagnosticity.
- **Verdict: `map_confuses_variable_role` (misconception).**

---

**Example 4 — clean content_gap (vocabulary, no model).**
Item `sciq_train_00053`, topic membrane_transport.

> **Stem:** "The ability for a plasma membrane to only allow certain molecules in or
> out of the cell is referred to as what?"
> **Distractor B:** "periodic permeability." (correct answer: A — selective
> permeability; other distractors: "total permeability", "moderate permeability")

- Step 1: Can I name the attractive wrong belief? **No.** "Periodic permeability" is
  not a concept any student *believes*; it is a made-up vocab word that sounds
  plausible next to "selective." There is no model here to name.
- Predictability: students who don't know the term will scatter across periodic /
  total / moderate — no single belief drives them to B.
- **Verdict: `content_gap`.** (Matches the frontier draft's `no_fit`.) This is the
  canonical "wrong vocabulary word / random plausible option" case of Step 3.

---

**Example 5 — content_gap (wrong scale) + a draft correction.**
Item `sciq_train_00132`, topic membrane_transport.

> **Stem:** "What term describes a collection of molecules surrounded by a
> phospholipid bilayer that is capable of reproducing itself?"
> **Distractor B:** "atom." **Distractor C:** "proteins." (correct answer: A — cell;
> other distractor: "organism")

- Step 1: For B ("atom") — attractive belief? **No.** Choosing "atom" reflects only
  a fuzzy sense of biological scale, not a coherent, reproducible theory. Scattered,
  low-diagnosticity → **`content_gap`.**
- **Draft correction:** the frontier draft tagged C ("proteins")
  `map_attributes_mass_to_solute` — a membrane osmosis tag that has nothing to do
  with this scale/definition item. Under this rubric C is also a
  scale/definition **`content_gap`**, and the mistagging is exactly the kind of
  force-fit the rubric exists to prevent (Step 3).
- **Verdict: `content_gap` for all three distractors.**

---

**Example 6 — taxonomy_gap (candidate).**
Item `mmlu_hsbio_validation_0013`, labeled topic enzymes (actually endocrine).

> **Stem:** "Destruction of all beta cells in the pancreas will cause which of the
> following to occur?"
> **Distractor A:** "Glucagon secretion will stop and blood glucose levels will
> increase." (correct answer: D — *Insulin* secretion will stop and blood glucose
> increases)

- Step 1: Can I name the attractive belief? **Yes** — the student **reverses the
  roles of insulin and glucagon** (thinks beta cells make glucagon, or that glucagon
  lowers blood glucose). This is a well-documented, reproducible, shared
  misconception in endocrine physiology.
- Step 2: Matches a taxonomy id? **No.** The 46-tag taxonomy (and the v2 proposal)
  has no endocrine/hormone-regulation misconception, and "enzymes" is a mislabeled
  topic here.
- Because the belief is real, coherent, and reproducible (not a scatter), the honest
  verdict is **not** `content_gap` and **not** a force-fit into an enzyme tag.
- **Verdict: `taxonomy_gap (candidate)` — "reverses insulin/glucagon roles in
  blood-glucose regulation."** (The frontier draft marked these `no_fit`; the rubric
  upgrades a coherent-belief `no_fit` to a taxonomy_gap candidate so the owner can
  decide whether to add an endocrine misconception.)

---

## Part B — Data-grounded taxonomy consolidation proposal

**Deliverable file:** `data/apbio_misconceptions_v2.proposal.json` (proposed set +
embedded crosswalk). **This is a proposal, not an applied change.**

### B.1 Usage statistics (240 drafted distractors, 80 items)

- **Total wrong distractors tagged:** 240.
- **`no_fit` (no misconception applies):** **122 / 240 = 50.8%.**
- **Distractors carrying a real misconception id:** 118 / 240 = 49.2%.
- **Distinct misconception ids ever used:** **29 of 46**. **17 have ZERO usage.**
- Two ids absorb **25%** of all tagged distractors (`gen_wrong_punnett_ratio` 16,
  `evo_goal_directed_progress` 13).
- Mean confidence: tagged 0.75, `no_fit` 0.80 (raters are, if anything, *more* sure
  when nothing fits — consistent with "clearly filler" distractors).

**`no_fit` rate by topic** (higher = more filler / off-taxonomy content):

| Topic | Items | Distractors | `no_fit` | `no_fit` % |
|---|---:|---:|---:|---:|
| enzymes | 9 | 27 | 20 | 74% |
| cellular_respiration | 10 | 30 | 20 | 67% |
| membrane_transport | 10 | 30 | 18 | 60% |
| photosynthesis | 10 | 30 | 16 | 53% |
| evolution | 17 | 51 | 23 | 45% |
| genetics | 21 | 63 | 22 | 35% |
| experimental_design | 3 | 9 | 3 | 33% |
| **Total** | **80** | **240** | **122** | **50.8%** |

**Tag usage (all 29 used ids), most→least:**

| n | v1 id | coarse |
|---:|---|---|
| 16 | gen_wrong_punnett_ratio | reasoning_error |
| 13 | evo_goal_directed_progress | reasoning_error |
| 8 | ps_plants_dont_respire | content_gap |
| 6 | map_confuses_variable_role | misread_or_passage_mapping |
| 6 | mt_active_passive_confusion | content_gap |
| 6 | evo_for_good_of_species | reasoning_error |
| 5 | ps_o2_comes_from_co2 | content_gap |
| 5 | gen_genotype_phenotype_confusion | content_gap |
| 5 | enz_specificity_any_substrate | content_gap |
| 5 | gen_ignores_sex_linkage | reasoning_error |
| 4 | gen_alleles_blend | content_gap |
| 4 | evo_fitness_is_strength | content_gap |
| 4 | gen_recessive_disappears | content_gap |
| 3 | cr_nadh_makes_atp_directly | content_gap |
| 3 | evo_selection_creates_variation | reasoning_error |
| 3 | evo_individual_adapts | content_gap |
| 3 | mt_ignores_gradient_reasoning | reasoning_error |
| 3 | map_answers_single_trait | misread_or_passage_mapping |
| 2 | cr_etc_runs_anaerobically | content_gap |
| 2 | cr_glycolysis_needs_o2 | content_gap |
| 2 | gen_ignores_independent_assortment | reasoning_error |
| 2 | mt_hypertonic_hypotonic_swap | content_gap |
| 2 | cr_fermentation_makes_atp | reasoning_error |
| 1 | map_attributes_mass_to_solute | reasoning_error |
| 1 | ps_calvin_needs_light_directly | reasoning_error |
| 1 | mt_solute_moves_not_water | content_gap |
| 1 | enz_higher_temp_always_faster | reasoning_error |
| 1 | enz_ph_has_no_effect | content_gap |
| 1 | evo_need_drives_mutation | reasoning_error |

**Zero-usage ids (17):** `evo_use_disuse_inherited`, `gen_dominant_is_more_common`,
`gen_homozygous_heterozygous_confusion`, `cr_o2_is_electron_donor`,
`cr_o2_phosphorylates_adp`, `cr_lactate_to_pyruvate_backward`,
`ps_light_dark_reaction_swap`, `mt_osmosis_wrong_direction`,
`mt_diffusion_requires_energy`, `mt_equilibrium_stops_movement`,
`enz_consumed_in_reaction`, `enz_raise_activation_energy`, `enz_change_equilibrium`,
`enz_competitive_noncompetitive_confusion`, `map_dismisses_trend_as_error`,
`map_misses_except_not_qualifier`, `map_misreads_figure_axis`.

### B.2 Near-duplicate / overlapping clusters (behaviorally indistinguishable)

A rater looking at **one** wrong distractor cannot reliably separate these:

- **Genetics cross math** — `gen_wrong_punnett_ratio`, `gen_genotype_phenotype_confusion`,
  `gen_ignores_independent_assortment`, `gen_homozygous_heterozygous_confusion`, and
  the joint-vs-single-trait `map_answers_single_trait`. From one wrong number you
  cannot tell *which* step of the cross broke. → **`gen_cross_computation_error`.**
- **Evolution teleology** — `evo_goal_directed_progress`, `evo_for_good_of_species`,
  `evo_need_drives_mutation`, `evo_selection_creates_variation`. All are the same
  "evolution is purposeful/directed" belief wearing different clothes. →
  **`evo_teleology`.**
- **Allele behavior** — `gen_dominant_is_more_common`, `gen_recessive_disappears`,
  `gen_alleles_blend` (all predict an intermediate/absent trait). →
  **`gen_allele_behavior_model`.**
- **Osmosis/tonicity** — `mt_osmosis_wrong_direction`, `mt_hypertonic_hypotonic_swap`,
  `mt_solute_moves_not_water`, `map_attributes_mass_to_solute` (all "wrong direction
  / wrong thing moves"). → **`mt_osmosis_tonicity_confusion`.**
- **O2 in respiration** — `cr_o2_is_electron_donor`, `cr_o2_phosphorylates_adp`,
  `cr_etc_runs_anaerobically`, `cr_glycolysis_needs_o2`. →
  **`cr_o2_role_confusion`.**
- Plus smaller merges: ATP source (CR), transport-energy (MT), equilibrium reasoning
  (MT), enzyme catalysis model, enzyme conditions, light/Calvin (PS), Lamarckian
  (EVO), passage-mapping (exp. design).

### B.3 Proposed consolidated set (46 → 19) and why 19

The usage data *alone* would justify a smaller set (~13–15): only 29 ids are used
and the distribution is very concentrated. We deliberately land at **19** — the top
of the requested 12–20 band — by **retaining four pedagogically central clusters
that this harvested eval structurally under-measures** (see A.5). The number is
"as small as the data wants, plus a small retained core the harvested eval can't
exercise," not a forced target.

| # | v2 id | coarse | topic | combined support | note |
|---:|---|---|---|---:|---|
| 1 | gen_cross_computation_error | reasoning_error | genetics | 26 | biggest merge (5 ids) |
| 2 | gen_allele_behavior_model | content_gap | genetics | 8 | 3 ids |
| 3 | gen_ignores_sex_linkage | reasoning_error | genetics | 5 | retained standalone |
| 4 | evo_teleology | reasoning_error | evolution | 23 | 4 ids; 2nd biggest |
| 5 | evo_lamarckian | content_gap | evolution | 3 | 2 ids |
| 6 | evo_fitness_is_strength | content_gap | evolution | 4 | retained standalone |
| 7 | cr_o2_role_confusion | content_gap | cellular_respiration | 4 | **RETAINED-central** (4 ids, 2 zero-use) |
| 8 | cr_atp_source_confusion | content_gap | cellular_respiration | 5 | 3 ids |
| 9 | ps_plants_dont_respire | content_gap | photosynthesis | 8 | retained standalone |
| 10 | ps_o2_comes_from_co2 | content_gap | photosynthesis | 5 | retained standalone |
| 11 | ps_light_calvin_confusion | content_gap | photosynthesis | 1 | **RETAINED-central** (2 ids) |
| 12 | mt_osmosis_tonicity_confusion | content_gap | membrane_transport | 4 | 4 ids |
| 13 | mt_transport_energy_confusion | content_gap | membrane_transport | 6 | 2 ids |
| 14 | mt_equilibrium_reasoning_error | reasoning_error | membrane_transport | 3 | 2 ids |
| 15 | enz_catalysis_model_error | content_gap | enzymes | 0 | **RETAINED-central** (3 zero-use ids) |
| 16 | enz_specificity_any_substrate | content_gap | enzymes | 5 | retained standalone |
| 17 | enz_conditions_effect_error | reasoning_error | enzymes | 2 | 2 ids |
| 18 | map_confuses_variable_role | misread_or_passage_mapping | experimental_design | 6 | retained standalone |
| 19 | map_passage_stem_mapping_error | misread_or_passage_mapping | experimental_design | 0 | **RETAINED-central** (3 zero-use ids) |

**Retained despite low/zero usage (pedagogical centrality):**
- **`cr_o2_role_confusion`** — "O2 is the terminal electron acceptor" is *the*
  classic ETC concept; the eval simply lacks clean ETC-role distractors.
- **`enz_catalysis_model_error`** — "enzymes lower activation energy and are reused"
  is foundational; zero harvested support reflects filler enzyme items, not that the
  misconception is unimportant.
- **`map_passage_stem_mapping_error`** — MCAT is passage-heavy; EXCEPT/NOT and
  figure-misread errors are first-class (`mcat_pivot_spec.md` §3.1) and must be
  authored, not harvested.
- **`ps_light_calvin_confusion`** — the light/dark-reaction split is core
  photosynthesis content.

**Dropped (1):** `enz_competitive_noncompetitive_confusion` — zero support and too
fine (competitive vs noncompetitive inhibition is a narrow application distinction,
not a mid-grained misconception). If it recurs, use `enz_catalysis_model_error` or
flag `taxonomy_gap (candidate)`.

### B.4 Old → new crosswalk (all 46)

The machine-readable crosswalk lives in
`data/apbio_misconceptions_v2.proposal.json` (`crosswalk` object). Summary:

| v1 id | → v2 id |
|---|---|
| evo_individual_adapts | evo_lamarckian |
| evo_use_disuse_inherited | evo_lamarckian |
| evo_need_drives_mutation | evo_teleology |
| evo_goal_directed_progress | evo_teleology |
| evo_for_good_of_species | evo_teleology |
| evo_selection_creates_variation | evo_teleology |
| evo_fitness_is_strength | evo_fitness_is_strength *(retained)* |
| gen_dominant_is_more_common | gen_allele_behavior_model |
| gen_recessive_disappears | gen_allele_behavior_model |
| gen_alleles_blend | gen_allele_behavior_model |
| gen_wrong_punnett_ratio | gen_cross_computation_error |
| gen_genotype_phenotype_confusion | gen_cross_computation_error |
| gen_ignores_independent_assortment | gen_cross_computation_error |
| gen_homozygous_heterozygous_confusion | gen_cross_computation_error |
| gen_ignores_sex_linkage | gen_ignores_sex_linkage *(retained)* |
| cr_o2_is_electron_donor | cr_o2_role_confusion |
| cr_o2_phosphorylates_adp | cr_o2_role_confusion |
| cr_etc_runs_anaerobically | cr_o2_role_confusion |
| cr_glycolysis_needs_o2 | cr_o2_role_confusion |
| cr_fermentation_makes_atp | cr_atp_source_confusion |
| cr_nadh_makes_atp_directly | cr_atp_source_confusion |
| cr_lactate_to_pyruvate_backward | cr_atp_source_confusion |
| ps_plants_dont_respire | ps_plants_dont_respire *(retained)* |
| ps_o2_comes_from_co2 | ps_o2_comes_from_co2 *(retained)* |
| ps_calvin_needs_light_directly | ps_light_calvin_confusion |
| ps_light_dark_reaction_swap | ps_light_calvin_confusion |
| mt_osmosis_wrong_direction | mt_osmosis_tonicity_confusion |
| mt_solute_moves_not_water | mt_osmosis_tonicity_confusion |
| mt_hypertonic_hypotonic_swap | mt_osmosis_tonicity_confusion |
| map_attributes_mass_to_solute | mt_osmosis_tonicity_confusion |
| mt_active_passive_confusion | mt_transport_energy_confusion |
| mt_diffusion_requires_energy | mt_transport_energy_confusion |
| mt_equilibrium_stops_movement | mt_equilibrium_reasoning_error |
| mt_ignores_gradient_reasoning | mt_equilibrium_reasoning_error |
| enz_consumed_in_reaction | enz_catalysis_model_error |
| enz_raise_activation_energy | enz_catalysis_model_error |
| enz_change_equilibrium | enz_catalysis_model_error |
| enz_higher_temp_always_faster | enz_conditions_effect_error |
| enz_ph_has_no_effect | enz_conditions_effect_error |
| enz_specificity_any_substrate | enz_specificity_any_substrate *(retained)* |
| enz_competitive_noncompetitive_confusion | **dropped** |
| map_answers_single_trait | gen_cross_computation_error |
| map_confuses_variable_role | map_confuses_variable_role *(retained)* |
| map_dismisses_trend_as_error | map_passage_stem_mapping_error |
| map_misses_except_not_qualifier | map_passage_stem_mapping_error |
| map_misreads_figure_axis | map_passage_stem_mapping_error |

*Note on `content_gap (no misconception)`:* none of the 46 v1 ids is a disguised
content-gap bucket, so no v1 id maps to that label. `content_gap` remains a
**distractor-level verdict** (Part A), applied per distractor at tag time — it is
deliberately not a taxonomy entry.

### B.5 What owner sign-off unblocks (nothing until approved)

If approved, the follow-up (separate, non-destructive) changes would be: promote
`apbio_misconceptions_v2.proposal.json` to a new versioned taxonomy file, re-run the
`litmus_tagging` / generation verifiers against the 19-id set, and re-map the
`real_bio_eval_drafted.jsonl` draft tags through the crosswalk for a v2 usage pass.
**None of this is done here.** `apbio_misconceptions.json` and all scripts remain
untouched.
