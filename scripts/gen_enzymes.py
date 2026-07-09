"""
By-construction generator for AP Bio ENZYMES items (v1) — conceptual/role-swap.

Third topic, chosen over photosynthesis because enzymes have 7 drafted
misconceptions that compete cleanly on shared stems (how an enzyme speeds a
reaction: lowers Ea vs raises Ea vs shifts equilibrium vs is consumed), giving
frames the >=4-competing-misconceptions combinatorics photosynthesis (4) can't.
Uses the shared frame engine in conceptual_engine.py.
"""

from __future__ import annotations

import conceptual_engine as engine

TOPIC = "enzymes"

ERROR_TYPE = {
    "enz_consumed_in_reaction": "content_gap",
    "enz_raise_activation_energy": "content_gap",
    "enz_change_equilibrium": "content_gap",
    "enz_higher_temp_always_faster": "reasoning_error",
    "enz_specificity_any_substrate": "content_gap",
    "enz_ph_has_no_effect": "content_gap",
}

FRAMES = [
    {
        "id": "mechanism",
        "subtopic": "catalysis_mechanism",
        "stems": [
            "How does an enzyme increase the rate of a chemical reaction?",
            "By what mechanism does an enzyme speed up a reaction?",
        ],
        "correct": [
            "It lowers the activation energy required for the reaction to proceed.",
            "It reduces the activation-energy barrier, so the reaction happens faster.",
        ],
        "distractors": {
            "enz_raise_activation_energy": [
                "It raises the activation energy needed for the reaction.",
                "It increases the activation-energy barrier of the reaction.",
            ],
            "enz_change_equilibrium": [
                "It shifts the reaction's equilibrium toward the products.",
                "It changes the equilibrium so more product is favored.",
            ],
            "enz_consumed_in_reaction": [
                "It is consumed as a reactant during the reaction it speeds up.",
                "It is used up in the reaction, so it must be continuously replaced.",
            ],
            "enz_specificity_any_substrate": [
                "It binds and converts any available molecule to force the reaction.",
                "It acts on whatever substrate is present, regardless of its shape.",
            ],
        },
    },
    # NOTE: an "enzyme_fate" frame was removed after human review — its
    # enz_change_equilibrium distractors ("converted into a product molecule")
    # read as enz_consumed_in_reaction, so the tag was not reliably separable on
    # a "fate of the enzyme" stem. See docs/behavior_spec.md quality gate.
    {
        "id": "temp_ph_factors",
        "subtopic": "environmental_factors",
        "stems": [
            "How do temperature and pH affect enzyme activity?",
            "Which statement about temperature and pH effects on enzymes is correct?",
        ],
        "correct": [
            "Each has an optimum; beyond it the enzyme denatures and activity falls.",
            "Activity peaks at an optimal temperature and pH, then drops as the enzyme denatures.",
        ],
        "distractors": {
            "enz_higher_temp_always_faster": [
                "Raising the temperature always increases enzyme activity.",
                "Higher temperature makes the enzyme faster without any limit.",
            ],
            "enz_ph_has_no_effect": [
                "pH has no effect on how the enzyme works.",
                "Changing the pH does not influence enzyme activity.",
            ],
            "enz_change_equilibrium": [
                "They act only by shifting the reaction's equilibrium.",
                "Their effect is to move the equilibrium rather than change enzyme shape.",
            ],
        },
    },
    {
        "id": "specificity",
        "subtopic": "active_site_specificity",
        "stems": [
            "What best describes enzyme specificity?",
            "What does it mean that enzymes are specific?",
        ],
        "correct": [
            "The active site's shape fits only a particular substrate or reaction.",
            "An enzyme's active site matches one specific substrate, so it catalyzes one reaction.",
        ],
        "distractors": {
            "enz_specificity_any_substrate": [
                "An enzyme can catalyze essentially any reaction it encounters.",
                "Enzymes act on any substrate, not just a particular one.",
            ],
            "enz_change_equilibrium": [
                "Specificity means the enzyme sets the reaction's final equilibrium.",
                "Being specific means it fixes where the equilibrium lies.",
            ],
            "enz_consumed_in_reaction": [
                "Specificity means the enzyme is consumed by its one substrate.",
                "It means one substrate uses the enzyme up completely.",
            ],
        },
    },
    {
        "id": "thermodynamics",
        "subtopic": "what_enzymes_cannot_do",
        "stems": [
            "Which statement about what enzymes can and cannot do is correct?",
            "What is a correct limit on what an enzyme does to a reaction?",
        ],
        "correct": [
            "They speed a reaction without changing its overall free-energy change or equilibrium.",
            "They accelerate a reaction but do not alter its net energy change or equilibrium position.",
        ],
        "distractors": {
            "enz_change_equilibrium": [
                "They shift the reaction's equilibrium toward the products.",
                "They change where the reaction's equilibrium lies.",
            ],
            "enz_raise_activation_energy": [
                "They raise the activation energy to drive the reaction forward.",
                "They push reactions forward by increasing the activation energy.",
            ],
            "enz_higher_temp_always_faster": [
                "They make reactions faster without limit as temperature keeps rising.",
                "They speed reactions ever faster the hotter it gets, without bound.",
            ],
            "enz_consumed_in_reaction": [
                "They are consumed, so the cell must constantly synthesize more.",
                "They get used up in the reactions they drive.",
            ],
        },
    },
]


if __name__ == "__main__":
    engine.run(frames=FRAMES, topic=TOPIC, error_type=ERROR_TYPE,
               id_prefix="gen_enzymes", generator="scripts/gen_enzymes.py",
               default_out="data/gen_enzymes.jsonl")
