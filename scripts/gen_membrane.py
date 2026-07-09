"""
By-construction generator for AP Bio MEMBRANE TRANSPORT items (conceptual).

Added in the Path-B data-iteration step (broaden training-topic coverage).
Osmosis / diffusion / active-vs-passive misconceptions compete cleanly on
shared stems. Uses conceptual_engine.py.
"""

from __future__ import annotations

import conceptual_engine as engine

TOPIC = "membrane_transport"

ERROR_TYPE = {
    "mt_osmosis_wrong_direction": "content_gap",
    "mt_solute_moves_not_water": "content_gap",
    "mt_hypertonic_hypotonic_swap": "content_gap",
    "mt_active_passive_confusion": "content_gap",
    "mt_diffusion_requires_energy": "content_gap",
    "mt_equilibrium_stops_movement": "reasoning_error",
    "mt_ignores_gradient_reasoning": "reasoning_error",
}

FRAMES = [
    {
        "id": "hypertonic_cell",
        "subtopic": "osmosis_direction",
        "stems": [
            "An animal cell is placed in a hypertonic solution. What happens to water, and why?",
            "A cell sits in a solution with higher solute concentration outside. What does water do?",
        ],
        "correct": [
            "Water leaves the cell, moving toward the higher solute concentration outside.",
            "Water moves out of the cell down its own concentration gradient toward the solute.",
        ],
        "distractors": {
            "mt_osmosis_wrong_direction": [
                "Water moves into the cell, toward the lower solute concentration.",
                "Water enters the cell, flowing toward where solute is less concentrated.",
            ],
            "mt_hypertonic_hypotonic_swap": [
                "Water enters, because 'hypertonic' means more water is outside the cell.",
                "Water flows in, since a hypertonic solution has the most water.",
            ],
            "mt_solute_moves_not_water": [
                "Solute moves into the cell to balance the concentrations.",
                "The dissolved solute crosses in to equalize both sides.",
            ],
        },
    },
    {
        "id": "what_is_osmosis",
        "subtopic": "osmosis_definition",
        "stems": [
            "Which statement best defines osmosis?",
            "What is osmosis?",
        ],
        "correct": [
            "The diffusion of water across a selectively permeable membrane down its gradient.",
            "Net movement of water across a membrane from lower to higher solute concentration.",
        ],
        "distractors": {
            "mt_solute_moves_not_water": [
                "The movement of solute across the membrane to balance concentrations.",
                "Dissolved particles crossing the membrane to equalize both sides.",
            ],
            "mt_osmosis_wrong_direction": [
                "The movement of water toward the lower solute concentration.",
                "Water flowing from high solute toward low solute.",
            ],
            "mt_diffusion_requires_energy": [
                "An energy-requiring pumping of water across the membrane.",
                "Active, ATP-driven transport of water across the membrane.",
            ],
        },
    },
    {
        "id": "passive_transport",
        "subtopic": "active_vs_passive",
        "stems": [
            "Which statement about passive transport is correct?",
            "What is true of transport down a concentration gradient?",
        ],
        "correct": [
            "It moves substances down their gradient and requires no cellular energy.",
            "It needs no ATP because it follows the concentration gradient.",
        ],
        "distractors": {
            "mt_diffusion_requires_energy": [
                "Diffusion down a gradient still requires ATP input.",
                "Moving down a gradient consumes cellular energy.",
            ],
            "mt_active_passive_confusion": [
                "Facilitated diffusion is active transport because it uses proteins.",
                "Using a membrane protein makes the process active transport.",
            ],
            "mt_equilibrium_stops_movement": [
                "Once equilibrium is reached, all molecular movement stops.",
                "At equilibrium molecules stop moving entirely.",
            ],
        },
    },
    {
        "id": "equilibrium",
        "subtopic": "dynamic_equilibrium",
        "stems": [
            "What is happening at equilibrium across a membrane?",
            "Which statement about diffusion equilibrium is correct?",
        ],
        "correct": [
            "There is no NET movement, but individual molecules keep crossing both ways.",
            "Molecules still move both directions; the net flux is zero.",
        ],
        "distractors": {
            "mt_equilibrium_stops_movement": [
                "All molecular movement ceases once equilibrium is reached.",
                "Molecules stop moving entirely at equilibrium.",
            ],
            "mt_ignores_gradient_reasoning": [
                "Nothing further can be said, since the concentrations are now identical.",
                "The gradient is gone, so there is simply no movement to reason about.",
            ],
            "mt_solute_moves_not_water": [
                "Solute alone shuttles back and forth to hold the balance.",
                "Only the dissolved solute keeps moving to maintain equilibrium.",
            ],
        },
    },
]


if __name__ == "__main__":
    engine.run(frames=FRAMES, topic=TOPIC, error_type=ERROR_TYPE,
               id_prefix="gen_membrane", generator="scripts/gen_membrane.py",
               default_out="data/gen_membrane.jsonl")
