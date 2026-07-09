"""
By-construction generator for AP Bio EVOLUTION items (conceptual/role-swap).

Added in the Path-B data-iteration step: the first tune generalized poorly to
unseen topics, so we broaden training-topic coverage. Evolution's misconceptions
(teleology + Lamarckism cluster) compete cleanly on natural-selection stems,
making strong >=3-misconception frames. Uses conceptual_engine.py.
"""

from __future__ import annotations

import conceptual_engine as engine

TOPIC = "evolution"

ERROR_TYPE = {
    "evo_individual_adapts": "content_gap",
    "evo_need_drives_mutation": "reasoning_error",
    "evo_use_disuse_inherited": "content_gap",
    "evo_goal_directed_progress": "reasoning_error",
    "evo_fitness_is_strength": "content_gap",
    "evo_for_good_of_species": "reasoning_error",
    "evo_selection_creates_variation": "reasoning_error",
}

FRAMES = [
    {
        "id": "antibiotic_resistance",
        "subtopic": "natural_selection",
        "stems": [
            "A bacterial population is treated with an antibiotic; most die but a resistant subpopulation grows back. What best explains how the population became resistant?",
            "After antibiotic exposure, a bacterial population is now mostly resistant. Which explanation is correct?",
        ],
        "correct": [
            "Rare resistant variants were already present and survived to reproduce, so their alleles increased.",
            "Pre-existing resistant variants were selected because they survived and passed on their alleles.",
        ],
        "distractors": {
            "evo_individual_adapts": [
                "Individual bacteria changed their own genes to survive the antibiotic.",
                "Each bacterium adapted during its lifetime to become resistant.",
            ],
            "evo_need_drives_mutation": [
                "The bacteria mutated because they needed resistance to survive.",
                "Exposure caused the specific mutations the bacteria required.",
            ],
            "evo_use_disuse_inherited": [
                "Bacteria that used resistance mechanisms passed that strengthening to offspring.",
                "Resistance built up through use and was inherited by the next generation.",
            ],
            "evo_selection_creates_variation": [
                "The antibiotic (natural selection) created the new resistance variation.",
                "Selection itself generated the resistance trait that spread.",
            ],
        },
    },
    {
        "id": "meaning_of_fitness",
        "subtopic": "fitness",
        "stems": [
            "What does evolutionary 'fitness' mean?",
            "In evolutionary terms, an organism's fitness refers to what?",
        ],
        "correct": [
            "Its relative reproductive success in a particular environment.",
            "How many surviving offspring it produces relative to others in its environment.",
        ],
        "distractors": {
            "evo_fitness_is_strength": [
                "How physically strong or fast the organism is.",
                "Being the strongest or most athletic individual.",
            ],
            "evo_goal_directed_progress": [
                "How advanced or complex the organism has become.",
                "How far it has progressed up the evolutionary ladder.",
            ],
            "evo_for_good_of_species": [
                "How much the organism benefits the survival of its whole species.",
                "How useful the organism is to the good of the species.",
            ],
        },
    },
    {
        "id": "direction_of_evolution",
        "subtopic": "misconceptions_of_direction",
        "stems": [
            "Which statement about the direction of evolution is correct?",
            "How should the 'direction' of evolution be understood?",
        ],
        "correct": [
            "Evolution has no goal; it reflects differential reproductive success in a context.",
            "There is no intended endpoint — allele frequencies shift with reproductive success.",
        ],
        "distractors": {
            "evo_goal_directed_progress": [
                "Evolution progresses steadily toward more perfect, complex organisms.",
                "Evolution is a ladder advancing life toward higher forms.",
            ],
            "evo_for_good_of_species": [
                "Traits evolve because they benefit the species as a whole.",
                "Evolution works to preserve the good of the species.",
            ],
            "evo_need_drives_mutation": [
                "Organisms evolve the specific traits they need when they need them.",
                "Need directs which mutations appear.",
            ],
        },
    },
    {
        "id": "source_of_variation",
        "subtopic": "variation",
        "stems": [
            "Where does the heritable variation that natural selection acts on come from?",
            "What is the origin of the variation selection operates on?",
        ],
        "correct": [
            "Random mutation and recombination produce pre-existing heritable variation.",
            "It arises randomly from mutation and recombination before selection acts.",
        ],
        "distractors": {
            "evo_selection_creates_variation": [
                "Natural selection creates the new variation it then favors.",
                "Selection generates the variants rather than acting on existing ones.",
            ],
            "evo_need_drives_mutation": [
                "Mutations arise specifically because the organism needs them.",
                "The environment triggers exactly the mutations required.",
            ],
            "evo_use_disuse_inherited": [
                "It comes from traits strengthened through use and passed on.",
                "Variation is produced by use and disuse during an organism's life.",
            ],
        },
    },
]


if __name__ == "__main__":
    engine.run(frames=FRAMES, topic=TOPIC, error_type=ERROR_TYPE,
               id_prefix="gen_evolution", generator="scripts/gen_evolution.py",
               default_out="data/gen_evolution.jsonl")
