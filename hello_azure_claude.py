"""Example: LLM-based candidate proposal using BoFire's LLMStrategy.

Demonstrates the LLMStrategy with an Azure AI Foundry endpoint,
proposing candidates for a chemical reaction optimization problem.
"""

import bofire.strategies.api as strategies
from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.llm.anthropic_foundry import AnthropicFoundryLLMProvider
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import LLMStrategy


def make_example_domain() -> Domain:
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key="temperature",
                bounds=(20.0, 200.0),
                context="Reaction temperature in Celsius. Higher temperatures "
                "generally increase reaction rate but may cause side reactions "
                "above 150°C.",
            ),
            ContinuousInput(
                key="pressure",
                bounds=(1.0, 10.0),
                context="Reactor pressure in bar. Must be balanced with "
                "temperature for safety.",
            ),
            ContinuousInput(
                key="catalyst_loading",
                bounds=(0.01, 0.5),
                context="Catalyst loading in mol%. More catalyst speeds up the "
                "reaction but is expensive. Sweet spot is typically 0.05-0.2.",
            ),
            CategoricalInput(
                key="solvent",
                categories=["water", "ethanol", "toluene", "dmso"],
                context="Reaction solvent. Polar solvents (water, dmso) favor "
                "the desired product. Ethanol is a good compromise.",
            ),
        ],
        outputs=[
            ContinuousOutput(
                key="yield",
                objective=MaximizeObjective(w=1.0),
                context="Product yield in %. Target is >90%.",
            ),
        ],
        constraints=[
            LinearInequalityConstraint(
                features=["temperature", "pressure"],
                coefficients=[1.0, 15.0],
                rhs=250.0,
                context="Safety constraint: temperature + 15*pressure must not "
                "exceed 250 to stay within reactor rating.",
            ),
        ],
    )
    domain.context = (
        "Optimizing a catalytic hydrogenation reaction for pharmaceutical "
        "intermediate production. Previous lab work suggests moderate "
        "temperatures (80-120°C) with polar solvents give the best "
        "yield/purity tradeoff."
    )
    return domain


def main():
    domain = make_example_domain()

    # Create the LLM strategy
    strategy_dm = LLMStrategy(
        domain=domain,
        llm=AnthropicFoundryLLMProvider(
            model="claude-opus-4-6",
        ),
        thinking="medium",
        n_recent_experiments=10,
        n_top_experiments=5,
    )

    # Map to functional strategy
    strategy = strategies.map(strategy_dm)

    # Ask for candidates (no prior experiments — cold start)
    print("=== Asking LLM for candidates (cold start) ===\n")
    candidates = strategy.ask(candidate_count=5)
    print("\n=== Proposed Candidates ===")
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
