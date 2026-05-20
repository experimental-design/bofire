"""Tests for the LLM-based problem formulator.

These tests verify the full pipeline: plain natural language text → classify → build → Domain.
The LLM is mocked so no API key is needed.
"""

import importlib.util
from unittest.mock import AsyncMock, patch

import pytest

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.llm.formulator_converter import convert_domain_spec
from bofire.llm.formulator_schemas import (
    ClassificationResult,
    ConstraintSpec,
    DomainSpec,
    InputSpec,
    OutputSpec,
)

PYDANTIC_AI_AVAILABLE = importlib.util.find_spec("pydantic_ai") is not None

pytestmark = pytest.mark.skipif(
    not PYDANTIC_AI_AVAILABLE,
    reason="pydantic-ai not installed",
)


# =============================================================================
# Tests for the deterministic converter (DomainSpec → Domain)
# These verify that given a correct intermediate spec, we get a valid Domain.
# =============================================================================


class TestConverter:
    """Tests for convert_domain_spec (pure Python, no LLM)."""

    def test_simple_continuous_domain(self):
        """Continuous inputs + maximize objective."""
        spec = DomainSpec(
            inputs=[
                InputSpec(
                    name="temperature",
                    type="continuous",
                    lower_bound=50.0,
                    upper_bound=200.0,
                    unit="C",
                    description="Reaction temperature",
                ),
                InputSpec(
                    name="pressure",
                    type="continuous",
                    lower_bound=1.0,
                    upper_bound=10.0,
                    unit="bar",
                ),
            ],
            outputs=[
                OutputSpec(name="yield_pct", objective="maximize"),
            ],
            context="Chemical reaction optimization",
        )
        domain = convert_domain_spec(spec)
        assert len(domain.inputs) == 2
        assert len(domain.outputs) == 1
        assert isinstance(domain.inputs.get_by_key("temperature"), ContinuousInput)
        assert domain.inputs.get_by_key("temperature").bounds == (50.0, 200.0)
        assert isinstance(
            domain.outputs.get_by_key("yield_pct").objective, MaximizeObjective
        )

    def test_mixed_input_types(self):
        """Continuous + discrete + categorical inputs."""
        spec = DomainSpec(
            inputs=[
                InputSpec(
                    name="speed", type="continuous", lower_bound=100, upper_bound=500
                ),
                InputSpec(name="num_layers", type="discrete", values=[1, 2, 3, 4, 5]),
                InputSpec(
                    name="material",
                    type="categorical",
                    categories=["steel", "aluminum", "titanium"],
                ),
            ],
            outputs=[OutputSpec(name="strength", objective="maximize")],
        )
        domain = convert_domain_spec(spec)
        assert isinstance(domain.inputs.get_by_key("speed"), ContinuousInput)
        assert isinstance(domain.inputs.get_by_key("num_layers"), DiscreteInput)
        assert isinstance(domain.inputs.get_by_key("material"), CategoricalInput)

    def test_multi_objective(self):
        """Multiple outputs with different objectives."""
        spec = DomainSpec(
            inputs=[
                InputSpec(name="x1", type="continuous", lower_bound=0, upper_bound=1),
                InputSpec(name="x2", type="continuous", lower_bound=0, upper_bound=1),
            ],
            outputs=[
                OutputSpec(name="cost", objective="minimize"),
                OutputSpec(name="quality", objective="maximize"),
                OutputSpec(
                    name="thickness",
                    objective="close_to_target",
                    target_value=2.5,
                ),
            ],
        )
        domain = convert_domain_spec(spec)
        assert isinstance(
            domain.outputs.get_by_key("cost").objective, MinimizeObjective
        )
        assert isinstance(
            domain.outputs.get_by_key("quality").objective, MaximizeObjective
        )
        obj = domain.outputs.get_by_key("thickness").objective
        assert isinstance(obj, CloseToTargetObjective)
        assert obj.target_value == 2.5

    def test_linear_equality_constraint(self):
        """Mixture constraint: fractions sum to 1."""
        spec = DomainSpec(
            inputs=[
                InputSpec(name="a", type="continuous", lower_bound=0, upper_bound=1),
                InputSpec(name="b", type="continuous", lower_bound=0, upper_bound=1),
                InputSpec(name="c", type="continuous", lower_bound=0, upper_bound=1),
            ],
            outputs=[OutputSpec(name="response", objective="maximize")],
            constraints=[
                ConstraintSpec(
                    type="linear_equality",
                    features=["a", "b", "c"],
                    coefficients=[1.0, 1.0, 1.0],
                    rhs=1.0,
                    description="Mixture fractions sum to 1",
                )
            ],
        )
        domain = convert_domain_spec(spec)
        assert len(domain.constraints) == 1
        assert isinstance(domain.constraints[0], LinearEqualityConstraint)
        assert domain.constraints[0].rhs == 1.0

    def test_linear_inequality_constraint(self):
        """Budget constraint: cost ≤ 100."""
        spec = DomainSpec(
            inputs=[
                InputSpec(name="x1", type="continuous", lower_bound=0, upper_bound=50),
                InputSpec(name="x2", type="continuous", lower_bound=0, upper_bound=50),
            ],
            outputs=[OutputSpec(name="profit", objective="maximize")],
            constraints=[
                ConstraintSpec(
                    type="linear_inequality",
                    features=["x1", "x2"],
                    coefficients=[2.0, 3.0],
                    rhs=100.0,
                )
            ],
        )
        domain = convert_domain_spec(spec)
        assert len(domain.constraints) == 1
        assert isinstance(domain.constraints[0], LinearInequalityConstraint)

    def test_missing_bounds_raises(self):
        """Continuous input without bounds should fail."""
        spec = DomainSpec(
            inputs=[InputSpec(name="x", type="continuous")],
            outputs=[OutputSpec(name="y", objective="maximize")],
        )
        with pytest.raises(ValueError, match="lower_bound and upper_bound"):
            convert_domain_spec(spec)

    def test_missing_categories_raises(self):
        """Categorical input without categories should fail."""
        spec = DomainSpec(
            inputs=[InputSpec(name="x", type="categorical")],
            outputs=[OutputSpec(name="y", objective="maximize")],
        )
        with pytest.raises(ValueError, match="categories"):
            convert_domain_spec(spec)

    def test_close_to_target_without_value_raises(self):
        """close_to_target without target_value should fail."""
        spec = DomainSpec(
            inputs=[
                InputSpec(name="x", type="continuous", lower_bound=0, upper_bound=1)
            ],
            outputs=[OutputSpec(name="y", objective="close_to_target")],
        )
        with pytest.raises(ValueError, match="target_value"):
            convert_domain_spec(spec)


# =============================================================================
# Tests for the full formulation pipeline (NL text → Domain)
# These mock the LLM to verify the orchestration logic.
# Each test starts with a PLAIN NATURAL LANGUAGE string as input.
# =============================================================================


class TestFormulatorPipeline:
    """Tests for the full formulate() pipeline with mocked LLM.

    Each test starts from a plain natural language string and verifies
    that the pipeline produces the expected Domain.
    """

    def _make_config(self):
        """Create a FormulatorConfig with a dummy provider."""
        from bofire.data_models.llm.formulator import FormulatorConfig
        from bofire.data_models.llm.provider import OpenAILLMProvider

        return FormulatorConfig(
            llm=OpenAILLMProvider(
                model="gpt-4o", api_key_env_var="FAKE_KEY_FOR_TESTING"
            ),
            max_retries=2,
        )

    def _run_formulate(self, nl_input, classification_response, domain_spec_response=None):
        """Run formulate() with mocked LLM mapper and Agent.run.

        Args:
            nl_input: The plain natural language problem description.
            classification_response: ClassificationResult the LLM would return.
            domain_spec_response: DomainSpec the LLM would return (None for not_experimental).

        Returns:
            FormulationResult from the pipeline.
        """
        from unittest.mock import MagicMock, patch

        from pydantic_ai.models.test import TestModel

        from bofire.llm.formulator import formulate

        call_count = {"n": 0}

        async def mock_agent_run(self_agent, prompt, **kwargs):
            nonlocal call_count
            call_count["n"] += 1

            class FakeResult:
                pass

            result = FakeResult()
            if call_count["n"] == 1:
                result.output = classification_response
            else:
                result.output = domain_spec_response
            return result

        # Return a real TestModel instance so pydantic-ai doesn't reject it
        mock_map = MagicMock(return_value=TestModel())

        with (
            patch("bofire.llm.formulator.llm_mapper.map", mock_map),
            patch("pydantic_ai.Agent.run", mock_agent_run),
        ):
            return formulate(self._make_config(), nl_input)

    def test_chemical_reaction_doe(self):
        """NL: 'Optimize a chemical reaction' → initial_design → Domain with continuous inputs."""
        nl_input = (
            "I want to design experiments for a chemical reaction. "
            "I can control temperature (50 to 200 degrees C), "
            "pressure (1 to 10 bar), and catalyst loading (0.1 to 5 percent). "
            "I want to maximize the reaction yield."
        )

        result = self._run_formulate(
            nl_input,
            classification_response=ClassificationResult(
                category="initial_design",
                reasoning="User wants to design experiments with 3 continuous factors.",
            ),
            domain_spec_response=DomainSpec(
                inputs=[
                    InputSpec(name="temperature", type="continuous", lower_bound=50.0, upper_bound=200.0, unit="C"),
                    InputSpec(name="pressure", type="continuous", lower_bound=1.0, upper_bound=10.0, unit="bar"),
                    InputSpec(name="catalyst_loading", type="continuous", lower_bound=0.1, upper_bound=5.0, unit="percent"),
                ],
                outputs=[OutputSpec(name="yield_pct", objective="maximize")],
                context="Chemical reaction optimization",
            ),
        )

        assert result.domain is not None
        assert result.classification == "initial_design"
        assert len(result.domain.inputs) == 3
        assert len(result.domain.outputs) == 1
        assert isinstance(result.domain.inputs.get_by_key("temperature"), ContinuousInput)
        assert result.domain.inputs.get_by_key("temperature").bounds == (50.0, 200.0)
        assert isinstance(result.domain.outputs.get_by_key("yield_pct").objective, MaximizeObjective)

    def test_multi_objective_drug_formulation(self):
        """NL: 'Drug formulation with competing objectives' → multi_objective → Domain."""
        nl_input = (
            "We're developing a tablet formulation. We can vary the amount of "
            "excipient A (0-50mg), excipient B (0-30mg), and compression force "
            "(5-20 kN). We want to maximize dissolution rate while minimizing "
            "friability. The tablet hardness should be close to 8 kP."
        )

        result = self._run_formulate(
            nl_input,
            classification_response=ClassificationResult(
                category="multi_objective",
                reasoning="Multiple competing objectives: dissolution, friability, hardness target.",
            ),
            domain_spec_response=DomainSpec(
                inputs=[
                    InputSpec(name="excipient_a", type="continuous", lower_bound=0, upper_bound=50, unit="mg"),
                    InputSpec(name="excipient_b", type="continuous", lower_bound=0, upper_bound=30, unit="mg"),
                    InputSpec(name="compression_force", type="continuous", lower_bound=5, upper_bound=20, unit="kN"),
                ],
                outputs=[
                    OutputSpec(name="dissolution_rate", objective="maximize"),
                    OutputSpec(name="friability", objective="minimize"),
                    OutputSpec(name="hardness", objective="close_to_target", target_value=8.0),
                ],
            ),
        )

        assert result.domain is not None
        assert result.classification == "multi_objective"
        assert len(result.domain.outputs) == 3
        assert isinstance(result.domain.outputs.get_by_key("dissolution_rate").objective, MaximizeObjective)
        assert isinstance(result.domain.outputs.get_by_key("friability").objective, MinimizeObjective)
        obj = result.domain.outputs.get_by_key("hardness").objective
        assert isinstance(obj, CloseToTargetObjective)
        assert obj.target_value == 8.0

    def test_not_experimental_gives_warning(self):
        """NL: 'Assign workers to tasks' → not_experimental → warning, no Domain."""
        from unittest.mock import MagicMock, patch

        from pydantic_ai.models.test import TestModel

        from bofire.llm.formulator import formulate

        nl_input = (
            "I need to assign 5 workers to 5 tasks. Each worker can do exactly "
            "one task. The cost of assigning worker i to task j is given by a "
            "5x5 cost matrix. Minimize total assignment cost."
        )

        classification_response = ClassificationResult(
            category="not_experimental",
            reasoning="This is a combinatorial assignment problem (MILP), not a physical experiment.",
        )

        async def mock_agent_run(self_agent, prompt, **kwargs):
            class FakeResult:
                pass
            result = FakeResult()
            result.output = classification_response
            return result

        mock_map = MagicMock(return_value=TestModel())

        with (
            patch("bofire.llm.formulator.llm_mapper.map", mock_map),
            patch("pydantic_ai.Agent.run", mock_agent_run),
        ):
            with pytest.warns(UserWarning, match="not_experimental"):
                result = formulate(self._make_config(), nl_input)

        assert result.domain is None
        assert result.classification == "not_experimental"
        assert "assignment" in result.reasoning.lower()

    def test_mixture_design_with_constraints(self):
        """NL: 'Mixture with fractions summing to 1' → initial_design → Domain with constraint."""
        nl_input = (
            "Design experiments for a 3-component polymer blend. "
            "Component A (0-100%), B (0-100%), C (0-100%). "
            "The three fractions must sum to 100%. "
            "Maximize tensile strength."
        )

        result = self._run_formulate(
            nl_input,
            classification_response=ClassificationResult(
                category="initial_design",
                reasoning="Mixture design with sum-to-one constraint.",
            ),
            domain_spec_response=DomainSpec(
                inputs=[
                    InputSpec(name="comp_a", type="continuous", lower_bound=0, upper_bound=1, unit="fraction"),
                    InputSpec(name="comp_b", type="continuous", lower_bound=0, upper_bound=1, unit="fraction"),
                    InputSpec(name="comp_c", type="continuous", lower_bound=0, upper_bound=1, unit="fraction"),
                ],
                outputs=[OutputSpec(name="tensile_strength", objective="maximize")],
                constraints=[
                    ConstraintSpec(
                        type="linear_equality",
                        features=["comp_a", "comp_b", "comp_c"],
                        coefficients=[1.0, 1.0, 1.0],
                        rhs=1.0,
                    )
                ],
            ),
        )

        assert result.domain is not None
        assert len(result.domain.constraints) == 1
        assert isinstance(result.domain.constraints[0], LinearEqualityConstraint)
        assert result.domain.constraints[0].rhs == 1.0

    def test_categorical_inputs(self):
        """NL: 'Compare 3 catalysts' → initial_design → Domain with categorical."""
        nl_input = (
            "I want to screen catalysts for a hydrogenation reaction. "
            "Catalysts to test: Pd/C, Pt/Al2O3, Raney-Ni. "
            "Temperature range 50-150C, hydrogen pressure 1-5 bar. "
            "Maximize conversion."
        )

        result = self._run_formulate(
            nl_input,
            classification_response=ClassificationResult(
                category="initial_design",
                reasoning="Screening experiment with categorical catalyst choice.",
            ),
            domain_spec_response=DomainSpec(
                inputs=[
                    InputSpec(name="catalyst", type="categorical", categories=["Pd_C", "Pt_Al2O3", "Raney_Ni"]),
                    InputSpec(name="temperature", type="continuous", lower_bound=50, upper_bound=150, unit="C"),
                    InputSpec(name="h2_pressure", type="continuous", lower_bound=1, upper_bound=5, unit="bar"),
                ],
                outputs=[OutputSpec(name="conversion", objective="maximize")],
            ),
        )

        assert result.domain is not None
        cat_input = result.domain.inputs.get_by_key("catalyst")
        assert isinstance(cat_input, CategoricalInput)
        assert set(cat_input.categories) == {"Pd_C", "Pt_Al2O3", "Raney_Ni"}

    def test_single_objective_with_constraint(self):
        """NL: realistic BO problem with existing data and a linear constraint."""
        nl_input = (
            "Using the experimental data we already collected, find the temperature "
            "and catalyst concentration that maximize production yield, while keeping "
            "the combined operating load under 100."
        )

        result = self._run_formulate(
            nl_input,
            classification_response=ClassificationResult(
                category="single_objective",
                reasoning=(
                    "User has existing data and wants to maximize a single output "
                    "(production yield) subject to a linear constraint on inputs."
                ),
            ),
            domain_spec_response=DomainSpec(
                inputs=[
                    InputSpec(name="temperature", type="continuous", lower_bound=20, upper_bound=200),
                    InputSpec(name="catalyst_concentration", type="continuous", lower_bound=0, upper_bound=100),
                ],
                outputs=[OutputSpec(name="production_yield", objective="maximize")],
                constraints=[
                    ConstraintSpec(
                        type="linear_inequality",
                        features=["temperature", "catalyst_concentration"],
                        coefficients=[1.0, 1.0],
                        rhs=100.0,
                    )
                ],
                context="Bayesian optimization with prior experimental data.",
            ),
        )

        assert result.domain is not None
        assert result.classification == "single_objective"
        # 2 continuous inputs
        assert len(result.domain.inputs) == 2
        assert isinstance(result.domain.inputs.get_by_key("temperature"), ContinuousInput)
        assert isinstance(result.domain.inputs.get_by_key("catalyst_concentration"), ContinuousInput)
        # 1 output with maximize
        assert len(result.domain.outputs) == 1
        assert isinstance(result.domain.outputs.get_by_key("production_yield").objective, MaximizeObjective)
        # 1 linear inequality constraint
        assert len(result.domain.constraints) == 1
        assert isinstance(result.domain.constraints[0], LinearInequalityConstraint)
        assert result.domain.constraints[0].rhs == 100.0


