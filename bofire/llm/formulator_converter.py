"""Deterministic converter from DomainSpec to BoFire Domain.

This module contains pure Python logic (no LLM calls) that maps the simplified
intermediate schemas produced by the LLM into proper BoFire Domain objects.
"""

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.constraints.constraint import Constraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.domain.constraints import Constraints
from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.features.feature import Input
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.llm.formulator_schemas import (
    ConstraintSpec,
    DomainSpec,
    InputSpec,
    OutputSpec,
)


def _convert_input(spec: InputSpec) -> Input:
    """Convert a single InputSpec to a BoFire Input feature."""
    if spec.type == "continuous":
        if spec.lower_bound is None or spec.upper_bound is None:
            raise ValueError(
                f"Input '{spec.name}': continuous type requires "
                f"lower_bound and upper_bound."
            )
        return ContinuousInput(
            key=spec.name,
            bounds=(spec.lower_bound, spec.upper_bound),
            unit=spec.unit,
            context=spec.description,
        )
    elif spec.type == "discrete":
        if spec.values is None or len(spec.values) == 0:
            raise ValueError(
                f"Input '{spec.name}': discrete type requires a non-empty 'values' list."
            )
        return DiscreteInput(
            key=spec.name,
            values=spec.values,
            unit=spec.unit,
            context=spec.description,
        )
    elif spec.type == "categorical":
        if spec.categories is None or len(spec.categories) == 0:
            raise ValueError(
                f"Input '{spec.name}': categorical type requires "
                f"a non-empty 'categories' list."
            )
        return CategoricalInput(
            key=spec.name,
            categories=spec.categories,
            context=spec.description,
        )
    else:
        raise ValueError(f"Input '{spec.name}': unknown type '{spec.type}'.")


def _convert_output(spec: OutputSpec) -> ContinuousOutput:
    """Convert a single OutputSpec to a BoFire ContinuousOutput."""
    if spec.objective == "maximize":
        objective = MaximizeObjective(w=1.0)
    elif spec.objective == "minimize":
        objective = MinimizeObjective(w=1.0)
    elif spec.objective == "close_to_target":
        if spec.target_value is None:
            raise ValueError(
                f"Output '{spec.name}': close_to_target objective requires target_value."
            )
        objective = CloseToTargetObjective(
            w=1.0, target_value=spec.target_value, exponent=2.0
        )
    else:
        raise ValueError(
            f"Output '{spec.name}': unknown objective '{spec.objective}'."
        )

    return ContinuousOutput(
        key=spec.name,
        objective=objective,
        context=spec.description,
    )


def _convert_constraint(spec: ConstraintSpec) -> Constraint:
    """Convert a single ConstraintSpec to a BoFire Constraint."""
    if len(spec.features) != len(spec.coefficients):
        raise ValueError(
            f"Constraint has {len(spec.features)} features but "
            f"{len(spec.coefficients)} coefficients."
        )
    if len(spec.features) < 2:
        raise ValueError("Constraints require at least 2 features.")

    if spec.type == "linear_equality":
        return LinearEqualityConstraint(
            features=spec.features,
            coefficients=spec.coefficients,
            rhs=spec.rhs,
            context=spec.description,
        )
    elif spec.type == "linear_inequality":
        return LinearInequalityConstraint(
            features=spec.features,
            coefficients=spec.coefficients,
            rhs=spec.rhs,
            context=spec.description,
        )
    else:
        raise ValueError(f"Unknown constraint type: '{spec.type}'.")


def convert_domain_spec(spec: DomainSpec) -> Domain:
    """Convert a DomainSpec to a BoFire Domain.

    Args:
        spec: The intermediate domain specification produced by the LLM.

    Returns:
        A fully validated BoFire Domain.

    Raises:
        ValueError: If the spec contains invalid or inconsistent data.
    """
    inputs = [_convert_input(i) for i in spec.inputs]
    outputs = [_convert_output(o) for o in spec.outputs]
    constraints: list[Constraint] = [_convert_constraint(c) for c in spec.constraints]

    return Domain(
        inputs=Inputs(features=inputs),
        outputs=Outputs(features=outputs),
        constraints=Constraints(constraints=constraints),
        context=spec.context,
    )
