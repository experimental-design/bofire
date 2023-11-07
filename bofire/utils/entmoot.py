from typing import Tuple

import entmoot.constraints as entconstr
import numpy as np
import pyomo.environ as pyo
from entmoot import ProblemConfig

from bofire.data_models.constraints.api import (
    AnyConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    AnyInput,
    AnyOutput,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective


def domain_to_problem_config(domain: Domain) -> Tuple[ProblemConfig, pyo.ConcreteModel]:
    """Convert a set of features and constraints from BoFire to ENTMOOT.

    Problems in BoFire are defined as `Domain`s. Before running an ENTMOOT strategy,
    the problem must be converted to an `entmoot.ProblemConfig`.

    Args:
        domain (Domain): the definition of the optimization problem.

    Returns:
        A tuple (problem_config, model_pyo), where problem_config is the problem definition
        in an ENTMOOT-friendly format, and model_pyo is the Pyomo model containing constraints.
    """
    problem_config = ProblemConfig()

    for input_feature in domain.inputs:
        _bofire_feat_to_entmoot(problem_config, input_feature)

    for output_feature in domain.outputs:
        _bofire_output_to_entmoot(problem_config, output_feature)

    constraints = []
    for constraint in domain.constraints:
        constraints.append(_bofire_constraint_to_entmoot(problem_config, constraint))

    # apply constraints to model
    model_pyo = problem_config.get_pyomo_model_core()
    model_pyo.problem_constraints = pyo.ConstraintList()
    entconstr.ConstraintList(constraints).apply_pyomo_constraints(
        model_pyo, problem_config.feat_list, model_pyo.problem_constraints
    )

    return problem_config, model_pyo


def _bofire_feat_to_entmoot(problem_config: ProblemConfig, feature: AnyInput) -> None:
    """Given a Bofire `Input`, create an ENTMOOT `FeatureType`.

    Args:
        problem_config (ProblemConfig): An ENTMOOT problem definition, modified in-place.
        feature (AnyInput): An input feature to be added to the problem_config object.
    """
    feat_type = None
    bounds = None
    name = feature.key

    if isinstance(feature, ContinuousInput):
        feat_type = "real"
        bounds = (feature.lower_bound, feature.upper_bound)

    elif isinstance(feature, DiscreteInput):
        x = feature.values
        assert (
            np.all(np.diff(x) == 1) and x[0] % 1 == 0
        ), "Discrete values must be consecutive integers"
        feat_type = "binary" if np.array_equal(x, np.array([0, 1])) else "integer"
        bounds = (int(feature.lower_bound), int(feature.upper_bound))

    elif isinstance(feature, CategoricalInput):
        feat_type = "categorical"
        bounds = tuple(feature.categories)

    else:
        raise NotImplementedError(f"Did not recognise input {feature}")

    problem_config.add_feature(feat_type, bounds, name)


def _bofire_output_to_entmoot(
    problem_config: ProblemConfig, feature: AnyOutput
) -> None:
    """Given a Bofire `Output`, create an ENTMOOT `MinObjective`.

    If the output feature has a maximise objective, this is added to the problem config as a
    `MinObjective`, and a factor of -1 is introduced in `EntingStrategy`.

    Args:
        problem_config (ProblemConfig): An ENTMOOT problem definition, modified in-place.
        feature (AnyOutput): An output feature to be added to the problem_config object.
    """
    if isinstance(feature.objective, MinimizeObjective):
        problem_config.add_min_objective(name=feature.key)

    elif isinstance(feature.objective, MaximizeObjective):
        problem_config.add_max_objective(name=feature.key)

    else:
        raise NotImplementedError(f"Did not recognise output {feature}")


def _bofire_constraint_to_entmoot(
    problem_config: ProblemConfig,
    constraint: AnyConstraint,
) -> None:
    """Convert a Bofire `Constraint` to an ENTMOOT `Constraint`.

    Args:
        problem_config (ProblemConfig): An ENTMOOT problem definition.
        constraint (AnyConstraint): A constraint to be applied to the Pyomo model.
    """

    if isinstance(constraint, LinearEqualityConstraint):
        ent_constraint = entconstr.LinearEqualityConstraint(
            feature_keys=constraint.features,
            coefficients=constraint.coefficients,
            rhs=constraint.rhs,
        )

    elif isinstance(constraint, LinearInequalityConstraint):
        ent_constraint = entconstr.LinearEqualityConstraint(
            feature_keys=constraint.features,
            coefficients=constraint.coefficients,
            rhs=constraint.rhs,
        )

    elif isinstance(constraint, NChooseKConstraint):
        ent_constraint = entconstr.NChooseKConstraint(
            feature_keys=constraint.features,
            min_count=constraint.min_count,
            max_count=constraint.max_count,
            none_also_valid=constraint.none_also_valid,
        )

    else:
        raise NotImplementedError("Only linear and nchoosek constraints are supported.")

    return ent_constraint
