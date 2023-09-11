from typing import Tuple

import numpy as np
import pyomo.environ as pyo
from entmoot.problem_config import ProblemConfig

from bofire.data_models.constraints.api import (
    AnyConstraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
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

    model_pyo = problem_config.get_pyomo_model_core()
    model_pyo.constr = pyo.ConstraintList()
    for constraint in domain.constraints:
        _bofire_constraint_to_entmoot(problem_config, constraint, model_pyo)

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
    if isinstance(feature.objective, (MinimizeObjective, MaximizeObjective)):
        problem_config.add_min_objective(name=feature.key)

    else:
        raise NotImplementedError(f"Did not recognise output {feature}")


def _bofire_constraint_to_entmoot(
    problem_config: ProblemConfig,
    constraint: AnyConstraint,
    model_core: pyo.ConcreteModel,
) -> None:
    """Apply a Bofire `Constraint` to an ENTMOOT model.

    To apply a constraint, the Pyomo model must be accessed. A reference to this model
    core should be retained to keep the constraints.

    Args:
        problem_config (ProblemConfig): An ENTMOOT problem definition.
        constraint (AnyConstraint): A constraint to be applied to the Pyomo model.
        model_core (pyo.ConcreteModel): The underlying solver model.
    """
    if not isinstance(constraint, LinearConstraint):
        raise NotImplementedError("Non-linear constraints are not supported")

    # get references to the Pyomo variables to create the constraint, keeping
    # the order of the variables in the Constraint

    feat_keys = [feat.name for feat in problem_config.feat_list]
    feat_idxs = [feat_keys.index(key) for key in constraint.features]
    features = [model_core._all_feat[i] for i in feat_idxs]

    lhs = sum(feat * coeff for feat, coeff in zip(features, constraint.coefficients))
    if isinstance(constraint, LinearEqualityConstraint):
        constr_expr = lhs == constraint.rhs
    elif isinstance(constraint, LinearInequalityConstraint):
        constr_expr = lhs <= constraint.rhs
    else:
        raise NotImplementedError(f"Did not recognise constraint {constraint.type}")
    model_core.constr.add(expr=constr_expr)
