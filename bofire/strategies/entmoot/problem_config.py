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

# from bofire.data_models.domain.constraints import Constraints
# from bofire.data_models.domain.features import Features, Inputs, Outputs
from bofire.data_models.features.api import (
    AnyInput,
    AnyOutput,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective


def domain_to_problem_config(domain: Domain) -> ProblemConfig:
    """Convert a set of features and constraints from BoFire to Entmoot"""
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


def _bofire_feat_to_entmoot(problem_config: ProblemConfig, feature: AnyInput):
    """Given a Bofire feature, create an entmoot feature"""
    feat_type = None
    bounds = None
    name = feature.key

    if isinstance(feature, ContinuousInput):
        feat_type = "real"
        bounds = (feature.lower_bound, feature.upper_bound)

    elif isinstance(feature, DiscreteInput):
        x = feature.values
        print(x)
        assert (
            np.all(np.diff(x) == 1) and x[0] % 1 == 0
        ), "Discrete values must be consecutive integers"
        feat_type = "binary" if np.array_equal(x, np.array([0, 1])) else "integer"
        bounds = (int(feature.lower_bound), int(feature.upper_bound))

    elif isinstance(feature, CategoricalInput):
        feat_type = "categorical"
        bounds = feature.categories

    problem_config.add_feature(feat_type, bounds, name)


def _bofire_output_to_entmoot(problem_config: ProblemConfig, feature: AnyOutput):
    """Given a Bofire output feature, create an entmoot constraint"""
    if isinstance(feature.objective, MinimizeObjective):
        problem_config.add_min_objective(name=feature.key)

    if isinstance(feature.objective, MaximizeObjective):
        raise NotImplementedError("Only minimization problems are supported in Entmoot")


def _bofire_constraint_to_entmoot(
    problem_config: ProblemConfig,
    constraint: AnyConstraint,
    model_core: pyo.ConcreteModel,
):
    """Apply bofire constraints to entmoot model.

    TODO: Make this optimiser-agnostic"""
    if not isinstance(constraint, LinearConstraint):
        raise NotImplementedError("Non-linear constraints are not supported")

    # feat_idxs = [i for i, feat in enumerate(problem_config.feat_list) if feat.name in constraint.features]
    # retain order of constraints.features
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
    print(constr_expr)
    model_core.constr.add(expr=constr_expr)
