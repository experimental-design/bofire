import random

import numpy as np
import pytest
from entmoot.problem_config import FeatureType, ProblemConfig
from pydantic.utils import deep_update

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.api import Hartmann
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Input,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.strategies.api import EntingStrategy
from bofire.strategies.predictives.enting import domain_to_problem_config
from tests.bofire.strategies.test_base import domains

VALID_ENTING_STRATEGY_SPEC = {
    "domain": domains[1],
    "enting_params": {"unc_params": {"dist_metric": "l1", "acq_sense": "penalty"}},
}

ENTING_STRATEGY_SPECS = {
    "valids": [
        VALID_ENTING_STRATEGY_SPEC,
        {
            **VALID_ENTING_STRATEGY_SPEC,
            "enting_params": {"tree_train_params": {"train_lib": "lgbm"}},
        },
    ],
    "invalids": [
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {"enting_params": {"unc_params": {"acq_sense": None}}},
        ),
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {"enting_params": {"unc_params": {"distance_metric": None}}},
        ),
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {"enting_params": {"tree_train_params": {"train_lib": None}}},
        ),
        deep_update(
            VALID_ENTING_STRATEGY_SPEC,
            {
                "enting_params": {
                    "tree_train_params": {"train_params": {"max_depth": -3}}
                }
            },
        ),
    ],
}

SOLVER_PARAMS = {"solver_name": "gurobi", "verbose": False}


@pytest.mark.parametrize(
    "domain, enting_params, solver_params",
    [
        (
            domains[0],
            VALID_ENTING_STRATEGY_SPEC["enting_params"],
            SOLVER_PARAMS,
        )
    ],
)
def test_enting_not_fitted(domain, enting_params, solver_params):
    data_model = data_models.EntingStrategy(
        domain=domain, enting_params=enting_params, solver_params=solver_params
    )
    strategy = EntingStrategy(data_model=data_model)

    msg = "Uncertainty model needs fit function call before it can predict."
    with pytest.raises(AssertionError, match=msg):
        strategy._ask(1)


@pytest.mark.parametrize(
    "domain, enting_params, solver_params",
    [
        (
            domains[0],
            VALID_ENTING_STRATEGY_SPEC["enting_params"],
            SOLVER_PARAMS,
        )
    ],
)
def test_enting_param_consistency(domain, enting_params, solver_params):
    # compare EntingParams objects between entmoot and bofire
    data_model = data_models.EntingStrategy(
        domain=domain, enting_params=enting_params, solver_params=solver_params
    )
    strategy = EntingStrategy(data_model=data_model)
    assert strategy._enting._acq_sense == data_model.enting_params.unc_params.acq_sense
    assert strategy._enting._beta == data_model.enting_params.unc_params.beta


@pytest.mark.parametrize(
    "allowed_k",
    [1, 3, 5, 6],
)
@pytest.mark.slow
def test_nchoosek_constraint_with_enting(allowed_k):
    benchmark = Hartmann(6, allowed_k=allowed_k)
    samples = benchmark.domain.inputs.sample(10)
    experiments = benchmark.f(samples, return_complete=True)

    enting_params = VALID_ENTING_STRATEGY_SPEC["enting_params"]
    solver_params = SOLVER_PARAMS
    data_model = data_models.EntingStrategy(
        domain=benchmark.domain,
        enting_params=enting_params,
        solver_params=solver_params,
    )
    strategy = EntingStrategy(data_model)

    strategy.tell(experiments)
    proposal = strategy.ask(1)

    input_values = proposal[benchmark.domain.get_feature_keys(Input)]
    assert (input_values != 0).sum().sum() == allowed_k


@pytest.mark.slow
def test_propose_optimal_point():
    # regression test, ensure that a good point is proposed
    np.random.seed(42)
    random.seed(42)

    benchmark = Hartmann(6)
    samples = benchmark.domain.inputs.sample(50)
    experiments = benchmark.f(samples, return_complete=True)

    enting_params = VALID_ENTING_STRATEGY_SPEC["enting_params"]
    solver_params = SOLVER_PARAMS
    data_model = data_models.EntingStrategy(
        domain=benchmark.domain,
        enting_params=enting_params,
        solver_params=solver_params,
    )
    strategy = EntingStrategy(data_model)

    # filter experiments to remove those in a box surrounding optimum
    radius = 0.5
    X = experiments[benchmark.domain.get_feature_keys(Input)].values
    X_opt = benchmark.get_optima()[benchmark.domain.get_feature_keys(Input)].values
    l1_dist_to_optimum = ((X - X_opt) ** 2).sum(axis=1)
    include = l1_dist_to_optimum > radius

    strategy.tell(experiments[include])
    proposal = strategy.ask(1)

    assert np.allclose(
        proposal.loc[0, benchmark.domain.get_feature_keys(Input)].tolist(),
        [0.275, 0.95454, 0.28484, 0.34005, 0.015457, 0.06135],
        atol=1e-6,
    )


# test enting utils
def feat_equal(a: FeatureType, b: FeatureType) -> bool:
    """Check if entmoot.FeatureTypes are equal.

    Args:
        a: First feature.
        b: Second feature.
    """
    # no __eq__ method is implemented for FeatureType, hence the need for this function
    print(
        (
            a.name,
            b.name,
            a.get_enc_bnds(),
            b.get_enc_bnds(),
            a.is_real(),
            b.is_real(),
            a.is_cat() == b.is_cat(),
            a.is_int() == b.is_int(),
            a.is_bin() == b.is_bin(),
        )
    )
    return all(
        (
            a.name == b.name,
            a.get_enc_bnds() == b.get_enc_bnds(),
            a.is_real() == b.is_real(),
            a.is_cat() == b.is_cat(),
            a.is_int() == b.is_int(),
            a.is_bin() == b.is_bin(),
        )
    )


if1 = CategoricalInput(key="if1", categories=("blue", "orange", "gray"))
if1_ent = {
    "feat_type": "categorical",
    "bounds": ("blue", "orange", "gray"),
    "name": "if1",
}

if2 = DiscreteInput(key="if2", values=[5, 6, 7])
if2_ent = {"feat_type": "integer", "bounds": (5, 7), "name": "if2"}

if3 = DiscreteInput(key="if3", values=[0, 1])
if3_ent = {"feat_type": "binary", "name": "if3"}

if4 = ContinuousInput(key="if4", bounds=[5.0, 6.0])
if4_ent = {"feat_type": "real", "bounds": (5.0, 6.0), "name": "if4"}

if5 = ContinuousInput(key="if5", bounds=[0.0, 10.0])

of1 = ContinuousOutput(key="of1", objective=MinimizeObjective(w=1.0))
of1_ent = {"name": "of1"}

of2 = ContinuousOutput(key="of2", objective=MaximizeObjective(w=1.0))
of2_ent = {"name": "of2"}

constr1 = LinearInequalityConstraint(
    features=["if4", "if5"], coefficients=[1, 1], rhs=12
)
constr2 = LinearEqualityConstraint(features=["if4", "if5"], coefficients=[1, 5], rhs=38)


def build_problem_config(inputs, outputs) -> ProblemConfig:
    problem_config = ProblemConfig()
    for feature in inputs:
        problem_config.add_feature(**feature)

    for objective in outputs:
        problem_config.add_min_objective(**objective)

    return problem_config


def test_domain_to_problem_config():
    domain = Domain.from_lists(inputs=[if1, if2, if3, if4], outputs=[of1, of2])
    ent_problem_config = build_problem_config(
        inputs=[if1_ent, if2_ent, if3_ent, if4_ent], outputs=[of1_ent, of2_ent]
    )
    bof_problem_config, _ = domain_to_problem_config(domain)
    for feat_a, feat_b in zip(
        ent_problem_config.feat_list, bof_problem_config.feat_list
    ):
        assert feat_equal(feat_a, feat_b)

    assert len(ent_problem_config.obj_list) == len(bof_problem_config.obj_list)


def test_convert_constraint_to_entmoot():
    constraints = [constr1, constr2]
    domain = Domain.from_lists(
        inputs=[if1, if2, if3, if4, if5], outputs=[of1, of2], constraints=constraints
    )
    _, model = domain_to_problem_config(domain)

    assert len(constraints) == len(model.problem_constraints)
