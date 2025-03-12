import importlib
import warnings

import numpy as np
import pytest


try:
    import gurobipy
    from entmoot.problem_config import FeatureType, ProblemConfig
except ImportError:
    warnings.warn(
        "entmoot not installed, BoFire's `EntingStrategy` cannot be used.",
        ImportWarning,
    )


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
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.strategies.api import EntingStrategy
from bofire.strategies.predictives.enting import domain_to_problem_config
from tests.bofire.strategies.test_base import domains


ENTMOOT_AVAILABLE = importlib.util.find_spec("entmoot") is not None
if ENTMOOT_AVAILABLE:
    try:
        # this is the recommended way to check presence  of gurobi license file
        gurobipy.Model()
        GUROBI_AVAILABLE = True
    except gurobipy.GurobiError:
        GUROBI_AVAILABLE = False

else:
    GUROBI_AVAILABLE = False


@pytest.fixture
def common_args():
    return {
        "dist_metric": "l1",
        "acq_sense": "exploration",
        "solver_name": "gurobi",
        "solver_verbose": False,
        "seed": 42,
    }


@pytest.mark.skipif(not ENTMOOT_AVAILABLE, reason="requires entmoot")
def test_enting_not_fitted(common_args):
    data_model = data_models.EntingStrategy(domain=domains[0], **common_args)
    strategy = EntingStrategy(data_model=data_model)

    msg = "Uncertainty model needs fit function call before it can predict."
    with pytest.raises(AssertionError, match=msg):
        strategy._ask(1)


@pytest.mark.skipif(not ENTMOOT_AVAILABLE, reason="requires entmoot")
@pytest.mark.parametrize(
    "params",
    [
        {"acq_sense": "penalty", "beta": 0.1, "dist_metric": "l2", "max_depth": 3},
    ],
)
def test_enting_param_consistency(common_args, params):
    # compare EntingParams objects between entmoot and bofire
    data_model = data_models.EntingStrategy(
        domain=domains[0],
        **{**common_args, **params},
    )
    strategy = EntingStrategy(data_model=data_model)
    strategy._init_problem_config()

    # check that the parameters propagate to the model correctly
    assert strategy._enting is not None
    assert strategy._enting._acq_sense == data_model.acq_sense
    assert strategy._enting._beta == data_model.beta


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="requires entmoot+gurobi")
@pytest.mark.parametrize(
    "allowed_k",
    [1, 3, 5, 6],
)
@pytest.mark.slow
def test_nchoosek_constraint_with_enting(common_args, allowed_k):
    benchmark = Hartmann(6, allowed_k=allowed_k)
    samples = benchmark.domain.inputs.sample(10, seed=43)
    experiments = benchmark.f(samples, return_complete=True)

    data_model = data_models.EntingStrategy(domain=benchmark.domain, **common_args)
    strategy = EntingStrategy(data_model)

    strategy.tell(experiments)
    proposal = strategy.ask(1)

    input_values = proposal[benchmark.domain.inputs.get_keys()]
    assert (input_values != 0).sum().sum() <= allowed_k


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="requires entmoot+gurobi")
@pytest.mark.slow
def test_propose_optimal_point(common_args):
    # regression test, ensure that a good point is proposed
    benchmark = Hartmann(6)
    samples = benchmark.domain.inputs.sample(50, seed=43)
    experiments = benchmark.f(samples, return_complete=True)

    data_model = data_models.EntingStrategy(domain=benchmark.domain, **common_args)
    strategy = EntingStrategy(data_model)

    # filter experiments to remove those in a box surrounding optimum
    radius = 0.5
    X = experiments[benchmark.domain.inputs.get_keys()].to_numpy()
    X_opt = benchmark.get_optima()[benchmark.domain.inputs.get_keys()].to_numpy()
    sq_dist_to_optimum = ((X - X_opt) ** 2).sum(axis=1)
    include = sq_dist_to_optimum > radius

    strategy.tell(experiments[include])
    proposal = strategy.ask(1)

    assert np.allclose(
        proposal.loc[0, benchmark.domain.inputs.get_keys()].tolist(),
        [0.0, 0.79439, 0.6124835, 0.0, 1.0, 0.0],
        atol=1e-6,
    )


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="requires entmoot+gurobi")
@pytest.mark.slow
def test_propose_unique_points(common_args):
    # ensure that the strategy does not repeat candidates
    benchmark = Hartmann(6)
    samples = benchmark.domain.inputs.sample(10)
    experiments = benchmark.f(samples, return_complete=True)

    data_model = data_models.EntingStrategy(domain=benchmark.domain, **common_args)
    strategy = EntingStrategy(data_model)

    strategy.tell(experiments)

    a = strategy.ask(candidate_count=5)
    b = strategy.ask(candidate_count=5, add_pending=True)
    c = strategy.ask(candidate_count=5)

    # without adding points to pending, a and b should propose the same points
    assert a.equals(b)
    # after adding points to pending, b and c should propose different points
    assert not b.equals(c)


# Test utils for converting from bofire problem definition to entmoot
def feat_equal(a: "FeatureType", b: "FeatureType") -> bool:
    """Check if entmoot.FeatureTypes are equal.

    Args:
        a: First feature.
        b: Second feature.

    """
    # no __eq__ method is implemented for FeatureType, hence the need for this function
    assert a is not None and b is not None
    return all(
        (
            a.name == b.name,
            a.get_enc_bnds() == b.get_enc_bnds(),
            a.is_real() == b.is_real(),
            a.is_cat() == b.is_cat(),
            a.is_int() == b.is_int(),
            a.is_bin() == b.is_bin(),
        ),
    )


if1 = CategoricalInput(key="if1", categories=["blue", "orange", "gray"])
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
    features=["if4", "if5"],
    coefficients=[1, 1],
    rhs=12,
)
constr2 = LinearEqualityConstraint(features=["if4", "if5"], coefficients=[1, 5], rhs=38)


def build_problem_config(inputs, outputs) -> "ProblemConfig":
    problem_config = ProblemConfig()
    for feature in inputs:
        problem_config.add_feature(**feature)

    for objective in outputs:
        problem_config.add_min_objective(**objective)

    return problem_config


@pytest.mark.skipif(not ENTMOOT_AVAILABLE, reason="requires entmoot")
def test_domain_to_problem_config():
    domain = Domain.from_lists(inputs=[if1, if2, if3, if4], outputs=[of1, of2])
    ent_problem_config = build_problem_config(
        inputs=[if1_ent, if2_ent, if3_ent, if4_ent],
        outputs=[of1_ent, of2_ent],
    )
    bof_problem_config, _ = domain_to_problem_config(domain)
    for feat_ent in ent_problem_config.feat_list:
        # get bofire feature with same name
        feat_bof = next(
            (f for f in bof_problem_config.feat_list if f.name == feat_ent.name),
            None,
        )
        assert feat_equal(feat_ent, feat_bof)

    assert len(ent_problem_config.obj_list) == len(bof_problem_config.obj_list)


@pytest.mark.skipif(not ENTMOOT_AVAILABLE, reason="requires entmoot")
def test_convert_constraint_to_entmoot():
    constraints = [constr1, constr2]
    domain = Domain.from_lists(
        inputs=[if1, if2, if3, if4, if5],
        outputs=[of1, of2],
        constraints=constraints,
    )
    _, model = domain_to_problem_config(domain)

    assert len(constraints) == len(model.problem_constraints)
