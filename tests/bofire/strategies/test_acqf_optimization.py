import unittest
from typing import cast

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.testing import MockAcquisitionFunction

import bofire.strategies.predictives.acqf_optimization as acqf_mod
from bofire.benchmarks.api import Hartmann
from bofire.data_models.acquisition_functions.api import qLogEI
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.data_models.strategies.predictives.acqf_optimization import (
    BotorchOptimizer as BotorchOptimizerModel,
)
from bofire.strategies.api import RandomStrategy, SoboStrategy
from bofire.strategies.predictives.acqf_optimization import (
    BotorchOptimizer,
    OptimizerEnum,
    _OptimizeAcqfInput,
    _OptimizeAcqfListInput,
    _OptimizeAcqfMixedAlternatingInput,
    _OptimizeAcqfMixedInput,
)
from bofire.utils.torch_tools import tkwargs


def test_determine_optimizer():
    optimizer_data = BotorchOptimizerModel()
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y1")],
    )
    optimizer = BotorchOptimizer(optimizer_data)
    assert (
        optimizer._determine_optimizer(domain, n_acqfs=2)
        == OptimizerEnum.OPTIMIZE_ACQF_LIST
    )
    assert (
        optimizer._determine_optimizer(domain, n_acqfs=1) == OptimizerEnum.OPTIMIZE_ACQF
    )
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["a", "b"]),
        ],
        outputs=[ContinuousOutput(key="y1")],
    )
    assert (
        optimizer._determine_optimizer(domain, n_acqfs=1)
        == OptimizerEnum.OPTIMIZE_ACQF_MIXED
    )
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=[f"cat_{i}" for i in range(12)]),
        ],
        outputs=[ContinuousOutput(key="y1")],
    )
    assert (
        optimizer._determine_optimizer(domain, n_acqfs=1)
        == OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING
    )
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            CategoricalInput(key="x3", categories=[f"cat_{i}" for i in range(12)]),
        ],
        constraints=[
            NChooseKConstraint(
                features=["x1", "x2"], min_count=0, max_count=1, none_also_valid=False
            )
        ],
        outputs=[ContinuousOutput(key="y1")],
    )
    # NChooseK on continuous features only -> pruning applicable -> NChooseK
    # is excluded from AF-time nonlinear constraints (handled by post-AF
    # pruning instead). With 12 categorical combinations > ALTERNATING
    # threshold and no remaining nonlinear constraints, routing falls
    # through to ALTERNATING.
    assert (
        optimizer._determine_optimizer(domain, n_acqfs=1)
        == OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING
    )


def test_get_arguments_for_optimizer():
    benchmark = Hartmann()

    optimizer_data = BotorchOptimizerModel()
    domain = benchmark.domain
    optimizer = BotorchOptimizer(optimizer_data)

    simple_acqf = cast(AcquisitionFunction, MockAcquisitionFunction())

    def get_bounds(domain: Domain) -> torch.Tensor:
        input_preprocessing_specs = optimizer._input_preprocessing_specs(domain)
        lower, upper = domain.inputs.get_bounds(
            specs=input_preprocessing_specs,
        )
        return torch.tensor([lower, upper]).to(**tkwargs)

    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF,
        domain=domain,
        candidate_count=1,
        acqfs=[simple_acqf],
        bounds=get_bounds(domain),
    )
    assert isinstance(optimizer_args, _OptimizeAcqfInput)
    assert optimizer_args.num_restarts == optimizer_data.n_restarts
    assert optimizer_args.raw_samples == optimizer_data.n_raw_samples
    assert optimizer_args.equality_constraints == []
    assert optimizer_args.inequality_constraints == []
    assert optimizer_args.ic_generator is None
    assert optimizer_args.nonlinear_inequality_constraints is None
    assert optimizer_args.sequential is False
    assert optimizer_args.fixed_features == {}
    assert optimizer_args.options == {"batch_limit": 20, "maxiter": 2000}

    # test with nchooseks: NChooseK on continuous features only is
    # handled by post-AF pruning, so the AF-time arguments must NOT
    # carry a smooth-NChooseK callable. ic_generator and the
    # constraint-aware batch_limit override also drop out.
    benchmark = Hartmann(dim=6, allowed_k=3)
    optimizer_data = BotorchOptimizerModel()
    domain = benchmark.domain
    optimizer = BotorchOptimizer(optimizer_data)

    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF,
        domain=domain,
        candidate_count=1,
        acqfs=[simple_acqf],
        bounds=get_bounds(domain),
    )

    assert optimizer_args.nonlinear_inequality_constraints is None
    assert optimizer_args.ic_generator is None
    assert optimizer_args.generator is None
    assert optimizer_args.options["batch_limit"] == optimizer_data.batch_limit

    domain.constraints.constraints.append(
        LinearInequalityConstraint(features=["x_1", "x_2"], coefficients=[1, 1], rhs=2)
    )
    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF,
        domain=domain,
        candidate_count=1,
        acqfs=[simple_acqf],
        bounds=get_bounds(domain),
    )
    assert len(optimizer_args.inequality_constraints) == 1

    # test for acqf mixed
    domain = Hartmann().domain
    domain.inputs.features.append(
        CategoricalInput(key="x_cat", categories=[f"cat_{i}" for i in range(4)])
    )
    domain.inputs.get_by_key("x_1").bounds = (0.5, 0.5)

    optimizer_data = BotorchOptimizerModel()
    optimizer = BotorchOptimizer(optimizer_data)
    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF_MIXED,
        domain=domain,
        candidate_count=1,
        bounds=get_bounds(domain),
        acqfs=[simple_acqf],
    )
    assert isinstance(optimizer_args, _OptimizeAcqfMixedInput)
    assert optimizer_args.fixed_features_list == [
        {1: 0.5, 6: 0},
        {1: 0.5, 6: 1},
        {1: 0.5, 6: 2},
        {1: 0.5, 6: 3},
    ]
    domain.inputs.get_by_key("x_1").bounds = (0, 1)
    domain.inputs.features.append(
        CategoricalInput(
            key="x_cat2",
            categories=[f"cat2_{i}" for i in range(3)],
            allowed=[True, False, True],
        )
    )

    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING,
        domain=domain,
        candidate_count=1,
        acqfs=[simple_acqf],
        bounds=get_bounds(domain),
    )
    assert isinstance(optimizer_args, _OptimizeAcqfMixedAlternatingInput)
    assert optimizer_args.discrete_dims == {}
    assert optimizer_args.cat_dims == {6: [0, 1, 2, 3], 7: [0, 2]}
    domain.inputs.features.append(DiscreteInput(key="x_discrete", values=[0, 1, 2, 7]))
    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING,
        domain=domain,
        candidate_count=1,
        acqfs=[simple_acqf],
        bounds=get_bounds(domain),
    )
    assert isinstance(optimizer_args, _OptimizeAcqfMixedAlternatingInput)
    assert optimizer_args.discrete_dims == {6: [0.0, 1.0, 2.0, 7.0]}
    assert optimizer_args.cat_dims == {7: [0, 1, 2, 3], 8: [0, 2]}
    domain.inputs.features.append(
        CategoricalInput(key="x_cat3", categories=["a", "b"], allowed=[True, False])
    )
    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING,
        domain=domain,
        candidate_count=1,
        acqfs=[simple_acqf],
        bounds=get_bounds(domain),
    )
    assert optimizer_args.discrete_dims == {6: [0, 1, 2, 7]}
    assert optimizer_args.cat_dims == {7: [0, 1, 2, 3], 8: [0, 2]}
    assert optimizer_args.fixed_features == {9: 0}
    # test for acqf list
    domain = Hartmann().domain
    domain.inputs.get_by_key("x_1").bounds = (0.5, 0.5)
    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING,
        domain=domain,
        candidate_count=2,
        acqfs=[simple_acqf, simple_acqf],
        bounds=get_bounds(domain),
    )
    assert optimizer_args.fixed_features == {1: 0.5}
    domain.inputs.features.append(
        CategoricalInput(key="x_cat", categories=[f"cat_{i}" for i in range(2)])
    )
    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF_LIST,
        domain=domain,
        candidate_count=2,
        acqfs=[simple_acqf, simple_acqf],
        bounds=get_bounds(domain),
    )
    assert isinstance(optimizer_args, _OptimizeAcqfListInput)
    assert optimizer_args.fixed_features_list == [{1: 0.5, 6: 0}, {1: 0.5, 6: 1}]
    assert optimizer_args.fixed_features is None


def test_get_fixed_features():
    domain = Hartmann().domain

    optimizer_data = BotorchOptimizerModel()
    optimizer = BotorchOptimizer(optimizer_data)

    assert optimizer.get_fixed_features(domain=domain) == {}
    domain.inputs.get_by_key("x_1").bounds = (0.5, 0.5)
    assert optimizer.get_fixed_features(domain=domain) == {1: 0.5}
    domain.inputs.features.append(
        CategoricalInput(key="x_cat", categories=["a", "b"], allowed=[False, True])
    )
    assert optimizer.get_fixed_features(domain=domain) == {1: 0.5, 6: 1}


def test_base_get_categorical_combinations():
    domain = Hartmann().domain

    optimizer_data = BotorchOptimizerModel()
    optimizer = BotorchOptimizer(optimizer_data)

    assert optimizer.get_categorical_combinations(domain) == [{}]
    domain.inputs.get_by_key("x_1").bounds = (0.5, 0.5)
    assert optimizer.get_categorical_combinations(domain) == [{1: 0.5}]
    domain.inputs.features.append(
        CategoricalInput(
            key="x_cat", categories=["a", "b", "c"], allowed=[False, True, True]
        )
    )
    assert optimizer.get_categorical_combinations(domain) == [
        {1: 0.5, 6: 1},
        {1: 0.5, 6: 2},
    ]
    domain.inputs.features.append(
        DiscreteInput(
            key="x_discrete",
            values=[0, 1],
        )
    )
    c = unittest.TestCase()
    c.assertCountEqual(
        optimizer.get_categorical_combinations(domain),
        [
            {1: 0.5, 6: 0, 7: 1},
            {1: 0.5, 6: 0, 7: 2},
            {1: 0.5, 6: 1, 7: 1},
            {1: 0.5, 6: 1, 7: 2},
        ],
    )


# -- BoTorch → pounce dispatch coverage (`use_ipopt=True` route) -------------


def _mixture_domain() -> Domain:
    """3-component continuous mixture in [0, 1] with x1+x2+x3 = 1."""
    return Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0.0, 1.0)),
            ContinuousInput(key="x2", bounds=(0.0, 1.0)),
            ContinuousInput(key="x3", bounds=(0.0, 1.0)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3"],
                coefficients=[1.0, 1.0, 1.0],
                rhs=1.0,
            )
        ],
    )


def _seed_sobo_with_random(domain: Domain, optimizer_dm: BotorchOptimizerModel):
    """Build a fitted SoboStrategy seeded with a handful of random experiments."""
    strategy = SoboStrategy(
        data_model=SoboStrategyDataModel(
            domain=domain,
            acquisition_function=qLogEI(),
            acquisition_optimizer=optimizer_dm,
        )
    )
    rs = RandomStrategy(data_model=RandomStrategyDataModel(domain=domain))
    seed = rs.ask(5)
    seed["y"] = np.random.default_rng(0).normal(size=len(seed))
    strategy.tell(seed)
    return strategy


def test_optimizer_options_merged_into_botorch_options():
    """`optimizer_options` from the data model must reach the options dict."""
    domain = _mixture_domain()
    optimizer_dm = BotorchOptimizerModel(
        use_ipopt=True,
        optimizer_options={"tol": 1e-12, "max_iter": 500, "print_level": 0},
    )
    optimizer = BotorchOptimizer(optimizer_dm)
    options = optimizer._get_optimizer_options(domain)
    assert options["tol"] == 1e-12
    assert options["max_iter"] == 500
    assert options["print_level"] == 0
    # The pounce method assignment is still present.
    assert options["method"] is acqf_mod.minimize_pounce
    # User overrides should beat BoFire defaults — verify with a known key.
    optimizer_dm2 = BotorchOptimizerModel(
        use_ipopt=True, optimizer_options={"maxiter": 1}
    )
    options2 = BotorchOptimizer(optimizer_dm2)._get_optimizer_options(domain)
    assert options2["maxiter"] == 1


def test_use_ipopt_actually_calls_pounce(monkeypatch):
    """End-to-end: ask(1) with use_ipopt=True must invoke pounce.minimize."""
    domain = _mixture_domain()

    call_count = {"n": 0}
    real_pounce = acqf_mod.minimize_pounce

    def tracking_pounce(*args, **kwargs):
        call_count["n"] += 1
        return real_pounce(*args, **kwargs)

    monkeypatch.setattr(acqf_mod, "minimize_pounce", tracking_pounce)

    optimizer_dm = BotorchOptimizerModel(
        use_ipopt=True, n_restarts=2, n_raw_samples=16, maxiter=50
    )
    strategy = _seed_sobo_with_random(domain, optimizer_dm)
    candidates = strategy.ask(1)

    # The route fired at least once during the acqf optimization.
    assert call_count["n"] > 0, (
        "pounce.minimize was never invoked despite use_ipopt=True; "
        "the route is broken"
    )
    # And the candidate is mixture-feasible to ipopt-level tolerance.
    mixture_sum = float(candidates[["x1", "x2", "x3"]].iloc[0].sum())
    assert abs(mixture_sum - 1.0) < 1e-6


def test_use_ipopt_false_does_not_call_pounce(monkeypatch):
    """Negative control: with use_ipopt=False the route is gated off."""
    domain = _mixture_domain()

    call_count = {"n": 0}
    real_pounce = acqf_mod.minimize_pounce

    def tracking_pounce(*args, **kwargs):
        call_count["n"] += 1
        return real_pounce(*args, **kwargs)

    monkeypatch.setattr(acqf_mod, "minimize_pounce", tracking_pounce)

    optimizer_dm = BotorchOptimizerModel(
        use_ipopt=False, n_restarts=2, n_raw_samples=16, maxiter=50
    )
    strategy = _seed_sobo_with_random(domain, optimizer_dm)
    strategy.ask(1)

    assert call_count["n"] == 0, (
        "pounce.minimize was invoked despite use_ipopt=False; " "the gate has regressed"
    )
