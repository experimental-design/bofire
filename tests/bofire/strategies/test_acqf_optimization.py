import unittest
from typing import cast

import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.testing import MockAcquisitionFunction

from bofire.benchmarks.api import Hartmann
from bofire.data_models.constraints.api import (
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
from bofire.data_models.strategies.predictives.acqf_optimization import (
    BotorchOptimizer as BotorchOptimizerModel,
)
from bofire.data_models.strategies.predictives.sobo import (
    SoboStrategy as SoboStrategyModel,
)
from bofire.strategies.api import SoboStrategy
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

    # Semi-continuous features are handled by post-AF pruning, so they must not
    # cause optimize_acqf_list to receive a fixed_features_list.
    semi_feature = domain.inputs.get_by_key("x_2")
    assert isinstance(semi_feature, ContinuousInput)
    semi_feature.bounds = (1.0, 2.0)
    semi_feature.allow_zero = True
    optimizer_args = optimizer._get_arguments_for_optimizer(
        optimizer=OptimizerEnum.OPTIMIZE_ACQF_LIST,
        domain=domain,
        candidate_count=2,
        acqfs=[simple_acqf, simple_acqf],
        bounds=get_bounds(domain),
    )
    assert isinstance(optimizer_args, _OptimizeAcqfListInput)
    assert optimizer_args.fixed_features == {1: 0.5}
    assert optimizer_args.fixed_features_list is None

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


def _semicontinuous_categorical_domain() -> Domain:
    """Domain of issue #795 plus a categorical, which forces the mixed
    optimizer while pruning handles the semi-continuous features.
    """
    return Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{i}", bounds=(1.0, 20.0), allow_zero=True)
            for i in (1, 2, 3)
        ]
        + [CategoricalInput(key="c", categories=["a", "b"])],
        outputs=[ContinuousOutput(key="y")],
    )


def test_get_categorical_combinations_excludes_semicontinuous_when_pruning():
    """Regression test for #795.

    When pruning is applicable, the semi-continuous features are resolved by
    the post-AF pruning step, so their on/off states must not additionally be
    enumerated into the `fixed_features_list`. Enumerating them yields
    combinations that pin *every* tensor column (all features off) next to
    combinations that do not; `optimize_acqf_mixed` stacks the per-combination
    acqf values, and botorch's all-features-fixed shortcut collapses the
    restart dimension, so the stack raises
    `RuntimeError: stack expects each tensor to be equal size`.
    """
    domain = _semicontinuous_categorical_domain()
    optimizer = BotorchOptimizer(BotorchOptimizerModel())

    combinations = optimizer.get_categorical_combinations(domain)

    # only the two categorical levels, not 2 (categories) * 2**3 (on/off)
    assert combinations == [{3: 0}, {3: 1}]

    # and no combination pins every tensor column
    n_columns = sum(len(idx) for idx in optimizer._features2idx(domain).values())
    assert all(len(combo) < n_columns for combo in combinations)

    # consistent with the count the optimizer routing is based on
    assert len(combinations) == domain.inputs.get_number_of_categorical_combinations(
        include_semicontinuous=False,
    )


def test_ask_with_semicontinuous_and_categorical():
    """End-to-end regression test for #795: `ask` must not blow up in
    `optimize_acqf_mixed` for a domain mixing `allow_zero` inputs with a
    categorical input.
    """
    domain = _semicontinuous_categorical_domain()
    experiments = pd.DataFrame(
        {
            "x1": [5.0, 10.0, 15.0, 18.0],
            "x2": [5.0, 10.0, 15.0, 8.0],
            "x3": [5.0, 10.0, 15.0, 12.0],
            "c": ["a", "b", "a", "b"],
            "y": [1.0, 2.0, 3.0, 4.0],
            "valid_y": [1, 1, 1, 1],
        },
    )

    data_model = SoboStrategyModel(domain=domain, seed=42)
    strategy = SoboStrategy(data_model=data_model)
    assert (
        strategy.acqf_optimizer._determine_optimizer(domain, n_acqfs=1)
        == OptimizerEnum.OPTIMIZE_ACQF_MIXED
    )

    strategy.tell(experiments)
    candidates = strategy.ask(candidate_count=2)

    assert len(candidates) == 2
    for key in ["x1", "x2", "x3"]:
        feat = cast(ContinuousInput, domain.inputs.get_by_key(key))
        values = candidates[key]
        # pruning resolves every semi-continuous feature to either zero or
        # a value inside its bounds -- never into the forbidden gap
        assert (
            (values == 0.0) | ((values >= feat.bounds[0]) & (values <= feat.bounds[1]))
        ).all()
