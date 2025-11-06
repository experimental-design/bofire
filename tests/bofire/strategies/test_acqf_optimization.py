import unittest
from typing import cast

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
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
    assert (
        optimizer._determine_optimizer(domain, n_acqfs=1)
        == OptimizerEnum.OPTIMIZE_ACQF_MIXED
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

    # test with nchooseks
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

    assert len(optimizer_args.nonlinear_inequality_constraints) == 1
    assert optimizer_args.ic_generator == gen_batch_initial_conditions
    assert optimizer_args.generator is not None
    assert optimizer_args.options["batch_limit"] == 1

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
