import random
import warnings
from typing import Type

import pandas as pd
import pytest
import torch
from botorch.models import MixedSingleTaskGP, ModelListGP, SingleTaskGP

from bofire.domain import Domain
from bofire.domain.constraints import (
    Constraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    Feature,
    OutputFeature,
)
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective
from bofire.domain.util import KeyModel
from bofire.strategies.botorch.base import BotorchBasicBoStrategy, ModelSpec
from bofire.strategies.botorch.sobo import AcquisitionFunctionEnum
from bofire.strategies.botorch.utils.objectives import MultiplicativeObjective
from bofire.utils.torch_tools import tkwargs
from tests.bofire.domain.test_domain_validators import generate_experiments
from tests.bofire.domain.test_features import (
    VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class DummyFeature(KeyModel):
    def is_fixed(self):
        pass

    def fixed_value(self):
        pass


class DummyStrategy(BotorchBasicBoStrategy):
    def _init_acqf(
        self,
    ) -> None:
        pass

    def _init_objective(
        self,
    ) -> None:
        self.objective = MultiplicativeObjective(
            targets=[
                var.objective
                for var in self.domain.output_features.get_by_objective(excludes=None)
            ]
        )
        return

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearConstraint,
            LinearEqualityConstraint,
            LinearInequalityConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousInput,
            CategoricalInput,
            CategoricalDescriptorInput,
            ContinuousOutput,
        ]

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [MinimizeObjective, MaximizeObjective]


if1 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    }
)
if2 = ContinuousInput(
    **{
        **VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if2",
    }
)

if3 = CategoricalInput(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if3",
    }
)

if4 = CategoricalInput(
    **{
        **VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if4",
    }
)

if5 = CategoricalDescriptorInput(
    **{
        **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if5",
    }
)

if6 = CategoricalDescriptorInput(
    **{
        **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if6",
    }
)

if7 = DummyFeature(key="if7")

if8 = CategoricalDescriptorInput(
    **{
        **VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if8",
    }
)

of1 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    }
)

of2 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of2",
    }
)

domains = [
    Domain(
        input_features=[if1, if3, if5],  # no fixed features
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        input_features=[
            if1,
            if2,
            if3,
            if4,
            if5,
            if6,
        ],  # all feature types incl. with fixed values
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        input_features=[
            if1,
            if2,
            if3,
            if4,
            if5,
            if6,
        ],  # all feature types incl. with fixed values + mutli-objective
        output_features=[of1, of2],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if2],  # only continuous features
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if3, if5],  # all feature types + mutli-objective
        output_features=[of1, of2],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if2, if8],
        output_features=[of1, of2],
        constraints=[],
    )
    # Domain(
    #     input_features=[if1, if7], # unknown dummy feature
    #     output_features=[of1],
    #     constraints=[],
    # )
]

data = [
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if3": ["c1", "c2", "c3", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "of1": [10, 11, 12, 13],
            "valid_of1": [1, 0, 1, 0],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if2": [3, 3, 3, 3],
            "if3": ["c1", "c2", "c3", "c1"],
            "if4": ["c1", "c1", "c1", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "if6": ["c1", "c1", "c1", "c1"],
            "of1": [10, 11, 12, 13],
            "valid_of1": [1, 0, 1, 0],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if2": [3, 3, 3, 3],
            "if3": ["c1", "c2", "c3", "c1"],
            "if4": ["c1", "c1", "c1", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "if6": ["c1", "c1", "c1", "c1"],
            "of1": [10, 11, 12, 13],
            "of2": [100, 103, 105, 110],
            "valid_of1": [1, 0, 1, 0],
            "valid_of2": [0, 1, 1, 0],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if2": [3, 3, 3, 3],
            "of1": [10, 11, 12, 13],
            "valid_of1": [1, 0, 1, 0],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if3": ["c1", "c2", "c3", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "of1": [10, 11, 12, 13],
            "of2": [100, 103, 105, 110],
            "valid_of1": [1, 0, 1, 0],
            "valid_of2": [0, 1, 1, 0],
        }
    ),
]


# model specs applicable for domains[2]
model_specs = [
    [
        ModelSpec(
            output_feature="of1",
            input_features=["if1", "if3", "if4", "if5", "if6"],
            kernel="RBF",
            scaler="NORMALIZE",
            ard=True,
        ),
        ModelSpec(
            output_feature="of2",
            input_features=["if2", "if3", "if4", "if5", "if6"],
            kernel="MATERN_25",
            scaler="STANDARDIZE",
            ard=False,
        ),
    ],
    [
        ModelSpec(
            output_feature="of2",
            input_features=["if1", "if3", "if4", "if5", "if6"],
            kernel="RBF",
            scaler="NORMALIZE",
            ard=True,
        ),
        ModelSpec(
            output_feature="of1",
            input_features=["if2", "if3", "if4", "if5", "if6"],
            kernel="MATERN_15",
            scaler="STANDARDIZE",
            ard=False,
        ),
    ],
    [
        ModelSpec(
            output_feature="of1",
            input_features=["if1", "if2", "if4", "if5", "if6"],
            kernel="RBF",
            scaler="NORMALIZE",
            ard=True,
        ),
        ModelSpec(
            output_feature="of2",
            input_features=["if1", "if2", "if3", "if5", "if6"],
            kernel="MATERN_25",
            scaler="STANDARDIZE",
            ard=False,
        ),
    ],
    [
        ModelSpec(
            output_feature="of1",
            input_features=["if1", "if2", "if3", "if4", "if6"],
            kernel="RBF",
            scaler="NORMALIZE",
            ard=True,
        ),
        ModelSpec(
            output_feature="of2",
            input_features=["if1", "if2", "if3", "if4", "if5"],
            kernel="MATERN_25",
            scaler="STANDARDIZE",
            ard=False,
        ),
    ],
    [
        ModelSpec(
            output_feature="of1",
            input_features=["if1", "if3", "if6"],
            kernel="RBF",
            scaler="NORMALIZE",
            ard=True,
        ),
        ModelSpec(
            output_feature="of2",
            input_features=["if2", "if4", "if5"],
            kernel="MATERN_25",
            scaler="STANDARDIZE",
            ard=False,
        ),
    ],
]


@pytest.mark.parametrize("domain", [(domain) for domain in domains])
def test_base_create(domain: Domain):
    with pytest.raises(ValueError, match="number sobol samples"):
        DummyStrategy(domain=domain, num_sobol_samples=5)

    with pytest.raises(ValueError, match="number raw samples"):
        DummyStrategy(domain=domain, num_raw_samples=5)


@pytest.mark.parametrize(
    "domain, descriptor_encoding, categorical_encoding, expected_bounds",
    [
        (
            domains[0],
            "DESCRIPTOR",
            "ONE_HOT",
            torch.tensor([[3, 0, 0, 0, 1, 1], [5.3, 1, 1, 1, 5, 7]]).to(**tkwargs),
        ),
        (
            domains[0],
            "CATEGORICAL",
            "ONE_HOT",
            torch.tensor([[3, 0, 0, 0, 0, 0, 0], [5.3, 1, 1, 1, 1, 1, 1]]).to(
                **tkwargs
            ),
        ),
        (
            domains[0],
            "DESCRIPTOR",
            "ORDINAL",
            torch.tensor([[3, 0, 1, 1], [5.3, 2, 5, 7]]).to(**tkwargs),
        ),
        (
            domains[0],
            "CATEGORICAL",
            "ORDINAL",
            torch.tensor([[3, 0, 0], [5.3, 2, 2]]).to(**tkwargs),
        ),
        (
            domains[1],
            "DESCRIPTOR",
            "ONE_HOT",
            torch.tensor(
                [
                    [3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2],
                    [5.3, 3, 1, 1, 1, 1, 1, 1, 5, 7, 1, 2],
                ]
            ).to(**tkwargs),
        ),
        (
            domains[1],
            "CATEGORICAL",
            "ONE_HOT",
            torch.tensor(
                [
                    [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [5.3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ).to(**tkwargs),
        ),
        (
            domains[1],
            "DESCRIPTOR",
            "ORDINAL",
            torch.tensor([[3, 3, 0, 0, 1, 1, 1, 2], [5.3, 3, 2, 2, 5, 7, 1, 2]]).to(
                **tkwargs
            ),
        ),
        (
            domains[1],
            "CATEGORICAL",
            "ORDINAL",
            torch.tensor([[3, 3, 0, 0, 0, 0], [5.3, 3, 2, 2, 2, 2]]).to(**tkwargs),
        ),
    ],
)
def test_base_get_bounds(
    domain, descriptor_encoding, categorical_encoding, expected_bounds
):
    strategy = DummyStrategy(
        domain=domain,
        descriptor_encoding=descriptor_encoding,
        categorical_encoding=categorical_encoding,
    )

    bounds = strategy.get_bounds()

    assert torch.allclose(
        bounds, expected_bounds
    )  # torch.equal asserts false due to deviation of 1e-7??


def test_base_get_bounds_fit():
    # at first the fix on the continuous ones is tested
    domain = domains[3]
    strategy = DummyStrategy(
        domain=domain, descriptor_encoding="DESCRIPTOR", categorical_encoding="ONE_HOT"
    )

    strategy.domain.set_experiments(generate_experiments(domain, 100, tol=2.0))
    opt_bounds = strategy.get_bounds(optimize=True)
    fit_bounds = strategy.get_bounds(optimize=False)
    for i, key in enumerate(domain.get_feature_keys(ContinuousInput)):
        assert fit_bounds[0, i] < opt_bounds[0, i]
        assert fit_bounds[1, i] > opt_bounds[1, i]
        assert fit_bounds[0, i] == strategy.domain.experiments[key].min()
        assert fit_bounds[1, i] == strategy.domain.experiments[key].max()
    # next test the fix for the CategoricalDescriptor feature
    domain = domains[1]
    strategy = DummyStrategy(
        domain=domain, descriptor_encoding="DESCRIPTOR", categorical_encoding="ONE_HOT"
    )

    strategy.domain.set_experiments(
        generate_experiments(domain, 100, tol=2.0, force_all_categories=True)
    )
    opt_bounds = strategy.get_bounds(optimize=True)
    fit_bounds = strategy.get_bounds(optimize=False)
    assert opt_bounds[0, -2] == opt_bounds[1, -2] == 1
    assert opt_bounds[0, -1] == opt_bounds[1, -1] == 2
    assert fit_bounds[0, -2] == 1
    assert fit_bounds[0, -1] == 1
    assert fit_bounds[1, -2] == 5
    assert fit_bounds[1, -1] == 7
    domain.experiments = None


@pytest.mark.parametrize(
    "domain, descriptor_encoding, categorical_encoding, categorical_method, expected",
    [
        (domains[0], "DESCRIPTOR", "ONE_HOT", "EXHAUSTIVE", {}),
        (domains[0], "CATEGORICAL", "ONE_HOT", "EXHAUSTIVE", {}),
        (domains[0], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE", {}),
        (domains[0], "CATEGORICAL", "ORDINAL", "EXHAUSTIVE", {}),
        (
            domains[1],
            "DESCRIPTOR",
            "ONE_HOT",
            "EXHAUSTIVE",
            {1: 3, 5: 1, 6: 0, 7: 0, 10: 1, 11: 2},
        ),
        (
            domains[1],
            "CATEGORICAL",
            "ONE_HOT",
            "EXHAUSTIVE",
            {1: 3, 5: 1, 6: 0, 7: 0, 11: 1, 12: 0, 13: 0},
        ),
        (
            domains[1],
            "DESCRIPTOR",
            "ONE_HOT",
            "FREE",
            {1: 3, 5: 1, 6: 0, 7: 0, 10: 1, 11: 2},
        ),
        (
            domains[1],
            "CATEGORICAL",
            "ONE_HOT",
            "FREE",
            {1: 3, 5: 1, 6: 0, 7: 0, 11: 1, 12: 0, 13: 0},
        ),
        (domains[1], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE", {1: 3, 3: 0, 6: 1, 7: 2}),
        (domains[1], "CATEGORICAL", "ORDINAL", "EXHAUSTIVE", {1: 3.0, 3: 0.0, 5: 0.0}),
        (domains[5], "CATEGORICAL", "ONE_HOT", "EXHAUSTIVE", {1: 3.0}),
        (domains[5], "CATEGORICAL", "ONE_HOT", "FREE", {1: 3.0, 2: 0}),
        (domains[5], "DESCRIPTOR", "ONE_HOT", "FREE", {1: 3.0}),
    ],
)
def test_base_get_fixed_features(
    domain, descriptor_encoding, categorical_encoding, categorical_method, expected
):

    myStrategy = DummyStrategy(
        domain=domain,
        descriptor_encoding=descriptor_encoding,
        categorical_encoding=categorical_encoding,
        categorical_method=categorical_method,
    )

    experiments = generate_experiments(domain, 100, tol=1.0)
    myStrategy.domain.set_experiments(experiments)

    fixed_features = myStrategy.get_fixed_features()

    assert fixed_features == expected
    domain.experiments = None


@pytest.mark.parametrize(
    "domain, descriptor_method, categorical_method, descriptor_encoding, categorical_encoding, expected",
    [
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "DESCRIPTOR",
            "ONE_HOT",
            [
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 2.0},
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 3.0, 5: 7.0},
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 5.0, 5: 1.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 2.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 3.0, 5: 7.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 5.0, 5: 1.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 2.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 3.0, 5: 7.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 5.0, 5: 1.0},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "CATEGORICAL",
            "ONE_HOT",
            [
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "DESCRIPTOR",
            "ORDINAL",
            [
                {1: 0.0, 2: 1.0, 3: 2.0},
                {1: 0.0, 2: 3.0, 3: 7.0},
                {1: 0.0, 2: 5.0, 3: 1.0},
                {1: 1.0, 2: 1.0, 3: 2.0},
                {1: 1.0, 2: 3.0, 3: 7.0},
                {1: 1.0, 2: 5.0, 3: 1.0},
                {1: 2.0, 2: 1.0, 3: 2.0},
                {1: 2.0, 2: 3.0, 3: 7.0},
                {1: 2.0, 2: 5.0, 3: 1.0},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "CATEGORICAL",
            "ORDINAL",
            [
                {1: 0.0, 2: 0.0},
                {1: 0.0, 2: 1.0},
                {1: 0.0, 2: 2.0},
                {1: 1.0, 2: 0.0},
                {1: 1.0, 2: 1.0},
                {1: 1.0, 2: 2.0},
                {1: 2.0, 2: 0.0},
                {1: 2.0, 2: 1.0},
                {1: 2.0, 2: 2.0},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "FREE",
            "DESCRIPTOR",
            "ONE_HOT",
            [{4: 1.0, 5: 2.0}, {4: 3.0, 5: 7.0}, {4: 5.0, 5: 1.0}],
        ),
        (domains[0], "EXHAUSTIVE", "FREE", "CATEGORICAL", "ONE_HOT", [{}]),
        (
            domains[0],
            "FREE",
            "EXHAUSTIVE",
            "CATEGORICAL",
            "ORDINAL",
            [
                {1: 0.0, 2: 0.0},
                {1: 0.0, 2: 1.0},
                {1: 0.0, 2: 2.0},
                {1: 1.0, 2: 0.0},
                {1: 1.0, 2: 1.0},
                {1: 1.0, 2: 2.0},
                {1: 2.0, 2: 0.0},
                {1: 2.0, 2: 1.0},
                {1: 2.0, 2: 2.0},
            ],
        ),
        (
            domains[0],
            "FREE",
            "EXHAUSTIVE",
            "CATEGORICAL",
            "ONE_HOT",
            [
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0},
                {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0},
            ],
        ),
        (
            domains[0],
            "FREE",
            "EXHAUSTIVE",
            "DESCRIPTOR",
            "ORDINAL",
            [{1: 0.0}, {1: 1.0}, {1: 2.0}],
        ),
        (
            domains[0],
            "FREE",
            "EXHAUSTIVE",
            "DESCRIPTOR",
            "ONE_HOT",
            [
                {1: 1.0, 2: 0.0, 3: 0.0},
                {1: 0.0, 2: 1.0, 3: 0.0},
                {1: 0.0, 2: 0.0, 3: 1.0},
            ],
        ),
        (domains[0], "FREE", "FREE", "CATEGORICAL", "ONE_HOT", [{}]),
        (domains[0], "FREE", "FREE", "DESCRIPTOR", "ONE_HOT", [{}]),
    ],
)
def test_base_get_categorical_combinations(
    domain,
    descriptor_method,
    categorical_method,
    descriptor_encoding,
    categorical_encoding,
    expected,
):
    myStrategy = DummyStrategy(
        domain=domain,
        descriptor_encoding=descriptor_encoding,
        categorical_encoding=categorical_encoding,
        descriptor_method=descriptor_method,
        categorical_method=categorical_method,
    )

    experiment = generate_experiments(domain, 10)
    myStrategy.tell(experiment)

    combo = myStrategy.get_categorical_combinations()

    assert combo == expected
    domain.experiments = None


@pytest.mark.parametrize("domain", [(domains[0])])
def test_base_invalid_pair_encoding_method(domain):
    with pytest.raises(ValueError):
        DummyStrategy(
            domain=domain, categorical_encoding="ORDINAL", categorical_method="FREE"
        )


def test_base_get_true_categorical_features():
    myStrategy = DummyStrategy(domain=domains[0], descriptor_encoding="CATEGORICAL")
    assert len(myStrategy.get_true_categorical_features()) == 2
    myStrategy = DummyStrategy(domain=domains[0], descriptor_encoding="DESCRIPTOR")
    assert len(myStrategy.get_true_categorical_features()) == 1


@pytest.mark.parametrize(
    "domain, expected",
    [
        # (domains[0], SingleTaskGP),  # FIXME: This produces a MixedSingleTaskGP
        (domains[1], MixedSingleTaskGP),
        (domains[2], ModelListGP),
        (domains[3], SingleTaskGP),
        (domains[4], ModelListGP),
    ],
)
def test_base_fit(domain, expected):
    data = generate_experiments(domain, row_count=5, tol=1.0, force_all_categories=True)
    domain.set_experiments(data)
    acq = random.choice(list(AcquisitionFunctionEnum))
    myStrategy = DummyStrategy(domain=domain, acquisition_function=acq)
    myStrategy.fit()
    assert isinstance(myStrategy.model, expected)
    domain.experiments = None


@pytest.mark.parametrize(
    "domain, acquisition_function",
    [
        (domains[0], random.choice(list(AcquisitionFunctionEnum))),
        (domains[1], random.choice(list(AcquisitionFunctionEnum))),
        (domains[2], random.choice(list(AcquisitionFunctionEnum))),
        (domains[3], random.choice(list(AcquisitionFunctionEnum))),
        (domains[4], random.choice(list(AcquisitionFunctionEnum))),
    ],
)
def test_base_predict(domain, acquisition_function):
    data = generate_experiments(domain, row_count=5, tol=1.0, force_all_categories=True)
    myStrategy = DummyStrategy(domain=domain, acquisition_function=acquisition_function)

    myStrategy.tell(data)
    predictions = myStrategy.predict(data)

    assert len(predictions.columns) == 2 * len(domain.get_feature_keys(OutputFeature))
    assert data.index[-1] == predictions.index[-1]
    domain.experiments = None


@pytest.mark.parametrize(
    "domain, descriptor_encoding, categorical_encoding, categorical_method, expected",
    [
        (domains[0], "DESCRIPTOR", "ONE_HOT", "FREE", list(range(1, 4))),
        (domains[0], "DESCRIPTOR", "ONE_HOT", "EXHAUSTIVE", list(range(1, 4))),
        (domains[0], "CATEGORICAL", "ONE_HOT", "EXHAUSTIVE", list(range(1, 7))),
        (domains[0], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE", [1]),
        (domains[0], "CATEGORICAL", "ORDINAL", "EXHAUSTIVE", [1, 2]),
        (domains[1], "DESCRIPTOR", "ONE_HOT", "FREE", list(range(2, 8))),
        (domains[1], "DESCRIPTOR", "ONE_HOT", "EXHAUSTIVE", list(range(2, 8))),
        (domains[1], "CATEGORICAL", "ONE_HOT", "EXHAUSTIVE", list(range(2, 14))),
        (domains[1], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE", [2, 3]),
        (domains[1], "CATEGORICAL", "ORDINAL", "EXHAUSTIVE", list(range(2, 6))),
    ],
)
def test_base_categorical_dims(
    domain, descriptor_encoding, categorical_encoding, categorical_method, expected
):
    myStrategy = DummyStrategy(
        domain=domain,
        descriptor_encoding=descriptor_encoding,
        categorical_encoding=categorical_encoding,
        categorical_method=categorical_method,
    )
    categorical_dims = myStrategy.categorical_dims
    assert categorical_dims == expected
    domain.experiments = None


# ask, tell and has_sufficient_experiments are tested in test_all


@pytest.mark.parametrize(
    "domain, descriptor_encoding, categorical_encoding, expected_feature_keys, expected_features2idx",
    [
        (
            domains[0],
            "DESCRIPTOR",
            "ONE_HOT",
            ["if1", "if3_c1", "if3_c2", "if3_c3", "if5_d1", "if5_d2"],
            {"if1": [0], "if3": [1, 2, 3], "if5": [4, 5]},
        ),
        (
            domains[0],
            "DESCRIPTOR",
            "ORDINAL",
            ["if1", "if3", "if5_d1", "if5_d2"],
            {"if1": [0], "if3": [1], "if5": [2, 3]},
        ),
        (
            domains[0],
            "CATEGORICAL",
            "ONE_HOT",
            ["if1", "if3_c1", "if3_c2", "if3_c3", "if5_c1", "if5_c2", "if5_c3"],
            {"if1": [0], "if3": [1, 2, 3], "if5": [4, 5, 6]},
        ),
        (
            domains[0],
            "CATEGORICAL",
            "ORDINAL",
            ["if1", "if3", "if5"],
            {"if1": [0], "if3": [1], "if5": [2]},
        ),
    ],
)
def test_base_init_domain(
    domain,
    descriptor_encoding,
    categorical_encoding,
    expected_feature_keys,
    expected_features2idx,
):
    myStrategy = DummyStrategy(
        domain=domain,
        descriptor_encoding=descriptor_encoding,
        categorical_encoding=categorical_encoding,
    )

    assert myStrategy.input_feature_keys == expected_feature_keys
    assert myStrategy.features2idx == expected_features2idx


@pytest.mark.parametrize("domain", domains)
def test_base_get_model_spec(domain):
    myStrategy = DummyStrategy(domain=domain)
    for key in myStrategy.domain.get_feature_keys(OutputFeature):
        spec = myStrategy.get_model_spec(key)
        assert spec.output_feature == key


@pytest.mark.parametrize(
    "domain, model_specs, descriptor_encoding, categorical_encoding, expected",
    [
        (
            domains[2],
            None,
            "DESCRIPTOR",
            "ONE_HOT",
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ],
        ),
        (
            domains[2],
            None,
            "DESCRIPTOR",
            "ORDINAL",
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
        ),
        (
            domains[2],
            None,
            "CATEGORICAL",
            "ONE_HOT",
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            ],
        ),
        (
            domains[2],
            None,
            "CATEGORICAL",
            "ORDINAL",
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
        ),
        (
            domains[2],
            model_specs[0],
            "DESCRIPTOR",
            "ONE_HOT",
            [[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
        ),
        (
            domains[2],
            model_specs[0],
            "DESCRIPTOR",
            "ORDINAL",
            [[0, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]],
        ),
        (
            domains[2],
            model_specs[0],
            "CATEGORICAL",
            "ONE_HOT",
            [
                [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            ],
        ),
        (
            domains[2],
            model_specs[0],
            "CATEGORICAL",
            "ORDINAL",
            [[0, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        ),
        (
            domains[2],
            model_specs[1],
            "DESCRIPTOR",
            "ONE_HOT",
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
        ),
        (
            domains[2],
            model_specs[1],
            "DESCRIPTOR",
            "ORDINAL",
            [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]],
        ),
        (
            domains[2],
            model_specs[1],
            "CATEGORICAL",
            "ONE_HOT",
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            ],
        ),
        (
            domains[2],
            model_specs[1],
            "CATEGORICAL",
            "ORDINAL",
            [[1, 2, 3, 4, 5], [0, 2, 3, 4, 5]],
        ),
        (
            domains[2],
            model_specs[2],
            "DESCRIPTOR",
            "ONE_HOT",
            [[0, 1, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3, 4, 8, 9, 10, 11]],
        ),
        (
            domains[2],
            model_specs[2],
            "DESCRIPTOR",
            "ORDINAL",
            [[0, 1, 3, 4, 5, 6, 7], [0, 1, 2, 4, 5, 6, 7]],
        ),
        (
            domains[2],
            model_specs[2],
            "CATEGORICAL",
            "ONE_HOT",
            [
                [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13],
            ],
        ),
        (
            domains[2],
            model_specs[2],
            "CATEGORICAL",
            "ORDINAL",
            [[0, 1, 3, 4, 5], [0, 1, 2, 4, 5]],
        ),
        (
            domains[2],
            model_specs[3],
            "DESCRIPTOR",
            "ONE_HOT",
            [[0, 1, 2, 3, 4, 5, 6, 7, 10, 11], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        ),
        (
            domains[2],
            model_specs[3],
            "DESCRIPTOR",
            "ORDINAL",
            [[0, 1, 2, 3, 6, 7], [0, 1, 2, 3, 4, 5]],
        ),
        (
            domains[2],
            model_specs[3],
            "CATEGORICAL",
            "ONE_HOT",
            [[0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        ),
        (
            domains[2],
            model_specs[3],
            "CATEGORICAL",
            "ORDINAL",
            [[0, 1, 2, 3, 5], [0, 1, 2, 3, 4]],
        ),
        (
            domains[2],
            model_specs[4],
            "DESCRIPTOR",
            "ONE_HOT",
            [[0, 2, 3, 4, 10, 11], [1, 5, 6, 7, 8, 9]],
        ),
        (
            domains[2],
            model_specs[4],
            "DESCRIPTOR",
            "ORDINAL",
            [[0, 2, 6, 7], [1, 3, 4, 5]],
        ),
        (
            domains[2],
            model_specs[4],
            "CATEGORICAL",
            "ONE_HOT",
            [[0, 2, 3, 4, 11, 12, 13], [1, 5, 6, 7, 8, 9, 10]],
        ),
        (domains[2], model_specs[4], "CATEGORICAL", "ORDINAL", [[0, 2, 5], [1, 3, 4]]),
    ],
)
def test_base_get_feature_indices(
    domain, model_specs, descriptor_encoding, categorical_encoding, expected
):
    myStrategy = DummyStrategy(
        domain=domain,
        descriptor_encoding=descriptor_encoding,
        categorical_encoding=categorical_encoding,
        model_specs=model_specs,
    )

    for i, key in enumerate(myStrategy.domain.get_feature_keys(OutputFeature)):
        assert myStrategy.get_feature_indices(key) == expected[i]
    domain.experiments = None
