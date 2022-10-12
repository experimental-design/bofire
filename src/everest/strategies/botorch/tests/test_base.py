import random
import warnings
from typing import Type

import pandas as pd
import pytest
import torch
from botorch.models import MixedSingleTaskGP, ModelListGP, SingleTaskGP
from everest.domain import Domain
from everest.domain.constraints import (Constraint, LinearConstraint,
                                        LinearEqualityConstraint,
                                        LinearInequalityConstraint)
from everest.domain.features import (CategoricalDescriptorInputFeature,
                                     CategoricalInputFeature,
                                     ContinuousInputFeature,
                                     ContinuousOutputFeature,
                                     ContinuousOutputFeature_woDesFunc,
                                     OutputFeature)
from everest.domain.tests.test_domain_validators import (generate_candidates,
                                                         generate_experiments)
from everest.domain.tests.test_features import (
    VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC, VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC)
from everest.domain.util import KeyModel
from everest.strategies.botorch import tkwargs
from everest.strategies.botorch.base import BotorchBasicBoStrategy
from everest.strategies.botorch.sobo import AcquisitionFunctionEnum
from everest.strategies.botorch.utils.objectives import MultiplicativeObjective
from everest.strategies.strategy import ModelSpec

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
        self.objective = MultiplicativeObjective(targets=[var.desirability_function for var in self.domain.get_features(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])])
        return

    @classmethod
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearConstraint,
            LinearEqualityConstraint,
            LinearInequalityConstraint,
        ]

if1 = ContinuousInputFeature(**{
    **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if1",
})
if2 = ContinuousInputFeature(**{
    **VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if2",
})

if3 = CategoricalInputFeature(**{
    **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    "key": "if3",
})

if4 = CategoricalInputFeature(**{
    **VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
    "key": "if4",
})

if5 = CategoricalDescriptorInputFeature(**{
    **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    "key": "if5",
})

if6 = CategoricalDescriptorInputFeature(**{
    **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    "key": "if6",
})

if7 = DummyFeature(key = "if7")

if8 = CategoricalDescriptorInputFeature(**{
    **VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    "key": "if8",
    })

of1 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of1",
})

of2 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of2",
})

domains = [
    Domain(
        input_features=[if1, if3, if5], # no fixed features
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if2, if3, if4, if5, if6], # all feature types incl. with fixed values
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if2, if3, if4, if5, if6], # all feature types incl. with fixed values + mutli-objective
        output_features=[of1, of2],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if2], # only continuous features
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if3, if5], # all feature types + mutli-objective
        output_features=[of1, of2],
        constraints=[],
    ),
    Domain(
        input_features=[if1,if2,if8],
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
    pd.DataFrame.from_dict({
        "if1": [3,4,5,4.5],
        "if3": ["c1", "c2", "c3","c1"],
        "if5": ["c1", "c2", "c3","c1"],
        "of1": [10,11,12,13],
        "valid_of1": [1,0,1,0]
    }
    ),
    pd.DataFrame.from_dict({
        "if1": [3,4,5,4.5],
        "if2": [3,3,3,3],
        "if3": ["c1", "c2", "c3","c1"],
        "if4": ["c1","c1","c1","c1"],
        "if5": ["c1", "c2", "c3","c1"],
        "if6": ["c1","c1","c1","c1"],
        "of1": [10,11,12,13],
        "valid_of1": [1,0,1,0]
    }
    ),
    pd.DataFrame.from_dict({
        "if1": [3,4,5,4.5],
        "if2": [3,3,3,3],
        "if3": ["c1", "c2", "c3","c1"],
        "if4": ["c1","c1","c1","c1"],
        "if5": ["c1", "c2", "c3","c1"],
        "if6": ["c1","c1","c1","c1"],
        "of1": [10,11,12,13],
        "of2": [100,103,105,110],
        "valid_of1": [1,0,1,0],
        "valid_of2": [0,1,1,0]
    }
    ),
    pd.DataFrame.from_dict({
        "if1": [3,4,5,4.5],
        "if2": [3,3,3,3],
        "of1": [10,11,12,13],
        "valid_of1": [1,0,1,0]
    }
    ),
    pd.DataFrame.from_dict({
        "if1": [3,4,5,4.5],
        "if3": ["c1", "c2", "c3","c1"],
        "if5": ["c1", "c2", "c3","c1"],
        "of1": [10,11,12,13],
        "of2": [100,103,105,110],
        "valid_of1": [1,0,1,0],
        "valid_of2": [0,1,1,0]
    }
    ),
]


# model specs applicable for domains[2]
model_specs = [
    [
        ModelSpec(
            output_feature = "of1",
            input_features = ["if1","if3","if4","if5","if6"],
            kernel = "RBF",
            scaler = "NORMALIZE",
            ard = True,
        ),
        ModelSpec(
            output_feature = "of2",
            input_features = ["if2","if3","if4","if5","if6"],
            kernel = "MATERN_25",
            scaler = "STANDARDIZE",
            ard = False,
        )
    ],
    [
        ModelSpec(
            output_feature = "of2",
            input_features = ["if1","if3","if4","if5","if6"],
            kernel = "RBF",
            scaler = "NORMALIZE",
            ard = True,
        ),
        ModelSpec(
            output_feature = "of1",
            input_features = ["if2","if3","if4","if5","if6"],
            kernel = "MATERN_15",
            scaler = "STANDARDIZE",
            ard = False,
        )
    ],
    [
        ModelSpec(
            output_feature = "of1",
            input_features = ["if1","if2","if4","if5","if6"],
            kernel = "RBF",
            scaler = "NORMALIZE",
            ard = True,
        ),
        ModelSpec(
            output_feature = "of2",
            input_features = ["if1","if2","if3","if5","if6"],
            kernel = "MATERN_25",
            scaler = "STANDARDIZE",
            ard = False,
        )
    ],
    [
        ModelSpec(
            output_feature = "of1",
            input_features = ["if1","if2","if3","if4","if6"],
            kernel = "RBF",
            scaler = "NORMALIZE",
            ard = True,
        ),
        ModelSpec(
            output_feature = "of2",
            input_features = ["if1","if2","if3","if4","if5"],
            kernel = "MATERN_25",
            scaler = "STANDARDIZE",
            ard = False,
        )
    ],
    [
        ModelSpec(
            output_feature = "of1",
            input_features = ["if1","if3","if6"],
            kernel = "RBF",
            scaler = "NORMALIZE",
            ard = True,
        ),
        ModelSpec(
            output_feature = "of2",
            input_features = ["if2","if4","if5"],
            kernel = "MATERN_25",
            scaler = "STANDARDIZE",
            ard = False,
        )
    ],
]


def test_base_create():
    with pytest.raises(ValueError, match="number sobol samples"):
        DummyStrategy(num_sobol_samples = 5)
        
    with pytest.raises(ValueError, match="number raw samples"):
        DummyStrategy(num_raw_samples = 5)

@pytest.mark.parametrize("domain, descriptor_encoding, categorical_encoding, expected_bounds", [
    (domains[0], "DESCRIPTOR", "ONE_HOT", torch.tensor([[3,0,0,0,1,1], [5.3,1,1,1,5,7]]).to(**tkwargs)),
    (domains[0], "CATEGORICAL", "ONE_HOT", torch.tensor([[3,0,0,0,0,0,0], [5.3,1,1,1,1,1,1]]).to(**tkwargs)),
    (domains[0], "DESCRIPTOR", "ORDINAL", torch.tensor([[3,0,1,1], [5.3,2,5,7]]).to(**tkwargs)),
    (domains[0], "CATEGORICAL", "ORDINAL", torch.tensor([[3,0,0], [5.3,2,2]]).to(**tkwargs)),
    (domains[1], "DESCRIPTOR", "ONE_HOT", torch.tensor([[3,3,0,0,0,0,0,0,1,1,1,2], [5.3,3,1,1,1,1,1,1,5,7,1,2]]).to(**tkwargs)),
    (domains[1], "CATEGORICAL", "ONE_HOT", torch.tensor([[3,3,0,0,0,0,0,0,0,0,0,0,0,0], [5.3,3,1,1,1,1,1,1,1,1,1,1,1,1]]).to(**tkwargs)),
    (domains[1], "DESCRIPTOR", "ORDINAL", torch.tensor([[3,3, 0,0,1,1,1,2], [5.3,3,2,2,5,7,1,2]]).to(**tkwargs)),
    (domains[1], "CATEGORICAL", "ORDINAL", torch.tensor([[3,3,0,0,0,0], [5.3,3,2,2,2,2]]).to(**tkwargs))
])
def test_base_get_bounds(domain,descriptor_encoding, categorical_encoding, expected_bounds):
    strategy = DummyStrategy(descriptor_encoding= descriptor_encoding,
            categorical_encoding= categorical_encoding)

    strategy.init_domain(domain=domain)
    
    bounds = strategy.get_bounds()

    assert torch.allclose(bounds, expected_bounds) #torch.equal asserts false due to deviation of 1e-7??

def test_base_get_bounds_fit():
    # at first the fix on the continuous ones is tested
    strategy = DummyStrategy(descriptor_encoding= "DESCRIPTOR",
            categorical_encoding= "ONE_HOT")
    domain = domains[3]
    strategy.init_domain(domain=domain)
    strategy.experiments = generate_experiments(domain, 100, tol = 2.)
    opt_bounds = strategy.get_bounds(optimize=True)
    fit_bounds = strategy.get_bounds(optimize=False)
    for i,key in enumerate(domain.get_feature_keys(ContinuousInputFeature)):
        assert fit_bounds[0,i] < opt_bounds[0,i]
        assert fit_bounds[1,i] > opt_bounds[1,i]
        assert fit_bounds[0,i] == strategy.experiments[key].min()
        assert fit_bounds[1,i] == strategy.experiments[key].max()
    # next test the fix for the CategoricalDescriptor feature
    strategy = DummyStrategy(descriptor_encoding= "DESCRIPTOR",
            categorical_encoding= "ONE_HOT")
    domain = domains[1]
    strategy.init_domain(domain=domain)
    strategy.experiments = generate_experiments(domain, 100, tol = 2.,force_all_categories=True)
    opt_bounds = strategy.get_bounds(optimize=True)
    fit_bounds = strategy.get_bounds(optimize=False)
    assert opt_bounds[0,-2] == opt_bounds[1,-2] == 1
    assert opt_bounds[0,-1] == opt_bounds[1,-1] == 2
    assert fit_bounds[0,-2] == 1
    assert fit_bounds[0,-1] == 1
    assert fit_bounds[1,-2] == 5
    assert fit_bounds[1,-1] == 7


@pytest.mark.parametrize("domain, descriptor_encoding, categorical_encoding, categorical_method, expected", [
    (domains[0], "DESCRIPTOR", "ONE_HOT","EXHAUSTIVE", {}),
    (domains[0], "CATEGORICAL", "ONE_HOT","EXHAUSTIVE",{}),
    (domains[0], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE",{}),
    (domains[0], "CATEGORICAL", "ORDINAL","EXHAUSTIVE",{}),
    (domains[1], "DESCRIPTOR", "ONE_HOT", "EXHAUSTIVE", {1: 3, 5: 1, 6: 0, 7: 0, 10: 1, 11: 2}),
    (domains[1], "CATEGORICAL", "ONE_HOT","EXHAUSTIVE", {1: 3, 5: 1, 6: 0, 7: 0, 11: 1, 12: 0, 13: 0}),
    (domains[1], "DESCRIPTOR", "ONE_HOT", "FREE", {1: 3, 5: 1, 6: 0, 7: 0, 10: 1, 11: 2}),
    (domains[1], "CATEGORICAL", "ONE_HOT","FREE", {1: 3, 5: 1, 6: 0, 7: 0, 11: 1, 12: 0, 13: 0}),
    (domains[1], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE", {1: 3, 3: 0, 6: 1, 7: 2}),
    (domains[1], "CATEGORICAL", "ORDINAL","EXHAUSTIVE", {1: 3., 3: 0., 5: 0.}),
    (domains[5], "CATEGORICAL", "ONE_HOT","EXHAUSTIVE", {1: 3.}),
    (domains[5], "CATEGORICAL", "ONE_HOT","FREE", {1: 3.,2:0}),
    (domains[5], "DESCRIPTOR", "ONE_HOT","FREE", {1: 3.})
    ])
def test_base_get_fixed_features(domain, descriptor_encoding, categorical_encoding, categorical_method, expected):

    myStrategy = DummyStrategy(
        domain=domain,
        descriptor_encoding = descriptor_encoding, 
        categorical_encoding = categorical_encoding,
        categorical_method = categorical_method)
    myStrategy.init_domain(domain)

    experiments = generate_experiments(domain, 100, tol=1.)
    myStrategy.experiments = experiments

    fixed_features = myStrategy.get_fixed_features()
    
    assert fixed_features == expected

@pytest.mark.parametrize("domain, descriptor_method, categorical_method, descriptor_encoding, categorical_encoding, expected", [
    (domains[0], "EXHAUSTIVE", "EXHAUSTIVE", "DESCRIPTOR", "ONE_HOT", [{1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 2.0},
                                                                        {1: 1.0, 2: 0.0, 3: 0.0, 4: 3.0, 5: 7.0},
                                                                        {1: 1.0, 2: 0.0, 3: 0.0, 4: 5.0, 5: 1.0},
                                                                        {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 2.0},
                                                                        {1: 0.0, 2: 1.0, 3: 0.0, 4: 3.0, 5: 7.0},
                                                                        {1: 0.0, 2: 1.0, 3: 0.0, 4: 5.0, 5: 1.0},
                                                                        {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 2.0},
                                                                        {1: 0.0, 2: 0.0, 3: 1.0, 4: 3.0, 5: 7.0},
                                                                        {1: 0.0, 2: 0.0, 3: 1.0, 4: 5.0, 5: 1.0}]),
    (domains[0], "EXHAUSTIVE", "EXHAUSTIVE", "CATEGORICAL", "ONE_HOT", [{1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                                                                        {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                                                                        {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                                                                        {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                                                                        {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                                                                        {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                                                                        {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
                                                                        {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0},
                                                                        {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0}]),
    (domains[0], "EXHAUSTIVE", "EXHAUSTIVE",  "DESCRIPTOR", "ORDINAL", [{1: 0.0, 2: 1.0, 3: 2.0},
                                                                        {1: 0.0, 2: 3.0, 3: 7.0},
                                                                        {1: 0.0, 2: 5.0, 3: 1.0},
                                                                        {1: 1.0, 2: 1.0, 3: 2.0},
                                                                        {1: 1.0, 2: 3.0, 3: 7.0},
                                                                        {1: 1.0, 2: 5.0, 3: 1.0},
                                                                        {1: 2.0, 2: 1.0, 3: 2.0},
                                                                        {1: 2.0, 2: 3.0, 3: 7.0},
                                                                        {1: 2.0, 2: 5.0, 3: 1.0}]),
    (domains[0], "EXHAUSTIVE", "EXHAUSTIVE",  "CATEGORICAL", "ORDINAL", [{1: 0.0, 2: 0.0},
                                                                        {1: 0.0, 2: 1.0},
                                                                        {1: 0.0, 2: 2.0},
                                                                        {1: 1.0, 2: 0.0},
                                                                        {1: 1.0, 2: 1.0},
                                                                        {1: 1.0, 2: 2.0},
                                                                        {1: 2.0, 2: 0.0},
                                                                        {1: 2.0, 2: 1.0},
                                                                        {1: 2.0, 2: 2.0}]),
    (domains[0], "EXHAUSTIVE", "FREE", "DESCRIPTOR", "ONE_HOT", [{4: 1.0, 5: 2.0},
                                                                {4: 3.0, 5: 7.0},
                                                                {4: 5.0, 5: 1.0}]),
    (domains[0], "EXHAUSTIVE", "FREE", "CATEGORICAL", "ONE_HOT", [{}]),
    (domains[0], "FREE", "EXHAUSTIVE",  "CATEGORICAL", "ORDINAL", [{1: 0.0, 2: 0.0},
                                                                        {1: 0.0, 2: 1.0},
                                                                        {1: 0.0, 2: 2.0},
                                                                        {1: 1.0, 2: 0.0},
                                                                        {1: 1.0, 2: 1.0},
                                                                        {1: 1.0, 2: 2.0},
                                                                        {1: 2.0, 2: 0.0},
                                                                        {1: 2.0, 2: 1.0},
                                                                        {1: 2.0, 2: 2.0}]),
    (domains[0], "FREE", "EXHAUSTIVE",  "CATEGORICAL", "ONE_HOT", [{1: 1.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                                                                    {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                                                                    {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                                                                    {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0},
                                                                    {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
                                                                    {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0},
                                                                    {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0},
                                                                    {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0},
                                                                    {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0}]),
    (domains[0], "FREE", "EXHAUSTIVE",  "DESCRIPTOR", "ORDINAL", [{1: 0.0}, {1: 1.0}, {1: 2.0}]),
    (domains[0], "FREE", "EXHAUSTIVE",  "DESCRIPTOR", "ONE_HOT", [{1: 1.0, 2: 0.0, 3: 0.0},
                                                                {1: 0.0, 2: 1.0, 3: 0.0},
                                                                {1: 0.0, 2: 0.0, 3: 1.0}]),
    (domains[0], "FREE", "FREE",  "CATEGORICAL", "ONE_HOT", [{}]),
    (domains[0], "FREE", "FREE",  "DESCRIPTOR", "ONE_HOT", [{}])
    ])
def test_base_get_categorical_combinations(domain, descriptor_method, categorical_method, descriptor_encoding, categorical_encoding, expected):
    myStrategy = DummyStrategy(domain=domain, descriptor_encoding = descriptor_encoding, categorical_encoding = categorical_encoding, descriptor_method=descriptor_method, categorical_method=categorical_method)
    myStrategy.init_domain(domain)

    experiment = generate_experiments(domain, 10)
    myStrategy.tell(experiment)

    combo = myStrategy.get_categorical_combinations()
   
    assert combo == expected

@pytest.mark.parametrize("domain", [
    (domains[0])
    ])
def test_base_invalid_pair_encoding_method(domain):
    with pytest.raises(ValueError):
        myStrategy = DummyStrategy(domain=domain, categorical_encoding = "ORDINAL", categorical_method="FREE")
    
def test_base_get_true_categorical_features():
    myStrategy = DummyStrategy(domain = domains[0], descriptor_encoding="CATEGORICAL")
    assert len(myStrategy.get_true_categorical_features()) == 2
    myStrategy = DummyStrategy(domain = domains[0], descriptor_encoding="DESCRIPTOR")
    assert len(myStrategy.get_true_categorical_features()) == 1


@pytest.mark.parametrize("domain, data, acquisition_function, expected", [
    (domains[0], generate_experiments(domains[0],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum)), SingleTaskGP),
    (domains[1], generate_experiments(domains[1],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum)), MixedSingleTaskGP),
    (domains[2], generate_experiments(domains[2],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum)), ModelListGP),
    (domains[3], generate_experiments(domains[3],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum)), SingleTaskGP),
    (domains[4], generate_experiments(domains[4],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum)), ModelListGP),
    ])
def test_base_fit(domain, data, acquisition_function, expected):
   
    myStrategy = DummyStrategy(acquisition_function=acquisition_function)
    myStrategy.init_domain(domain=domain)
    myStrategy.experiments = data

    myStrategy.fit()

    assert isinstance(myStrategy.model, expected)

@pytest.mark.parametrize("domain, data, acquisition_function", [
    (domains[0], generate_experiments(domains[0],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum))),
    (domains[1], generate_experiments(domains[1],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum))),
    (domains[2], generate_experiments(domains[2],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum))),
    (domains[3], generate_experiments(domains[3],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum))),
    (domains[4], generate_experiments(domains[4],row_count=5, tol=1., force_all_categories=True), random.choice(list(AcquisitionFunctionEnum))),
    ])
def test_base_predict(domain, data, acquisition_function):
    myStrategy = DummyStrategy(acquisition_function=acquisition_function)
    myStrategy.init_domain(domain=domain)

    myStrategy.tell(data)
    predictions = myStrategy.predict(data)

    assert len(predictions.columns.tolist()) == 2*len(domain.get_feature_keys(OutputFeature))
    assert data.index[-1] == predictions.index[-1]


@pytest.mark.parametrize("domain, descriptor_encoding, categorical_encoding, categorical_method, expected", [
    (domains[0], "DESCRIPTOR", "ONE_HOT", "FREE", list(range(1,4))),
    (domains[0], "DESCRIPTOR", "ONE_HOT", "EXHAUSTIVE", list(range(1, 4))),
    (domains[0], "CATEGORICAL", "ONE_HOT", "EXHAUSTIVE",list(range(1, 7))),
    (domains[0], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE", [1]),
    (domains[0], "CATEGORICAL", "ORDINAL", "EXHAUSTIVE", [1, 2]),
    (domains[1], "DESCRIPTOR", "ONE_HOT", "FREE", list(range(2, 8))),
    (domains[1], "DESCRIPTOR", "ONE_HOT", "EXHAUSTIVE", list(range(2, 8))),
    (domains[1], "CATEGORICAL", "ONE_HOT", "EXHAUSTIVE", list(range(2, 14))),
    (domains[1], "DESCRIPTOR", "ORDINAL", "EXHAUSTIVE", [2, 3]),
    (domains[1], "CATEGORICAL", "ORDINAL", "EXHAUSTIVE", list(range(2,6)))
    ])
def test_base_categorical_dims(domain, descriptor_encoding, categorical_encoding, categorical_method, expected):
    myStrategy = DummyStrategy(domain=domain, descriptor_encoding = descriptor_encoding, 
                                categorical_encoding = categorical_encoding, categorical_method = categorical_method
                                )
    myStrategy.init_domain(domain)
    categorical_dims = myStrategy.categorical_dims
    assert categorical_dims == expected

# ask, tell and has_sufficient_experiments are tested in test_all

@pytest.mark.parametrize("domain, descriptor_encoding, categorical_encoding, expected_feature_keys, expected_features2idx", [
    (
        domains[0], "DESCRIPTOR", "ONE_HOT",
        ["if1", "if3_c1", "if3_c2", "if3_c3","if5_d1", "if5_d2"],
        {"if1":[0], "if3":[1,2,3], "if5":[4,5]}
    ),
    (
        domains[0], "DESCRIPTOR", "ORDINAL",
        ["if1", "if3","if5_d1", "if5_d2"],
        {"if1":[0], "if3":[1], "if5":[2,3]}
    ),
    (
        domains[0], "CATEGORICAL", "ONE_HOT",
        ["if1", "if3_c1", "if3_c2", "if3_c3","if5_c1", "if5_c2", "if5_c3"],
        {"if1":[0], "if3":[1,2,3], "if5":[4,5,6]}
    ),
    (
        domains[0], "CATEGORICAL", "ORDINAL",
        ["if1", "if3", "if5"],
        {"if1":[0], "if3":[1], "if5":[2]}
    ),
])
def test_base_init_domain(domain,descriptor_encoding, categorical_encoding, expected_feature_keys, expected_features2idx):
    myStrategy = DummyStrategy(
        descriptor_encoding = descriptor_encoding,
        categorical_encoding = categorical_encoding,
    )
    myStrategy.init_domain(domain)
    assert myStrategy.input_feature_keys == expected_feature_keys
    assert myStrategy.features2idx == expected_features2idx

@pytest.mark.parametrize("domain", domains)
def test_base_get_model_spec(domain):
    myStrategy = DummyStrategy()
    myStrategy.init_domain(domain)
    for key in myStrategy.domain.get_feature_keys(OutputFeature):
        spec = myStrategy.get_model_spec(key)
        assert spec.output_feature == key

@pytest.mark.parametrize("domain, model_specs, descriptor_encoding, categorical_encoding, expected",[
    (domains[2],None,"DESCRIPTOR", "ONE_HOT",[[0,1,2,3,4,5,6,7,8,9,10,11],[0,1,2,3,4,5,6,7,8,9,10,11]]),
    (domains[2],None,"DESCRIPTOR", "ORDINAL",[[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]]),
    (domains[2],None,"CATEGORICAL", "ONE_HOT",[[0,1,2,3,4,5,6,7,8,9,10,11,12,13],[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]),
    (domains[2],None,"CATEGORICAL", "ORDINAL",[[0,1,2,3,4,5],[0,1,2,3,4,5]]),
    (domains[2],model_specs[0],"DESCRIPTOR", "ONE_HOT",[[0,2,3,4,5,6,7,8,9,10,11],[1,2,3,4,5,6,7,8,9,10,11]]),
    (domains[2],model_specs[0],"DESCRIPTOR", "ORDINAL",[[0,2,3,4,5,6,7],[1,2,3,4,5,6,7]]),
    (domains[2],model_specs[0],"CATEGORICAL", "ONE_HOT",[[0,2,3,4,5,6,7,8,9,10,11,12,13],[1,2,3,4,5,6,7,8,9,10,11,12,13]]),
    (domains[2],model_specs[0],"CATEGORICAL", "ORDINAL",[[0,2,3,4,5],[1,2,3,4,5]]),
    (domains[2],model_specs[1],"DESCRIPTOR", "ONE_HOT",[[1,2,3,4,5,6,7,8,9,10,11],[0,2,3,4,5,6,7,8,9,10,11]]),
    (domains[2],model_specs[1],"DESCRIPTOR", "ORDINAL",[[1,2,3,4,5,6,7],[0,2,3,4,5,6,7]]),
    (domains[2],model_specs[1],"CATEGORICAL", "ONE_HOT",[[1,2,3,4,5,6,7,8,9,10,11,12,13],[0,2,3,4,5,6,7,8,9,10,11,12,13]]),
    (domains[2],model_specs[1],"CATEGORICAL", "ORDINAL",[[1,2,3,4,5],[0,2,3,4,5]]),
    (domains[2],model_specs[2],"DESCRIPTOR", "ONE_HOT",[[0,1,5,6,7,8,9,10,11],[0,1,2,3,4,8,9,10,11]]),
    (domains[2],model_specs[2],"DESCRIPTOR", "ORDINAL",[[0,1,3,4,5,6,7],[0,1,2,4,5,6,7]]),
    (domains[2],model_specs[2],"CATEGORICAL", "ONE_HOT",[[0,1,5,6,7,8,9,10,11,12,13],[0,1,2,3,4,8,9,10,11,12,13]]),
    (domains[2],model_specs[2],"CATEGORICAL", "ORDINAL",[[0,1,3,4,5],[0,1,2,4,5]]),
    (domains[2],model_specs[3],"DESCRIPTOR", "ONE_HOT",[[0,1,2,3,4,5,6,7,10,11],[0,1,2,3,4,5,6,7,8,9]]),
    (domains[2],model_specs[3],"DESCRIPTOR", "ORDINAL",[[0,1,2,3,6,7],[0,1,2,3,4,5]]),
    (domains[2],model_specs[3],"CATEGORICAL", "ONE_HOT",[[0,1,2,3,4,5,6,7,11,12,13],[0,1,2,3,4,5,6,7,8,9,10]]),
    (domains[2],model_specs[3],"CATEGORICAL", "ORDINAL",[[0,1,2,3,5],[0,1,2,3,4]]),
    (domains[2],model_specs[4],"DESCRIPTOR", "ONE_HOT",[[0,2,3,4,10,11],[1,5,6,7,8,9]]),
    (domains[2],model_specs[4],"DESCRIPTOR", "ORDINAL",[[0,2,6,7],[1,3,4,5]]),
    (domains[2],model_specs[4],"CATEGORICAL", "ONE_HOT",[[0,2,3,4,11,12,13],[1,5,6,7,8,9,10]]),
    (domains[2],model_specs[4],"CATEGORICAL", "ORDINAL",[[0,2,5],[1,3,4]]),
])
def test_base_get_feature_indices(domain, model_specs, descriptor_encoding, categorical_encoding, expected):
    myStrategy = DummyStrategy(
        descriptor_encoding = descriptor_encoding,
        categorical_encoding = categorical_encoding,
        model_specs = model_specs,
    )
    myStrategy.init_domain(domain)
    for i, key in enumerate(myStrategy.domain.get_feature_keys(OutputFeature)):
        assert myStrategy.get_feature_indices(key) == expected[i]





