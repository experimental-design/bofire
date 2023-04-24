import random

import bofire.data_models.surrogates.api as models
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    HammondDistanceKernel,
    MaternKernel,
    ScaleKernel,
)
from bofire.data_models.priors.api import BOTORCH_LENGTHCALE_PRIOR, BOTORCH_SCALE_PRIOR
from bofire.data_models.surrogates.api import ScalerEnum
from tests.bofire.data_models.specs.features import specs as features
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    models.SaasSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [CategoricalInput(key="cat1", categories=["a", "b", "c"])]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "warmup_steps": 16,
        "num_samples": 4,
        "thinning": 2,
        "scaler": ScalerEnum.NORMALIZE,
        "input_preprocessing_specs": {"cat1": CategoricalEncodingEnum.ONE_HOT},
        "dump": None,
    },
)

specs.add_valid(
    models.MixedSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [CategoricalInput(key="cat1", categories=["a", "b", "c"])]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "continuous_kernel": MaternKernel(ard=True, nu=random.random()),
        "categorical_kernel": HammondDistanceKernel(ard=True),
        "scaler": ScalerEnum.NORMALIZE,
        "input_preprocessing_specs": {"cat1": CategoricalEncodingEnum.ONE_HOT},
        "dump": None,
    },
)
specs.add_valid(
    models.SingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR,
        ),
        "scaler": ScalerEnum.NORMALIZE,
        "input_preprocessing_specs": {},
        "dump": None,
    },
)
specs.add_valid(
    models.RandomForestSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "input_preprocessing_specs": {},
        "n_estimators": 100,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "random_state": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
        "dump": None,
    },
)
specs.add_valid(
    models.MLPEnsemble,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "n_estimators": 2,
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "dropout": 0.0,
        "batch_size": 10,
        "n_epochs": 200,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "subsample_fraction": 1.0,
        "shuffle": True,
        "scaler": ScalerEnum.NORMALIZE,
        "input_preprocessing_specs": {},
        "dump": None,
    },
)
