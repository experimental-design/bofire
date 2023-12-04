import random

import bofire.data_models.surrogates.api as models
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    MolecularInput,
)
from bofire.data_models.kernels.api import (
    HammondDistanceKernel,
    MaternKernel,
    ScaleKernel,
    TanimotoKernel,
)
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.priors.api import (
    BOTORCH_LENGTHCALE_PRIOR,
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
)
from bofire.data_models.surrogates.api import (
    MeanAggregation,
    ScalerEnum,
    SumAggregation,
)
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPHyperconfig
from tests.bofire.data_models.specs.features import specs as features
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    models.SingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=(0, 1)),
                ContinuousInput(key="b", bounds=(0, 1)),
            ]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR()
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR(),
        ),
        "aggregations": [
            random.choice(
                [
                    SumAggregation(features=["a", "b"]),
                    MeanAggregation(features=["a", "b"]),
                ]
            )
        ],
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": BOTORCH_NOISE_PRIOR(),
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig(),
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
        "aggregations": None,
        "continuous_kernel": MaternKernel(ard=True, nu=random.random()),
        "categorical_kernel": HammondDistanceKernel(ard=True),
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {"cat1": CategoricalEncodingEnum.ONE_HOT},
        "dump": None,
        "hyperconfig": None,
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
                ard=True, nu=2.5, lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR()
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR(),
        ),
        "aggregations": None,
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": BOTORCH_NOISE_PRIOR(),
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig(),
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
        "aggregations": None,
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
        "hyperconfig": None,
        "scaler": ScalerEnum.IDENTITY,
        "output_scaler": ScalerEnum.IDENTITY,
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
        "aggregations": None,
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
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": None,
    },
)

specs.add_valid(
    models.XGBoostSurrogate,
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
        "aggregations": None,
        "n_estimators": 10,
        "max_depth": 6,
        "max_leaves": 0,
        "max_bin": 256,
        "grow_policy": "depthwise",
        "learning_rate": 0.3,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "n_jobs": 1,
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "max_delta_step": 0.0,
        "subsample": 1.0,
        "sampling_method": "uniform",
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1,
        "random_state": None,
        "num_parallel_tree": 1,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": None,
    },
)
specs.add_valid(
    models.TanimotoGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                MolecularInput(key="mol1"),
            ]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ]
        ),
        "kernel": ScaleKernel(
            base_kernel=TanimotoKernel(
                ard=True,
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR(),
        ),
        "aggregations": None,
        "scaler": ScalerEnum.IDENTITY,
        "output_scaler": ScalerEnum.IDENTITY,
        "noise_prior": BOTORCH_NOISE_PRIOR(),
        "input_preprocessing_specs": {"mol1": Fingerprints(n_bits=32, bond_radius=3)},
        "dump": None,
        "hyperconfig": None,
    },
)
