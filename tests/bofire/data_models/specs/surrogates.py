import random

import bofire.data_models.surrogates.api as models
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
    MolecularInput,
    TaskInput,
)
from bofire.data_models.kernels.api import (
    HammingDistanceKernel,
    InfiniteWidthBNNKernel,
    MaternKernel,
    ScaleKernel,
    TanimotoKernel,
    WassersteinKernel,
)
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.priors.api import (
    ROBUSTGP_LENGTHSCALE_CONSTRAINT,
    ROBUSTGP_OUTPUTSCALE_CONSTRAINT,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    LogNormalPrior,
)
from bofire.data_models.surrogates.api import (
    MeanAggregation,
    ScalerEnum,
    SumAggregation,
)
from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPHyperconfig
from bofire.data_models.surrogates.shape import PiecewiseLinearGPSurrogateHyperconfig
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPHyperconfig
from tests.bofire.data_models.specs.features import specs as features
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    models.SingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=[0, 1]),
                ContinuousInput(key="b", bounds=[0, 1]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ).model_dump(),
        "aggregations": [
            random.choice(
                [
                    SumAggregation(features=["a", "b"]).model_dump(),
                    MeanAggregation(features=["a", "b"]).model_dump(),
                ],
            ),
        ],
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig().model_dump(),
    },
)

specs.add_valid(
    models.RobustSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=[0, 1]),
                ContinuousInput(key="b", bounds=[0, 1]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True,
                nu=2.5,
                lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
            outputscale_constraint=ROBUSTGP_OUTPUTSCALE_CONSTRAINT(),
        ).model_dump(),
        "aggregations": [
            random.choice(
                [
                    SumAggregation(features=["a", "b"]).model_dump(),
                    MeanAggregation(features=["a", "b"]).model_dump(),
                ],
            ),
        ],
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "cache_model_trace": False,
        "convex_parametrization": True,
        "prior_mean_of_support": None,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig().model_dump(),
    },
)

specs.add_invalid(
    models.RobustSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=[0, 1]),
                ContinuousInput(key="b", bounds=[0, 1]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                ContinuousOutput(key="a"),
                ContinuousOutput(key="b"),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True,
                nu=2.5,
                lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
            outputscale_constraint=ROBUSTGP_OUTPUTSCALE_CONSTRAINT(),
        ).model_dump(),
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "cache_model_trace": False,
        "convex_parametrization": True,
        "prior_mean_of_support": None,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig().model_dump(),
    },
    error=ValueError,
    message="RobustGP only supports one output.",
)


specs.add_valid(
    models.SingleTaskIBNNSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=(0, 1)),
                ContinuousInput(key="b", bounds=(0, 1)),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "aggregations": None,
        "dump": None,
        "kernel": InfiniteWidthBNNKernel(depth=3).model_dump(),
    },
)

specs.add_valid(
    models.AdditiveMapSaasSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "aggregations": None,
        "n_taus": 4,
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {},
        "hyperconfig": None,
        "dump": None,
    },
)

specs.add_valid(
    models.FullyBayesianSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "warmup_steps": 256,
        "num_samples": 128,
        "thinning": 16,
        "aggregations": None,
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {},
        "hyperconfig": None,
        "dump": None,
        "model_type": "saas",
        "features_to_warp": [],
    },
)


specs.add_invalid(
    models.FullyBayesianSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "warmup_steps": 256,
        "num_samples": 128,
        "thinning": 256,
    },
    error=ValueError,
    message="`num_samples` has to be larger than `thinning`.",
)

specs.add_invalid(
    models.FullyBayesianSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "features_to_warp": ["warped_feature"],
    },
    error=ValueError,
    message="Feature 'warped_feature' in features_to_warp is not a valid input key.",
)


specs.add_valid(
    models.MixedSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [CategoricalInput(key="cat1", categories=["a", "b", "c"])],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "aggregations": None,
        "continuous_kernel": MaternKernel(ard=True, nu=2.5).model_dump(),
        "categorical_kernel": HammingDistanceKernel(ard=True).model_dump(),
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
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
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ).model_dump(),
        "aggregations": None,
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig().model_dump(),
    },
)
specs.add_valid(
    models.RandomForestSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
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
    models.RegressionMLPEnsemble,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "aggregations": None,
        "n_estimators": 2,
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "final_activation": "identity",
        "dropout": 0.0,
        "batch_size": 10,
        "n_epochs": 200,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "subsample_fraction": 1.0,
        "shuffle": True,
        "scaler": ScalerEnum.IDENTITY,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": None,
    },
)
specs.add_invalid(
    models.RegressionMLPEnsemble,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(CategoricalOutput).obj(),
            ],
        ).model_dump(),
        "aggregations": None,
        "n_estimators": 2,
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "final_activation": "softmax",
        "dropout": 0.0,
        "batch_size": 10,
        "n_epochs": 200,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "subsample_fraction": 1.0,
        "shuffle": True,
        "scaler": ScalerEnum.IDENTITY,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": None,
    },
    error=ValueError,
)

specs.add_valid(
    models.ClassificationMLPEnsemble,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(CategoricalOutput).obj(),
            ],
        ).model_dump(),
        "aggregations": None,
        "n_estimators": 2,
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "final_activation": "softmax",
        "dropout": 0.0,
        "batch_size": 10,
        "n_epochs": 200,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "subsample_fraction": 1.0,
        "shuffle": True,
        "scaler": ScalerEnum.IDENTITY,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": None,
    },
)
specs.add_invalid(
    models.ClassificationMLPEnsemble,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "aggregations": None,
        "n_estimators": 2,
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "final_activation": "identity",
        "dropout": 0.0,
        "batch_size": 10,
        "n_epochs": 200,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "subsample_fraction": 1.0,
        "shuffle": True,
        "scaler": ScalerEnum.IDENTITY,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": None,
    },
    error=ValueError,
)

specs.add_valid(
    models.XGBoostSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
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
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=TanimotoKernel(
                ard=True,
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ).model_dump(),
        "aggregations": None,
        "scaler": ScalerEnum.IDENTITY,
        "output_scaler": ScalerEnum.IDENTITY,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "input_preprocessing_specs": {
            "mol1": Fingerprints(n_bits=32, bond_radius=3).model_dump(),
        },
        "dump": None,
        "hyperconfig": None,
    },
)

specs.add_valid(
    models.MixedTanimotoGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [MolecularInput(key="mol1")]
            + [CategoricalInput(key="cat1", categories=["a", "b", "c"])],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "aggregations": None,
        "molecular_kernel": TanimotoKernel(ard=True).model_dump(),
        "continuous_kernel": MaternKernel(
            ard=True,
            nu=random.choice([0.5, 1.5, 2.5]),
        ).model_dump(),
        "categorical_kernel": HammingDistanceKernel(ard=True).model_dump(),
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {
            "mol1": Fingerprints(n_bits=32, bond_radius=3).model_dump(),
            "cat1": CategoricalEncodingEnum.ONE_HOT,
        },
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "dump": None,
        "hyperconfig": None,
    },
)

specs.add_valid(
    models.CategoricalDeterministicSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                CategoricalInput(key="x_cat", categories=["a", "b", "c"]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                ContinuousOutput(key="y_cat"),
            ],
        ).model_dump(),
        "input_preprocessing_specs": {"x_cat": CategoricalEncodingEnum.ONE_HOT},
        "mapping": {"a": 0.1, "b": 0.2, "c": 1.0},
        "dump": None,
    },
)


specs.add_invalid(
    models.CategoricalDeterministicSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                CategoricalInput(key="x_cat", categories=["a", "b", "c"]),
                CategoricalInput(key="x_cat2", categories=["a", "b", "c"]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                ContinuousOutput(key="y_cat"),
            ],
        ).model_dump(),
        "input_preprocessing_specs": {"x_cat": CategoricalEncodingEnum.ONE_HOT},
        "mapping": {"a": 0.1, "b": 0.2, "c": 1.0},
        "dump": None,
    },
    error=ValueError,
    message="Only one input is supported for the `CategoricalDeterministicSurrogate`",
)

specs.add_invalid(
    models.CategoricalDeterministicSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                CategoricalInput(key="x_cat", categories=["a", "b", "c"]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                ContinuousOutput(key="y_cat"),
            ],
        ).model_dump(),
        "input_preprocessing_specs": {"x_cat": CategoricalEncodingEnum.ONE_HOT},
        "mapping": {"a": 0.1, "b": 0.2, "d": 1.0},
        "dump": None,
    },
    error=ValueError,
    message="Mapping keys do not match input feature keys.",
)

specs.add_valid(
    models.LinearDeterministicSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=(0, 1)),
                ContinuousInput(key="b", bounds=(0, 1)),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "intercept": 5.0,
        "coefficients": {"a": 2.0, "b": -3.0},
        "input_preprocessing_specs": {},
        "dump": None,
    },
)

specs.add_invalid(
    models.LinearDeterministicSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=(0, 1)),
                ContinuousInput(key="b", bounds=(0, 1)),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "intercept": 5.0,
        "coefficients": {"a": 2.0, "b": -3.0, "c": 5.0},
        "input_preprocessing_specs": {},
        "dump": None,
    },
    error=ValueError,
    message="coefficient keys do not match input feature keys.",
)

specs.add_invalid(
    models.LinearDeterministicSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=(0, 1)),
                CategoricalInput(key="b", categories=["a", "b"]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "intercept": 5.0,
        "coefficients": {"a": 2.0, "b": -3.0},
        "input_preprocessing_specs": {},
        "dump": None,
    },
    error=ValueError,
    message="Only numerical inputs are supported for the `LinearDeterministicSurrogate`",
)

specs.add_valid(
    models.MultiTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [TaskInput(key="task", categories=["a", "b", "c"])],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ).model_dump(),
        "aggregations": None,
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "task_prior": None,
        "input_preprocessing_specs": {
            "task": CategoricalEncodingEnum.ORDINAL,
        },
        "dump": None,
        "hyperconfig": MultiTaskGPHyperconfig().model_dump(),
    },
)

# if wrong encoding (one-hot) is used, there should be a validation error
specs.add_invalid(
    models.MultiTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [TaskInput(key="task", categories=["a", "b", "c"])],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ).model_dump(),
        "aggregations": None,
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "task_prior": None,
        "input_preprocessing_specs": {
            "task": CategoricalEncodingEnum.ONE_HOT,
        },
        "dump": None,
        "hyperconfig": MultiTaskGPHyperconfig().model_dump(),
    },
    error=ValueError,
)

# if there is no task input, there should be a validation error
specs.add_invalid(
    models.MultiTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=MaternKernel(
                ard=True, nu=2.5, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ).model_dump(),
        "aggregations": None,
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "task_prior": None,
        "input_preprocessing_specs": {
            "task": CategoricalEncodingEnum.ORDINAL,
        },
        "dump": None,
        "hyperconfig": MultiTaskGPHyperconfig().model_dump(),
    },
    error=ValueError,
)


specs.add_valid(
    models.PiecewiseLinearGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[ContinuousInput(key=f"phi_{i}", bounds=(0, 1)) for i in range(4)]
            + [ContinuousInput(key=f"t_{i+1}", bounds=(0, 1)) for i in range(2)]
            + [ContinuousInput(key=f"t_{3}", bounds=(2, 60))],
        ).model_dump(),
        "outputs": Outputs(features=[ContinuousOutput(key="alpha")]).model_dump(),
        "interpolation_range": [0, 1],
        "n_interpolation_points": 1000,
        "x_keys": ["t_1", "t_2"],
        "y_keys": [f"phi_{i}" for i in range(4)],
        "continuous_keys": ["t_3"],
        "prepend_x": [0.0],
        "append_x": [1.0],
        "prepend_y": [],
        "append_y": [],
        "normalize_y": 100.0,
        "shape_kernel": WassersteinKernel(
            squared=False,
            lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
        ).model_dump(),
        "continuous_kernel": MaternKernel(
            ard=True, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
        ).model_dump(),
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "outputscale_prior": THREESIX_SCALE_PRIOR().model_dump(),
        "dump": None,
        "aggregations": None,
        "input_preprocessing_specs": {},
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
        "hyperconfig": PiecewiseLinearGPSurrogateHyperconfig().model_dump(),
    },
)

specs.add_invalid(
    models.PiecewiseLinearGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[ContinuousInput(key=f"phi_{i}", bounds=(0, 1)) for i in range(4)]
            + [ContinuousInput(key=f"t_{i+1}", bounds=(0, 1)) for i in range(2)],
        ).model_dump(),
        "outputs": Outputs(features=[ContinuousOutput(key="alpha")]).model_dump(),
        "interpolation_range": (0, 1),
        "n_interpolation_points": 1000,
        "x_keys": ["t_1", "t_2"],
        "y_keys": [f"phi_{i}" for i in range(4)],
        "continuous_keys": [],
        "prepend_x": [0.0],
        "append_x": [1.0],
        "prepend_y": [],
        "append_y": [],
        "normalize_y": 100.0,
        "shape_kernel": WassersteinKernel(
            squared=False,
            lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
        ).model_dump(),
        "continuous_kernel": MaternKernel(
            ard=True, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
        ).model_dump(),
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "outputscale_prior": THREESIX_SCALE_PRIOR().model_dump(),
        "dump": None,
        "aggregations": None,
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
    },
    error=ValueError,
    message="Continuous kernel specified but no features for continuous kernel.",
)

specs.add_invalid(
    models.PiecewiseLinearGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[ContinuousInput(key=f"phi_{i}", bounds=(0, 1)) for i in range(4)]
            + [ContinuousInput(key=f"t_{i+1}", bounds=(0, 1)) for i in range(3)],
        ).model_dump(),
        "outputs": Outputs(features=[ContinuousOutput(key="alpha")]).model_dump(),
        "interpolation_range": (0, 1),
        "n_interpolation_points": 1000,
        "x_keys": [],
        "y_keys": [],
        "continuous_keys": ["t_1", "t_2", "t_3"] + [f"phi_{i}" for i in range(4)],
        "prepend_x": [0.0],
        "append_x": [1.0],
        "prepend_y": [],
        "append_y": [],
        "normalize_y": 100.0,
        "shape_kernel": WassersteinKernel(
            squared=False,
            lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
        ).model_dump(),
        "continuous_kernel": MaternKernel(
            ard=True, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
        ).model_dump(),
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "outputscale_prior": THREESIX_SCALE_PRIOR().model_dump(),
        "dump": None,
        "aggregations": None,
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
    },
    error=ValueError,
    message="No features for interpolation. Please provide `x_keys` and `y_keys`.",
)


specs.add_invalid(
    models.PiecewiseLinearGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[ContinuousInput(key=f"x_{i}", bounds=(0, 60)) for i in range(4)]
            + [ContinuousInput(key=f"y_{i}", bounds=(0, 1)) for i in range(4)],
        ).model_dump(),
        "outputs": Outputs(features=[ContinuousOutput(key="alpha")]).model_dump(),
        "interpolation_range": (0, 1),
        "n_interpolation_points": 400,
        "x_keys": [f"x_{i}" for i in range(3)],
        "y_keys": [f"y_{i}" for i in range(4)],
        "continuous_keys": ["x_3"],
        "prepend_x": [],
        "append_x": [],
        "prepend_y": [],
        "append_y": [],
        "normalize_y": 100,
        "shape_kernel": WassersteinKernel(
            squared=False,
            lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
        ).model_dump(),
        "continuous_kernel": MaternKernel(
            ard=True, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
        ).model_dump(),
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "outputscale_prior": THREESIX_SCALE_PRIOR().model_dump(),
        "dump": None,
        "aggregations": None,
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
    },
    error=ValueError,
    message="Different number of x and y values for interpolation.",
)

specs.add_invalid(
    models.PiecewiseLinearGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[ContinuousInput(key=f"x_{i}", bounds=(0, 60)) for i in range(4)]
            + [ContinuousInput(key=f"y_{i}", bounds=(0, 1)) for i in range(4)],
        ).model_dump(),
        "outputs": Outputs(features=[ContinuousOutput(key="alpha")]).model_dump(),
        "interpolation_range": (0, 1),
        "n_interpolation_points": 400,
        "x_keys": [f"x_{i}" for i in range(3)],
        "y_keys": [f"y_{i}" for i in range(4)],
        "continuous_keys": ["x_3", "dummy"],
        "prepend_x": [],
        "append_x": [60],
        "prepend_y": [],
        "append_y": [],
        "shape_kernel": WassersteinKernel(
            squared=False,
            lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
        ).model_dump(),
        "continuous_kernel": MaternKernel(
            ard=True, lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR()
        ).model_dump(),
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "outputscale_prior": THREESIX_SCALE_PRIOR().model_dump(),
        "dump": None,
        "aggregations": None,
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "scaler": ScalerEnum.NORMALIZE,
        "output_scaler": ScalerEnum.STANDARDIZE,
    },
    error=ValueError,
    message="Feature keys do not match input keys.",
)
