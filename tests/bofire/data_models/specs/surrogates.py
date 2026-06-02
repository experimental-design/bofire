import bofire.data_models.surrogates.api as models
from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalMolecularInput,
    CategoricalOutput,
    CategoricalTaskInput,
    CloneFeature,
    ContinuousInput,
    ContinuousOutput,
    MeanFeature,
    SumFeature,
)
from bofire.data_models.kernels.api import (
    HammingDistanceKernel,
    InfiniteWidthBNNKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
)
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.priors.api import (
    PAIRWISEGP_LENGTHSCALE_CONSTRAINT,
    PAIRWISEGP_LENGTHSCALE_PRIOR,
    PAIRWISEGP_OUTPUTSCALE_CONSTRAINT,
    PAIRWISEGP_OUTPUTSCALE_PRIOR,
    ROBUSTGP_LENGTHSCALE_CONSTRAINT,
    ROBUSTGP_OUTPUTSCALE_CONSTRAINT,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    GreaterThan,
)
from bofire.data_models.surrogates.api import Normalize, ScalerEnum
from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPHyperconfig
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
        "engineered_features": EngineeredFeatures(
            features=[MeanFeature(key="mean1", features=["a", "b"])]
        ).model_dump(),
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig().model_dump(),
    },
)

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
                ard=True,
                nu=2.5,
                lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR(),
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ).model_dump(),
        "engineered_features": EngineeredFeatures(
            features=[CloneFeature(key="__clone_continuous__", features=["a"])]
        ).model_dump(),
        "scaler": Normalize(features=["__clone_continuous__"]).model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": None,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
        "engineered_features": EngineeredFeatures(
            features=[SumFeature(key="sum1", features=["a", "b"])]
        ).model_dump(),
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "cache_model_trace": False,
        "convex_parametrization": True,
        "prior_mean_of_support": None,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
        "scaler": Normalize().model_dump(),
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

specs.add_invalid(
    models.SingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=[0, 1]),
                ContinuousInput(key="b", bounds=[0, 1]),
            ],
        ),
        "outputs": Outputs(
            features=[
                ContinuousOutput(key="a"),
            ],
        ),
        "scaler": Normalize(features=["d", "e"]),
    },
    error=ValueError,
    message="The following features are missing in inputs",
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
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "engineered_features": EngineeredFeatures().model_dump(),
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "n_taus": 4,
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "hyperconfig": None,
        "dump": None,
    },
)

specs.add_valid(
    models.EnsembleMapSaasSingleTaskGPSurrogate,
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "n_taus": 4,
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "hyperconfig": None,
        "dump": None,
    },
)

specs.add_invalid(
    models.EnsembleMapSaasSingleTaskGPSurrogate,
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "n_taus": 4,
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.LOG,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "hyperconfig": None,
        "dump": None,
    },
    error=ValueError,
    message="LOG and CHAINED_LOG_STANDARDIZE are not supported as output transforms for EnsembleMapSaasSingleTaskGPSurrogate.",
)

specs.add_invalid(
    models.EnsembleMapSaasSingleTaskGPSurrogate,
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "n_taus": 4,
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.CHAINED_LOG_STANDARDIZE,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "hyperconfig": None,
        "dump": None,
    },
    error=ValueError,
    message="LOG and CHAINED_LOG_STANDARDIZE are not supported as output transforms for EnsembleMapSaasSingleTaskGPSurrogate.",
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
                ContinuousInput(key="num1", bounds=[0, 1]),
            ]
            + [CategoricalInput(key="cat1", categories=["a", "b", "c"])],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "engineered_features": EngineeredFeatures().model_dump(),
        "continuous_kernel": MaternKernel(
            ard=True, nu=2.5, features=["num1"]
        ).model_dump(),
        "categorical_kernel": HammingDistanceKernel(
            ard=True, features=["cat1"]
        ).model_dump(),
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "input_preprocessing_specs": {"cat1": CategoricalEncodingEnum.ORDINAL},
        "categorical_encodings": {"cat1": CategoricalEncodingEnum.ORDINAL},
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "dump": None,
        "hyperconfig": SingleTaskGPHyperconfig().model_dump(),
    },
)
specs.add_invalid(
    models.MixedSingleTaskGPSurrogate,
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
    },
    error=ValueError,
    message="MixedSingleTaskGPSurrogate can only be used if at least one categorical feature is present.",
)

specs.add_invalid(
    models.MixedSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                CategoricalInput(key="x_cat", categories=["a", "b", "c"]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "categorical_encodings": {"x_cat": CategoricalEncodingEnum.ONE_HOT},
    },
    error=ValueError,
    message="MixedSingleTaskGPSurrogate can only be used if at least one categorical feature is ordinal encoded.",
)

specs.add_invalid(
    models.MixedSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="x_cont", bounds=[0, 1]),
                CategoricalInput(key="x_cat", categories=["a", "b", "c"]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "continuous_kernel": MaternKernel(nu=2.5, features=["x_cat"]).model_dump(),
    },
    error=ValueError,
    message="The features defined in",
)

specs.add_invalid(
    models.MixedSingleTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="x_cont", bounds=[0, 1]),
                CategoricalInput(key="x_cat", categories=["a", "b", "c"]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(),
            ],
        ).model_dump(),
        "categorical_kernel": HammingDistanceKernel(
            ard=True, features=["x_cont"]
        ).model_dump(),
    },
    error=ValueError,
    message="The features defined in the categorical",
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
        "scaler": None,
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
        "engineered_features": EngineeredFeatures().model_dump(),
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
        "scaler": None,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
        "scaler": None,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
        "engineered_features": EngineeredFeatures().model_dump(),
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
        "scaler": None,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
        "scaler": None,
        "output_scaler": ScalerEnum.IDENTITY,
        "input_preprocessing_specs": {},
        "dump": None,
        "hyperconfig": None,
    },
    error=ValueError,
)

specs.add_valid(
    models.TanimotoGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                CategoricalMolecularInput(key="mol1", categories=["C", "CC", "CCC"]),
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "scaler": None,
        "output_scaler": ScalerEnum.IDENTITY,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "input_preprocessing_specs": {"mol1": CategoricalEncodingEnum.ORDINAL},
        "categorical_encodings": {
            "mol1": Fingerprints(n_bits=32, bond_radius=3).model_dump(),
        },
        "dump": None,
        "hyperconfig": None,
        "tanimoto_calculation_mode": "pre_computed",
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
        "input_preprocessing_specs": {"x_cat": CategoricalEncodingEnum.ORDINAL},
        "categorical_encodings": {"x_cat": CategoricalEncodingEnum.ORDINAL},
        "engineered_features": EngineeredFeatures().model_dump(),
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
        "input_preprocessing_specs": {"x_cat": CategoricalEncodingEnum.ORDINAL},
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
        "input_preprocessing_specs": {"x_cat": CategoricalEncodingEnum.ORDINAL},
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "intercept": 5.0,
        "coefficients": {"a": 2.0, "b": -3.0},
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
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
            + [CategoricalTaskInput(key="task", categories=["a", "b", "c"])],
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
        "engineered_features": EngineeredFeatures().model_dump(),
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "task_prior": None,
        "input_preprocessing_specs": {
            "task": CategoricalEncodingEnum.ORDINAL,
        },
        "categorical_encodings": {
            "task": CategoricalEncodingEnum.ORDINAL,
        },
        "dump": None,
        "hyperconfig": MultiTaskGPHyperconfig().model_dump(),
    },
)

specs.add_invalid(
    models.MultiTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [CategoricalTaskInput(key="task", categories=["a", "b", "c"])],
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
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
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
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "task_prior": None,
        "input_preprocessing_specs": {
            "task": CategoricalEncodingEnum.ORDINAL,
        },
        "dump": None,
        "hyperconfig": MultiTaskGPHyperconfig().model_dump(),
    },
    error=ValueError,
    message="Exactly one task input",
)

specs.add_invalid(
    models.MultiTaskGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(),
            ]
            + [CategoricalTaskInput(key="task", categories=["a", "b", "c"])],
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
        "scaler": Normalize().model_dump(),
        "output_scaler": ScalerEnum.STANDARDIZE,
        "noise_prior": THREESIX_NOISE_PRIOR().model_dump(),
        "noise_constraint": GreaterThan(lower_bound=1e-4).model_dump(),
        "task_prior": None,
        "input_preprocessing_specs": {
            "task": CategoricalEncodingEnum.ORDINAL,
        },
        "categorical_encodings": {
            "task": CategoricalEncodingEnum.ONE_HOT,
        },
        "dump": None,
        "hyperconfig": MultiTaskGPHyperconfig().model_dump(),
    },
    error=ValueError,
    message="The task feature task has to be encoded as ordinal",
)
specs.add_valid(
    models.PairwiseGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=[0, 1]),
                ContinuousInput(key="b", bounds=[0, 1]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[features.valid(ContinuousOutput).obj()],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=RBFKernel(
                ard=True,
                lengthscale_prior=PAIRWISEGP_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=PAIRWISEGP_LENGTHSCALE_CONSTRAINT(),
            ),
            outputscale_prior=PAIRWISEGP_OUTPUTSCALE_PRIOR(),
            outputscale_constraint=PAIRWISEGP_OUTPUTSCALE_CONSTRAINT(),
        ).model_dump(),
        "scaler": Normalize().model_dump(),
        "likelihood": "probit",
        "engineered_features": EngineeredFeatures().model_dump(),
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "dump": None,
    },
)

specs.add_valid(
    models.PairwiseGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[
                ContinuousInput(key="a", bounds=[0, 1]),
                ContinuousInput(key="b", bounds=[0, 1]),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[features.valid(ContinuousOutput).obj()],
        ).model_dump(),
        "kernel": ScaleKernel(
            base_kernel=RBFKernel(
                ard=True,
                lengthscale_prior=PAIRWISEGP_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=PAIRWISEGP_LENGTHSCALE_CONSTRAINT(),
            ),
            outputscale_prior=PAIRWISEGP_OUTPUTSCALE_PRIOR(),
            outputscale_constraint=PAIRWISEGP_OUTPUTSCALE_CONSTRAINT(),
        ).model_dump(),
        "scaler": Normalize().model_dump(),
        "likelihood": "logit",
        "engineered_features": EngineeredFeatures().model_dump(),
        "hyperconfig": None,
        "input_preprocessing_specs": {},
        "categorical_encodings": {},
        "dump": None,
    },
)

specs.add_invalid(
    models.PairwiseGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[ContinuousInput(key="a", bounds=[0, 1])],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                ContinuousOutput(key="y1"),
                ContinuousOutput(key="y2"),
            ],
        ).model_dump(),
    },
    error=ValueError,
    message="PairwiseGPSurrogate supports exactly one output",
)

specs.add_invalid(
    models.PairwiseGPSurrogate,
    lambda: {
        "inputs": Inputs(
            features=[ContinuousInput(key="a", bounds=[0, 1])],
        ).model_dump(),
        "outputs": Outputs(
            features=[features.valid(ContinuousOutput).obj()],
        ).model_dump(),
        "kernel": RBFKernel(ard=True).model_dump(),
    },
    error=ValueError,
    message="PairwiseGPSurrogate.kernel must be a ScaleKernel",
)
