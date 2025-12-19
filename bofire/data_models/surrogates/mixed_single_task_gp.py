from typing import Literal, Optional, Type

import pandas as pd
from pydantic import Field, model_serializer, model_validator

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum, RegressionMetricsEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    ContinuousOutput,
    TaskInput,
)
from bofire.data_models.kernels.api import (
    AnyCategoricalKernel,
    AnyContinuousKernel,
    HammingDistanceKernel,
    IndexKernel,
    MaternKernel,
    PositiveIndexKernel,
    RBFKernel,
)
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    MBO_LENGTHSCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyPrior,
    GreaterThan,
)
from bofire.data_models.surrogates.trainable import Hyperconfig
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MixedSingleTaskGPHyperconfig(Hyperconfig):
    type: Literal["MixedSingleTaskGPHyperconfig"] = "MixedSingleTaskGPHyperconfig"
    inputs: Inputs = Inputs(
        features=[
            CategoricalInput(
                key="continuous_kernel",
                categories=["rbf", "matern_1.5", "matern_2.5"],
            ),
            CategoricalInput(key="prior", categories=["mbo", "threesix", "hvarfner"]),
            CategoricalInput(key="ard", categories=["True", "False"]),
        ],
    )
    target_metric: RegressionMetricsEnum = RegressionMetricsEnum.MAE
    hyperstrategy: Literal[
        "FractionalFactorialStrategy", "SoboStrategy", "RandomStrategy"
    ] = "FractionalFactorialStrategy"

    @staticmethod
    def _update_hyperparameters(
        surrogate_data: "MixedSingleTaskGPSurrogate",
        hyperparameters: pd.Series,
    ):
        if hyperparameters.prior == "mbo":
            noise_prior, lengthscale_prior, _ = (
                MBO_NOISE_PRIOR(),
                MBO_LENGTHSCALE_PRIOR(),
                MBO_OUTPUTSCALE_PRIOR(),
            )
        elif hyperparameters.prior == "threesix":
            noise_prior, lengthscale_prior, _ = (
                THREESIX_NOISE_PRIOR(),
                THREESIX_LENGTHSCALE_PRIOR(),
                THREESIX_SCALE_PRIOR(),
            )
        else:
            noise_prior, lengthscale_prior = (
                HVARFNER_NOISE_PRIOR(),
                HVARFNER_LENGTHSCALE_PRIOR(),
            )

        surrogate_data.noise_prior = noise_prior
        if hyperparameters.continuous_kernel == "rbf":
            surrogate_data.continuous_kernel = RBFKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
            )

        elif hyperparameters.continuous_kernel == "matern_2.5":
            surrogate_data.continuous_kernel = MaternKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                nu=2.5,
            )

        elif hyperparameters.continuous_kernel == "matern_1.5":
            surrogate_data.continuous_kernel = MaternKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                nu=1.5,
            )

        else:
            raise ValueError(f"Kernel {hyperparameters.kernel} not known.")


class MixedSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """
    A Gaussian Process surrogate model for mixed (continuous and categorical) single-task optimization problems.
    This surrogate combines continuous and categorical kernels to handle optimization problems with
    both continuous variables and categorical features. It requires at least one categorical input
    and supports ordinal encoding for categorical variables.
    Attributes:
        type: Literal identifier for the surrogate type.
        continuous_kernel: Kernel for continuous features, defaults to RBF kernel with ARD.
        categorical_kernel: Kernel for categorical features, defaults to Hamming distance kernel.
        kernel_dict: Optional dictionary mapping feature keys to specific kernels. Useful for custom kernel assignments like different kernels for different different features.
        noise_prior: Prior distribution for observation noise.
        hyperconfig: Configuration for hyperparameter optimization.
    Requirements:
        - At least one categorical input feature must be present
        - At least one categorical feature must use ordinal encoding
        - Only supports continuous output variables
    The model automatically assigns appropriate kernels to features based on their types:
    - Continuous features use the continuous_kernel
    - Categorical features with ordinal encoding use the categorical_kernel
    - Custom kernel assignments can be specified via kernel_dict
    Raises:
        ValueError: If no categorical features are present, if no ordinal-encoded categorical
                   features exist, or if kernel configurations don't match input features.
    """  # noqa: E501

    type: Literal["MixedSingleTaskGPSurrogate"] = "MixedSingleTaskGPSurrogate"
    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
            lengthscale_constraint=GreaterThan(lower_bound=2.500e-02),
        )
    )
    categorical_kernel: AnyCategoricalKernel = Field(
        default_factory=lambda: HammingDistanceKernel(
            ard=True, lengthscale_constraint=GreaterThan(lower_bound=1.000e-06)
        ),
    )
    kernel_dict: Optional[dict] = None
    noise_prior: AnyPrior = Field(default_factory=lambda: HVARFNER_NOISE_PRIOR())
    hyperconfig: Optional[MixedSingleTaskGPHyperconfig] = Field(
        default_factory=lambda: MixedSingleTaskGPHyperconfig(),
    )

    @classmethod
    def _default_categorical_encodings(
        cls,
    ) -> dict[Type[CategoricalInput], CategoricalEncodingEnum | Fingerprints]:
        return {
            CategoricalInput: CategoricalEncodingEnum.ORDINAL,
            CategoricalMolecularInput: Fingerprints(),
            CategoricalDescriptorInput: CategoricalEncodingEnum.DESCRIPTOR,
            TaskInput: CategoricalEncodingEnum.ORDINAL,
        }

    @model_serializer(mode="wrap", when_used="always")
    def _serialize_model(self, serializer, info):
        data = serializer(self)
        # Remove kernel_dict if it's None
        if "kernel_dict" in data and data["kernel_dict"] is None:
            del data["kernel_dict"]
        return data

    @model_validator(mode="after")
    def validate_categoricals(self):
        # check that at least one categorical is present
        if (
            len(categoricals := self.inputs.get_keys(CategoricalInput, exact=False))
            == 0
        ):
            raise ValueError(
                "MixedSingleTaskGPSurrogate can only be used if at least one categorical feature is present.",
            )
        # check that a least one of the categorical features is ordinal or not encoded
        if not any(
            self.categorical_encodings.get(cat, CategoricalEncodingEnum.ORDINAL)
            == CategoricalEncodingEnum.ORDINAL
            for cat in categoricals
        ):
            raise ValueError(
                "MixedSingleTaskGPSurrogate can only be used if at least one categorical feature is ordinal encoded.",
            )
        # now we validate the kernels and the features being present there
        categorical_feature_keys = [
            cat
            for cat in categoricals
            if self.categorical_encodings.get(cat, CategoricalEncodingEnum.ORDINAL)
            == CategoricalEncodingEnum.ORDINAL
        ]
        ordinal_feature_keys = list(
            set(self.inputs.get_keys()) - set(categorical_feature_keys)
        )
        if len(ordinal_feature_keys) > 0:
            # check that feature keys are set correctly in kernels
            if self.continuous_kernel.features is None:
                self.continuous_kernel.features = ordinal_feature_keys
            else:
                if set(self.continuous_kernel.features) != set(ordinal_feature_keys):
                    raise ValueError(
                        "The features defined in the continuous kernel do not match the ordinal (encoded) features in the inputs.",
                    )
        else:
            self.continuous_kernel.features = []
        if self.categorical_kernel.features is None:
            self.categorical_kernel.features = categorical_feature_keys
        else:
            if set(self.categorical_kernel.features) != set(categorical_feature_keys):
                raise ValueError(
                    "The features defined in the categorical kernel do not match the categorical features in the inputs.",
                )
        return self

    @model_validator(mode="after")
    def validate_kernel_dict(self):
        if self.kernel_dict is not None:
            # check if the length of the kernel_dict matches the number of categorical features + continuous features
            num_all_features = len(self.inputs.get_keys())
            # num_categorical_features = len(
            #     self.inputs.get_keys(CategoricalInput, exact=False)
            # )
            # num_continuous_features = num_all_features - num_categorical_features
            assert (
                len(self.kernel_dict) == num_all_features
            ), "The length of kernel_dict must match the number of all features (categorical + continuous)."
            # check if the categorical feature is mapped to a categorical kernel and continuous to continuous kernel
            for feature_key, kernel in self.kernel_dict.items():
                if feature_key not in self.inputs.get_keys():
                    raise ValueError(
                        f"The feature '{feature_key}' in kernel_dict is not present in the inputs."
                    )
                if feature_key in self.inputs.get_keys(CategoricalInput, exact=False):
                    if not isinstance(kernel, AnyCategoricalKernel):
                        raise ValueError(
                            f"The feature '{feature_key}' is categorical and must be mapped to a categorical kernel."
                        )
                    # if the kernel is IndexKernel then check if num_categories matches
                    if isinstance(kernel, IndexKernel) or isinstance(
                        kernel, PositiveIndexKernel
                    ):
                        num_categories = len(
                            self.inputs.get_by_key(feature_key).categories  # type: ignore
                        )
                        if kernel.num_categories != num_categories:
                            raise ValueError(
                                f"The num_categories of the IndexKernel for feature '{feature_key}' does not match the number of categories in the input."
                            )
                else:
                    if not isinstance(kernel, AnyContinuousKernel):
                        raise ValueError(
                            f"The feature '{feature_key}' is continuous and must be mapped to a continuous kernel."
                        )
        return self

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
