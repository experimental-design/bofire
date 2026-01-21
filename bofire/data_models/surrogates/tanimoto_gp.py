from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalMolecularInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import AnyKernel, ScaleKernel
from bofire.data_models.kernels.molecular import TanimotoKernel
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
)
from bofire.data_models.priors.api import (
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyPrior,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class TanimotoGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["TanimotoGPSurrogate"] = "TanimotoGPSurrogate"
    pre_compute_similarities: bool = False
    fingerprint_settings_for_similarities: Fingerprints = Fingerprints()

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=TanimotoKernel(
                ard=True,
            ),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: THREESIX_NOISE_PRIOR())
    scaler: ScalerEnum = ScalerEnum.IDENTITY

    @model_validator(mode="after")
    def check_and_pre_compute_similarities(self):
        if not self.pre_compute_similarities:
            return self

        # settings
        if isinstance(self.kernel, ScaleKernel):
            base_kernel = self.kernel.base_kernel
            if isinstance(base_kernel, TanimotoKernel):
                molecular_inputs = self.inputs.get(
                    includes=CategoricalMolecularInput,
                    exact=False,
                )
                for inp_ in molecular_inputs:
                    if inp_.key in list(self.categorical_encodings):
                        self.categorical_encodings.pop(
                            inp_.key
                        )  # remove categorical encodings

                base_kernel._molecular_inputs = molecular_inputs
                base_kernel._fingerprint_settings_for_similarities = (
                    self.fingerprint_settings_for_similarities
                )
                base_kernel.pre_compute_similarities = (
                    True  # this triggers computation in the kernel data-model
                )

                return self

        raise NotImplementedError(
            "no supperted kernel-architecture for pre-computed tanimoto similarities"
        )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    @model_validator(mode="after")
    def validate_moleculars(self):
        """Checks that at least one of fingerprints, fragments, or fingerprintsfragments features are present."""
        if not self.pre_compute_similarities:
            if not any(
                isinstance(value, Fingerprints)
                or isinstance(value, Fragments)
                or isinstance(value, FingerprintsFragments)
                for value in self.categorical_encodings.values()
            ):
                raise ValueError(
                    "TanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
                )
        return self
