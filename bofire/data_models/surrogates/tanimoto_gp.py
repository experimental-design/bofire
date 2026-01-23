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
    pre_compute_similarities: bool = True

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
    def provide_info_for_pre_compute_similarities(self):
        if not self.pre_compute_similarities:
            return self

        # settings
        if isinstance(self.kernel, ScaleKernel):
            base_kernel = self.kernel.base_kernel
            if isinstance(base_kernel, TanimotoKernel):
                molecular_inputs: list[CategoricalMolecularInput] = self.inputs.get(
                    includes=CategoricalMolecularInput,
                    exact=False,
                )
                base_kernel._molecular_inputs = molecular_inputs.features

                # move fingerprint data model fro categorical encodings to kernel-specs
                base_kernel._fingerprint_settings_for_similarities = {}
                for inp_ in molecular_inputs:
                    if inp_.key in list(self.categorical_encodings):
                        assert isinstance(
                            self.categorical_encodings[inp_.key], Fingerprints
                        ), (
                            f"Categorical encoding for input {inp_.key} must be a Fingerprint. "
                            f"Found {type(self.categorical_encodings[inp_.key])}"
                        )
                        fingerprint: Fingerprints = self.categorical_encodings.pop(inp_.key)
                        base_kernel._fingerprint_settings_for_similarities[inp_.key] = fingerprint

                base_kernel._pre_compute_similarities = True

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
