from typing import Generator, Literal, Type

from pydantic import Field, field_validator, model_validator

from bofire.data_models.acquisition_functions.acquisition_function import qMFHVKG
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.features.api import CategoricalOutput, ContinuousOutput, Feature
from bofire.data_models.features.task import (
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousTaskInput,
    TaskInput,
)
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    AnyKernel,
    DownsamplingKernel,
    FidelityKernel,
    MultiplicativeKernel,
    PolynomialFeatureInteractionKernel,
    RBFKernel,
    ScaleKernel,
)
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
)
from bofire.data_models.strategies.predictives.mobo import MoboStrategy
from bofire.data_models.surrogates.api import AnyBotorchSurrogate, SingleTaskGPSurrogate
from bofire.data_models.surrogates.deterministic import LinearDeterministicSurrogate


def _traverse_kernels(kernel: AnyKernel) -> Generator[AnyKernel, None, None]:
    yield kernel
    if isinstance(kernel, ScaleKernel):
        yield from _traverse_kernels(kernel.base_kernel)
    if isinstance(
        kernel,
        (AdditiveKernel, MultiplicativeKernel, PolynomialFeatureInteractionKernel),
    ):
        for base_kernel in kernel.kernels:
            yield from _traverse_kernels(base_kernel)


class MultiFidelityHVKGStrategy(MoboStrategy):
    type: Literal["MultiFidelityHVKGStrategy"] = "MultiFidelityHVKGStrategy"
    acquisition_function: qMFHVKG = Field(default_factory=lambda: qMFHVKG())
    fidelity_cost_model_spec: LinearDeterministicSurrogate | None = None

    @field_validator("domain", mode="after")
    @classmethod
    def validate_domain_has_continuous_task_input(cls, domain: Domain) -> Domain:
        task_inputs = domain.inputs.get(includes=TaskInput)
        cat_task_inputs = domain.inputs.get(includes=CategoricalTaskInput)
        if len(task_inputs) == 0:
            raise ValueError("Must provide at least one fidelity.")

        if len(cat_task_inputs) > 0:
            raise ValueError("MFHVKG only supports continuous fidelities.")

        return domain

    @model_validator(mode="after")
    def validate_fidelity_cost_model_spec(self):
        if self.fidelity_cost_model_spec is None:  # required to prevent RecursionError
            self.fidelity_cost_model_spec = (
                MultiFidelityHVKGStrategy._generate_fidelity_cost_model_spec(
                    self.domain, self.fidelity_cost_model_spec
                )
            )
        fidelity_inputs = self.fidelity_cost_model_spec.inputs
        if len(fidelity_inputs.get(ContinuousTaskInput)) != len(fidelity_inputs):
            raise ValueError("Some inputs to the cost model are not fidelities.")

        if (
            self.domain.inputs.get_keys(ContinuousTaskInput)
            != fidelity_inputs.get_keys()
        ):
            raise ValueError("All fidelities should be included in the cost model.")

        return self

    @staticmethod
    def _generate_fidelity_cost_model_spec(
        domain: Domain,
        fidelity_cost_model_spec: LinearDeterministicSurrogate | None,
    ) -> LinearDeterministicSurrogate:
        if fidelity_cost_model_spec is not None:
            return fidelity_cost_model_spec

        fidelity_inputs = domain.inputs.get(TaskInput)
        fidelity_outputs = Outputs(
            features=[ContinuousOutput(key="fidelity cost", objective=None)]
        )
        coefficients = {key: 1.0 for key in fidelity_inputs.get_keys()}

        return LinearDeterministicSurrogate(
            inputs=fidelity_inputs,
            outputs=fidelity_outputs,
            coefficients=coefficients,
            intercept=1.0,
        )

    @model_validator(mode="after")
    def validate_surrogate_specs(self):
        """Ensures that a single-task multi-fidelity model is specified for each output feature"""
        MultiFidelityHVKGStrategy._generate_surrogate_specs(
            self.domain,
            self.surrogate_specs,
        )

        def _validate_surrogate(m: AnyBotorchSurrogate):
            if not isinstance(m, SingleTaskGPSurrogate):
                raise ValueError(f"Must use a SingleTaskGPSurrogate with {self.type}.")

            all_fidelity_kernels = [
                k for k in _traverse_kernels(m.kernel) if isinstance(k, FidelityKernel)
            ]
            if not all_fidelity_kernels:
                raise ValueError(
                    f"Must provide at least one fidelity kernel when using {self.type}."
                )

            for kernel in all_fidelity_kernels:
                assert kernel.features is not None
                task_features = self.domain.inputs.get_by_keys(kernel.features)
                if not all(isinstance(tf, TaskInput) for tf in task_features.get()):
                    raise ValueError(
                        "Fidelity kernel can only operate on task features."
                    )

        for m in self.surrogate_specs.surrogates:
            _validate_surrogate(m)

        self.acquisition_optimizer.validate_surrogate_specs(self.surrogate_specs)

        return self

    @classmethod
    def _generate_single_surrogate_spec_for_output(
        cls, domain: Domain, output_feature: str
    ) -> SingleTaskGPSurrogate:
        """Generate a SingleTaskGPSurrogate surrogate if one is not specified for a given output feature.
        The kernel uses a DownsamplingKernel for the fidelity features.

        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            output_feature (str): The key of the target output feature.

        Returns:
            SingleTaskGPSurrogate: Spec for the surrogate for the given output feature.
        """
        task_keys = domain.inputs.get_keys(includes=TaskInput)
        non_task_keys = domain.inputs.get_keys(excludes=TaskInput)

        surrogate_kernel = MultiplicativeKernel(
            kernels=[
                RBFKernel(
                    features=non_task_keys,
                    ard=True,
                    lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                )
            ]
            + [
                DownsamplingKernel(
                    features=[task_key],
                    offset_prior=THREESIX_LENGTHSCALE_PRIOR(),
                    power_prior=THREESIX_LENGTHSCALE_PRIOR(),
                )
                for task_key in task_keys
            ]
        )

        return SingleTaskGPSurrogate(
            inputs=domain.inputs,
            outputs=Outputs(features=[domain.outputs.get_by_key(output_feature)]),
            kernel=surrogate_kernel,
        )

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise

        """
        return my_type not in [CategoricalOutput, CategoricalInput]
