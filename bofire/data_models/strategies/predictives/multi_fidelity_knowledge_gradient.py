from typing import Dict, Generator, Literal, Optional, Union

from pydantic import Field, model_validator

from bofire.data_models.acquisition_functions.acquisition_function import qMFHVKG
from bofire.data_models.acquisition_functions.cost_aware_utility import (
    CostAwareUtility,
    InverseCostWeightedUtility,
)
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.features.task import ContinuousTaskInput, TaskInput
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
from bofire.data_models.strategies.predictives.mobo import ExplicitReferencePoint
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)
from bofire.data_models.strategies.predictives.sobo import _ForbidPFMixin
from bofire.data_models.surrogates.api import BotorchSurrogates, SingleTaskGPSurrogate


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


class MultiFidelityHVKGStrategy(MultiobjectiveStrategy, _ForbidPFMixin):
    type: Literal["MultiFidelityHVKGStrategy"] = "MultiFidelityHVKGStrategy"  # type: ignore
    ref_point: Optional[Union[ExplicitReferencePoint, Dict[str, float]]] = None
    acquisition_function: qMFHVKG = Field(default_factory=lambda: qMFHVKG())
    cost_aware_utility: CostAwareUtility = Field(
        default_factory=lambda: InverseCostWeightedUtility()
    )

    @model_validator(mode="after")
    def validate_multitask_allowed(self):
        """Overwrites BotorchSurrogate.validate_multitask_allowed, as multiple tasks are allowed."""
        return self

    @model_validator(mode="after")
    def validate_surrogate_specs(self):
        """Ensures that a single-task multi-fidelity model is specified for each output feature"""
        MultiFidelityHVKGStrategy._generate_surrogate_specs(
            self.domain,
            self.surrogate_specs,
        )

        for m in self.surrogate_specs.surrogates:
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
                if not all(
                    isinstance(tf, ContinuousTaskInput) for tf in task_features.get()
                ):
                    raise ValueError(
                        "Fidelity kernel can only operate on task features."
                    )

        self.acquisition_optimizer.validate_surrogate_specs(self.surrogate_specs)

        return self

    @staticmethod
    def _generate_surrogate_specs(
        domain: Domain,
        surrogate_specs: BotorchSurrogates,
    ) -> BotorchSurrogates:
        """Method to generate multi-task model specifications when no model specs are passed
        As default specification, a 5/2 matern kernel with automated relevance detection and normalization of the input features is used.
        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            surrogate_specs (BotorchSurrogates, optional): List of model specification classes specifying the models to be used in the strategy. Defaults to None.
        Raises:
            KeyError: if there is a model spec for an unknown output feature
            KeyError: if a model spec has an unknown input feature
        Returns:
            BotorchSurrogates: List of model specification classes
        """
        existing_keys = surrogate_specs.outputs.get_keys()
        non_exisiting_keys = list(set(domain.outputs.get_keys()) - set(existing_keys))
        _surrogate_specs = surrogate_specs.surrogates

        task_keys = domain.inputs.get_keys(includes=TaskInput)
        non_task_keys = domain.inputs.get_keys(excludes=TaskInput)

        for output_feature in non_exisiting_keys:
            _surrogate_specs.append(
                SingleTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=Outputs(
                        features=[domain.outputs.get_by_key(output_feature)]
                    ),
                    kernel=ScaleKernel(
                        base_kernel=MultiplicativeKernel(
                            kernels=[
                                RBFKernel(
                                    features=non_task_keys,
                                    ard=True,
                                    lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                                ),
                                DownsamplingKernel(
                                    features=task_keys,
                                    offset_prior=THREESIX_LENGTHSCALE_PRIOR(),
                                    power_prior=THREESIX_LENGTHSCALE_PRIOR(),
                                ),
                            ]
                        )
                    ),
                )
            )
        surrogate_specs.surrogates = _surrogate_specs
        surrogate_specs._check_compability(inputs=domain.inputs, outputs=domain.outputs)
        return surrogate_specs
