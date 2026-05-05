from typing import List, Literal, Union

from pydantic import model_validator

from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.features.api import CategoricalTaskInput
from bofire.data_models.strategies.predictives.sobo import SoboStrategy, _ForbidPFMixin
from bofire.data_models.surrogates.api import MultiTaskGPSurrogate


class MultiFidelityVarianceBasedStrategy(SoboStrategy, _ForbidPFMixin):
    type: Literal["MultiFidelityVarianceBasedStrategy"] = (
        "MultiFidelityVarianceBasedStrategy"
    )

    fidelity_thresholds: Union[List[float], float] = 0.1

    @model_validator(mode="after")
    def validate_tasks_and_fidelity_thresholds(self):
        """Ensures that there is one threshold per fidelity"""
        task_input, *_ = self.domain.inputs.get(
            includes=CategoricalTaskInput, exact=True
        )
        num_tasks = len(task_input.categories)  # ty: ignore[possibly-missing-attribute]

        if (
            isinstance(self.fidelity_thresholds, list)
            and len(self.fidelity_thresholds) != num_tasks
        ):
            raise ValueError(
                f"The number of tasks should be equal to the number of fidelity thresholds (got {num_tasks} tasks, {len(self.fidelity_thresholds)} thresholds)."
            )

        return self

    @model_validator(mode="after")
    def validate_only_one_target_fidelity(self):
        """Ensures that there is only one target fidelity (task where fidelity==0)."""
        task_input, *_ = self.domain.inputs.get(
            includes=CategoricalTaskInput, exact=True
        )
        num_target = sum(
            t == 0
            for t in task_input.fidelities  # ty: ignore[unresolved-attribute]
        )
        if num_target != 1:
            raise ValueError(
                f"Only one task can be the target fidelity (got {num_target})."
            )

        return self

    @model_validator(mode="after")
    def validate_multitask_allowed(self):
        """Overwrites BotorchSurrogate.validate_multitask_allowed, as multiple tasks are allowed."""
        return self

    @model_validator(mode="after")
    def validate_surrogate_specs(self):
        """Ensures that a multi-task model is specified for each output feature"""
        MultiFidelityVarianceBasedStrategy._generate_surrogate_specs(
            self.domain,
            self.surrogate_specs,
        )

        if not all(
            isinstance(m, MultiTaskGPSurrogate) for m in self.surrogate_specs.surrogates
        ):
            raise ValueError(f"Must use a MultiTaskGPSurrogate with {self.type}.")

        self.acquisition_optimizer.validate_surrogate_specs(self.surrogate_specs)

        return self

    @classmethod
    def _generate_single_surrogate_spec_for_output(
        cls, domain: Domain, output_feature: str
    ) -> MultiTaskGPSurrogate:
        """Generate a single MultiTask surrogate if one is not specified for a given output feature.

        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            output_feature (str): The key of the target output feature.

        Returns:
            MultiTaskGPSurrogate: Spec for the surrogate for the given output feature.
        """
        return MultiTaskGPSurrogate(
            inputs=domain.inputs,
            outputs=Outputs(features=[domain.outputs.get_by_key(output_feature)]),
        )
