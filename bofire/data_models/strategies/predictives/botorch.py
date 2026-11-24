from typing import Type

from pydantic import Field, model_validator

from bofire.data_models.constraints.api import Constraint, InterpointConstraint
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousInput,
)
from bofire.data_models.strategies.predictives.acqf_optimization import (
    AnyAcqfOptimizer,
    BotorchOptimizer,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.surrogates.api import (
    AnyBotorchSurrogate,
    BotorchSurrogates,
    MultiTaskGPSurrogate,
    SingleTaskGPSurrogate,
)


class BotorchStrategy(PredictiveStrategy):
    """Bayesian optimization with BoTorch: fits one surrogate per output and
    proposes the candidates that maximize an acquisition function.

    Outputs without a surrogate in `surrogate_specs` get a single-task GP, or the
    mixed GP if the domain has categorical inputs without descriptors.
    """

    acquisition_optimizer: AnyAcqfOptimizer = Field(
        default=BotorchOptimizer(),
        description="How the acquisition function is maximized over the domain. It "
        "also decides which constraints the strategy supports.",
    )
    surrogate_specs: BotorchSurrogates = Field(
        default=BotorchSurrogates(surrogates=[]),
        validate_default=True,
        description="Surrogates for some or all outputs; each output without one "
        "gets a default surrogate.",
    )
    include_infeasible_exps_in_acqf_calc: bool = Field(
        default=False,
        description="Whether infeasible experiments should be included in the set "
        "of experiments used to compute the acquisition function.",
    )

    @model_validator(mode="after")
    def validate_domain_for_optimizer(self):
        self.acquisition_optimizer.validate_domain(self.domain)
        return self

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy. For optimizer-specific
        strategies, this is passed to the optimizer check.

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise

        """
        return self.acquisition_optimizer.is_constraint_implemented(my_type)

    @model_validator(mode="after")
    def validate_interpoint_constraints(self):
        if self.domain.constraints.get(InterpointConstraint) and len(
            self.domain.inputs.get(ContinuousInput),
        ) != len(self.domain.inputs):
            raise ValueError(
                "Interpoint constraints can only be used for pure continuous search spaces.",
            )
        return self

    @model_validator(mode="after")
    def validate_surrogate_specs(self):
        """Ensures that a prediction model is specified for each output feature"""
        BotorchStrategy._generate_surrogate_specs(
            self.domain,
            self.surrogate_specs,
        )

        self.acquisition_optimizer.validate_surrogate_specs(self.surrogate_specs)

        return self

    @classmethod
    def _generate_surrogate_specs(
        cls,
        domain: Domain,
        surrogate_specs: BotorchSurrogates,
    ) -> BotorchSurrogates:
        """Adds a default surrogate, as built by
        `_generate_single_surrogate_spec_for_output`, for every output of the domain
        without one, in the order of the domain's outputs.

        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            surrogate_specs: The surrogates passed to the strategy; extended in place.

        Raises:
            ValueError: if the surrogates don't fit the domain's inputs and outputs.

        Returns:
            BotorchSurrogates: The extended surrogates.

        """
        existing_keys = surrogate_specs.outputs.get_keys()
        for output_feature in domain.outputs.get_keys():
            if output_feature not in existing_keys:
                surrogate_specs.surrogates.append(
                    cls._generate_single_surrogate_spec_for_output(
                        domain, output_feature
                    )
                )
        surrogate_specs._check_compability(inputs=domain.inputs, outputs=domain.outputs)
        return surrogate_specs

    @classmethod
    def _generate_single_surrogate_spec_for_output(
        cls, domain: Domain, output_feature: str
    ) -> AnyBotorchSurrogate:
        """Generate a single BoTorch surrogate if one is not specified for a given output feature.

        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            output_feature (str): The key of the target output feature.

        Returns:
            AnyBotorchSurrogate: Spec for the surrogate for the given output feature.
        """

        # A categorical needs the mixed GP only if it is *not* descriptor-encoded: with
        # descriptor data it turns into continuous columns a plain GP handles. Keyed on
        # the data the feature carries, not on its type.
        if any(
            feat.descriptors is None
            for feat in domain.inputs.get(CategoricalInput, exact=False)
            if not isinstance(feat, CategoricalTaskInput)
        ):
            return SingleTaskGPSurrogate.mixed(
                domain.inputs,
                Outputs(features=[domain.outputs.get_by_key(output_feature)]),
            )

        return SingleTaskGPSurrogate(
            inputs=domain.inputs,
            outputs=Outputs(
                features=[
                    domain.outputs.get_by_key(output_feature),
                ],
            ),
        )

    @model_validator(mode="after")
    def validate_multitask_allowed(self):
        """Ensures that if a multitask model is used there is only a single allowed task category"""
        if any(
            isinstance(m, MultiTaskGPSurrogate) for m in self.surrogate_specs.surrogates
        ):
            # find the task input
            task_input = self.domain.inputs.get(CategoricalTaskInput, exact=True)
            # check if there is only one allowed task category
            assert (
                sum(task_input.features[0].allowed) == 1
            ), "Exactly one allowed task category must be specified for strategies with MultiTask models."
        return self
