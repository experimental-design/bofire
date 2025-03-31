import warnings
from abc import abstractmethod
from typing import Literal, Optional, Type

from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints import api as constraints
from bofire.data_models.constraints.api import InterpointConstraint
from bofire.data_models.domain.domain import Domain
from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import CategoricalDescriptorInput, ContinuousInput
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    MixedSingleTaskGPSurrogate,
)
from bofire.data_models.types import IntPowerOfTwo


class AcquisitionOptimizer(BaseModel):
    prefer_exhaustive_search_for_purely_categorical_domains: bool = True

    @abstractmethod
    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        """Checks if a constraint is implemented.

        Args:
            my_type (Type[Feature]): The type of the constraint.

        Returns:
            bool: True if the constraint is implemented, False otherwise.

        """
        pass

    @abstractmethod
    def validate_domain(self, domain: Domain):
        """Validates the fit of the domain to the optimizer.

        Args:
            domain (Domain): The domain to be validated.
        """
        pass

    @abstractmethod
    def validate_surrogate_specs(self, surrogate_specs: BotorchSurrogates):
        """Validates the surrogate specs for the optimizer.

        Args:
            surrogate_specs: The surrogate specs to be validated.

        Returns:
            SurrogateSpecs: The validated surrogate specs.

        """
        pass


class LocalSearchConfig(BaseModel):
    """LocalSearchConfigs provide a way to define how to switch between global
    acqf optimization in the global bounds and local acqf optimization in the local
    reference bounds.
    """

    @abstractmethod
    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        """Abstract switching function between local and global acqf optimum.

        Args:
            acqf_local (float): Local acqf value.
            acqf_global (float): Global acqf value.

        Returns:
            bool: If true, do local step, else a step towards the global acqf maximum.

        """


class LSRBO(LocalSearchConfig):
    """LSRBO implements the local search region method published in.
    https://www.merl.com/publications/docs/TR2023-057.pdf

    Attributes:
        gamma (float): The switsching parameter between local and global optimization.
            Defaults to 0.1. . The default is chosen for `qEI` as acquisition function.
            It has to be adapted to the acquisition function used, especially when log
            based acqfs are used.

    """

    type: Literal["LSRBO"] = "LSRBO"  # type: ignore
    gamma: float = 0.1

    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        return acqf_local >= self.gamma


AnyLocalSearchConfig = LSRBO


class BotorchOptimizer(AcquisitionOptimizer):
    type: Literal["BotorchOptimizer"] = "BotorchOptimizer"  # type: ignore
    n_restarts: PositiveInt = 8
    n_raw_samples: IntPowerOfTwo = 1024
    maxiter: PositiveInt = 2000
    batch_limit: Optional[PositiveInt] = Field(default=None, validate_default=True)

    # encoding params
    descriptor_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    categorical_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    discrete_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE

    # local search region params
    local_search_config: Optional[AnyLocalSearchConfig] = None

    @field_validator("batch_limit")
    @classmethod
    def validate_batch_limit(cls, batch_limit: int, info):
        batch_limit = min(
            batch_limit or info.data["n_restarts"],
            info.data["n_restarts"],
        )
        return batch_limit

    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        """Checks if a constraint is implemented. Currently only linear constraints are supported.

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise

        """
        if my_type in [
            constraints.NonlinearInequalityConstraint,
            constraints.NonlinearEqualityConstraint,
        ]:
            return False
        return True

    def validate_domain(self, domain: Domain):
        def validate_local_search_config(domain: Domain):
            if self.local_search_config is not None:
                if has_local_search_region(domain) is False:
                    warnings.warn(
                        "`local_search_region` config is specified, but no local search region is defined in `domain`",
                    )
                if (
                    len(domain.constraints)
                    - len(domain.constraints.get(constraints.LinearConstraint))
                    > 0
                ):
                    raise ValueError("LSR-BO only supported for linear constraints.")

        def validate_interpoint_constraints(domain: Domain):
            if domain.constraints.get(InterpointConstraint) and len(
                domain.inputs.get(ContinuousInput),
            ) != len(domain.inputs):
                raise ValueError(
                    "Interpoint constraints can only be used for pure continuous search spaces.",
                )

        validate_local_search_config(domain)
        validate_interpoint_constraints(domain)

    def validate_surrogate_specs(self, surrogate_specs: BotorchSurrogates):
        # we also have to check here that the categorical method is compatible with the chosen models
        # categorical_method = (
        #   values["categorical_method"] if "categorical_method" in values else None
        # )

        if self.categorical_method == CategoricalMethodEnum.FREE:
            for m in surrogate_specs.surrogates:
                if isinstance(m, MixedSingleTaskGPSurrogate):
                    raise ValueError(
                        "Categorical method FREE not compatible with a a MixedSingleTaskGPModel.",
                    )
        # we also check that if a categorical with descriptor method is used as one hot encoded the same method is
        # used for the descriptor as for the categoricals
        for m in surrogate_specs.surrogates:
            keys = m.inputs.get_keys(CategoricalDescriptorInput)
            for k in keys:
                input_proc_specs = (
                    m.input_preprocessing_specs[k]
                    if k in m.input_preprocessing_specs
                    else None
                )
                if input_proc_specs == CategoricalEncodingEnum.ONE_HOT:
                    if self.categorical_method != self.descriptor_method:
                        raise ValueError(
                            "One-hot encoded CategoricalDescriptorInput features has to be treated with the same method as categoricals.",
                        )


AnyAcqfOptimizer = BotorchOptimizer
