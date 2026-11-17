from typing import Literal, Type, Union

from pydantic import Field

from bofire.data_models.acquisition_functions.api import qEI, qLogEI, qLogNEI, qNEI
from bofire.data_models.constraints.api import Constraint, InterpointConstraint
from bofire.data_models.features.api import Feature
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)


class QparegoStrategy(MultiobjectiveStrategy):
    """Multi-objective Bayesian optimization by random scalarizations (qParEGO).

    Each candidate of a batch optimizes its own randomly weighted Chebyshev
    combination of the objectives with a single-objective acquisition function, so
    the batch spreads along the Pareto front. Cheaper than the hypervolume-based
    `MoboStrategy` with many objectives or large batches. Interpoint constraints are
    not supported.

    Examples:
        >>> QparegoStrategy(domain=domain, acquisition_function=qLogNEI())
    """

    type: Literal["QparegoStrategy"] = "QparegoStrategy"
    acquisition_function: Union[qEI, qLogEI, qLogNEI, qNEI] = Field(
        # a factory, as constructing the deprecated qNEI warns
        default_factory=lambda: qNEI(),
        description="Single-objective acquisition function applied to each "
        "scalarization.",
    )

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise

        """
        if my_type == InterpointConstraint:
            return False

        return super().is_constraint_implemented(my_type)
