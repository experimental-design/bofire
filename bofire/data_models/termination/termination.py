"""Data models for BO termination conditions.

This module provides termination conditions for Bayesian optimization loops,
implementing methods from:

1. Makarova et al. (2022): "Automatic Termination for Hyperparameter Optimization"
   (AutoML 2022)
   - UCBLCBRegretTermination: True implementation of the UCB-LCB regret bound.
     Terminates when min_UCB(evaluated) - min_LCB(domain) < epsilon, where
     epsilon is based on the noise variance.

These termination conditions can reduce experimental costs in chemical optimization
by automatically determining when to stop the BO loop.
"""

from abc import abstractmethod
from typing import Annotated, Any, List, Literal, Optional, Union

import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain


class EvaluatableTermination:
    """Protocol for termination conditions that can be evaluated."""

    @abstractmethod
    def should_terminate(
        self,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
        iteration: int,
        **kwargs,
    ) -> bool:
        """Evaluate whether the optimization should terminate.

        Args:
            domain: The optimization domain.
            experiments: The experiments conducted so far.
            iteration: The current iteration number.
            **kwargs: Additional keyword arguments (e.g., strategy, metric values).

        Returns:
            True if the optimization should terminate, False otherwise.
        """
        pass


class TerminationCondition(BaseModel):
    """Base class for termination conditions."""

    type: Any


class MaxIterationsTermination(TerminationCondition, EvaluatableTermination):
    """Terminate after a fixed number of iterations.

    This is the simplest termination condition, commonly used as a fallback
    to ensure the optimization eventually stops.

    Attributes:
        max_iterations: Maximum number of iterations before termination.
    """

    type: Literal["MaxIterationsTermination"] = "MaxIterationsTermination"
    max_iterations: PositiveInt

    def should_terminate(
        self,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
        iteration: int,
        **kwargs,
    ) -> bool:
        return iteration >= self.max_iterations


class AlwaysContinue(TerminationCondition, EvaluatableTermination):
    """Never terminate (always continue).

    Use this as a placeholder when no termination condition is desired,
    or when termination is handled externally.
    """

    type: Literal["AlwaysContinue"] = "AlwaysContinue"

    def should_terminate(
        self,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
        iteration: int,
        **kwargs,
    ) -> bool:
        return False


class UCBLCBRegretTermination(TerminationCondition, EvaluatableTermination):
    """Terminate based on UCB-LCB regret bound from Makarova et al. (2022).

    Implements the stopping criterion from:
    "Automatic Termination for Hyperparameter Optimization" (AutoML 2022)

    The regret bound is computed as:
        regret_bound = min_{x in evaluated} UCB(x) - min_{x in domain} LCB(x)

    Where:
    - UCB(x) = mu(x) + beta * sigma(x)  (Upper Confidence Bound)
    - LCB(x) = mu(x) - beta * sigma(x)  (Lower Confidence Bound)

    If the strategy uses qUCB as its acquisition function, the same beta value
    will be used. Otherwise, beta is computed from GP-UCB theory:
        beta = 2 * log(d * t^2 * pi^2 / (6 * delta))

    The optimization terminates when the regret bound falls below epsilon_BO,
    which is set based on the noise variance of the observations.

    This method is particularly useful when:
    - You have an estimate of the observation noise variance
    - You want a principled, theoretically-grounded stopping criterion
    - The optimization uses GP-based surrogates

    Attributes:
        noise_variance: Observation noise variance (sigma^2). If None, will be
            estimated from the GP model's likelihood.
        threshold_factor: Multiplier for noise_variance to get epsilon_BO.
            Default is 1.0, meaning epsilon_BO = noise_variance.
        min_iterations: Minimum iterations before checking termination.
    """

    type: Literal["UCBLCBRegretTermination"] = "UCBLCBRegretTermination"
    noise_variance: Optional[PositiveFloat] = None
    threshold_factor: PositiveFloat = 1.0
    min_iterations: PositiveInt = 5

    def should_terminate(
        self,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
        iteration: int,
        **kwargs,
    ) -> bool:
        if iteration < self.min_iterations:
            return False

        regret_bound = kwargs.get("regret_bound", None)
        if regret_bound is None:
            return False

        # Get noise variance: user-provided or estimated from model
        if self.noise_variance is not None:
            noise_var = self.noise_variance
        else:
            noise_var = kwargs.get("estimated_noise_variance", None)
            if noise_var is None:
                return False

        # Compute threshold: epsilon_BO = threshold_factor * noise_variance
        epsilon_bo = self.threshold_factor * noise_var

        return regret_bound < epsilon_bo


class CombiTerminationCondition(TerminationCondition, EvaluatableTermination):
    """Combine multiple termination conditions.

    This allows creating complex termination logic by combining simple conditions.
    The combination can require either any condition (OR) or all conditions (AND)
    to be satisfied.

    Attributes:
        conditions: List of termination conditions to combine.
        n_required_conditions: Number of conditions that must be satisfied.
            - 1 means any condition (OR logic)
            - len(conditions) means all conditions (AND logic)
    """

    type: Literal["CombiTerminationCondition"] = "CombiTerminationCondition"
    conditions: Annotated[
        List[
            Union[
                MaxIterationsTermination,
                UCBLCBRegretTermination,
                AlwaysContinue,
                "CombiTerminationCondition",
            ]
        ],
        Field(min_length=1),
    ]
    n_required_conditions: PositiveInt = 1

    @field_validator("n_required_conditions")
    @classmethod
    def validate_n_required_conditions(cls, v, info):
        if "conditions" in info.data and v > len(info.data["conditions"]):
            raise ValueError(
                "n_required_conditions cannot be larger than number of conditions.",
            )
        return v

    def should_terminate(
        self,
        domain: Domain,
        experiments: Optional[pd.DataFrame],
        iteration: int,
        **kwargs,
    ) -> bool:
        n_satisfied = 0
        for condition in self.conditions:
            if condition.should_terminate(domain, experiments, iteration, **kwargs):
                n_satisfied += 1
                if n_satisfied >= self.n_required_conditions:
                    return True
        return False


# Type alias for any termination condition
AnyTerminationCondition = Union[
    MaxIterationsTermination,
    UCBLCBRegretTermination,
    CombiTerminationCondition,
    AlwaysContinue,
]
