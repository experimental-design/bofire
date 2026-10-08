from typing import List, Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.bnn import SingleTaskIBNNSurrogate
from bofire.data_models.surrogates.fully_bayesian import (
    FullyBayesianSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.linear import LinearSurrogate
from bofire.data_models.surrogates.map_saas import (
    AdditiveMapSaasSingleTaskGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.mixed_single_task_gp import (
    MixedSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.mlp import RegressionMLPEnsemble
from bofire.data_models.surrogates.polynomial import PolynomialSurrogate
from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
from bofire.data_models.surrogates.robust_single_task_gp import (
    RobustSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate
from bofire.data_models.unions import tagged_union


AnyOptionSurrogate = tagged_union(
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    RobustSingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    RegressionMLPEnsemble,
    FullyBayesianSingleTaskGPSurrogate,
    LinearSurrogate,
    PolynomialSurrogate,
    SingleTaskIBNNSurrogate,
    TanimotoGPSurrogate,
    AdditiveMapSaasSingleTaskGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
)


class SelectionSurrogate(Surrogate):
    """Surrogate that chooses one of several complete surrogates by cross-validation.

    When fitted, every option is cross-validated on the data, and the one with the
    best score in `metric` is fitted on all of the data and makes the predictions.
    Every option has the inputs and the single continuous output of this surrogate.

    Examples:
        >>> SelectionSurrogate(
        ...     inputs=inputs,
        ...     outputs=outputs,
        ...     options=SingleTaskGPSurrogate.options(inputs, outputs),
        ... )
    """

    type: Literal["SelectionSurrogate"] = "SelectionSurrogate"
    options: List[AnyOptionSurrogate] = Field(
        min_length=2,
        description="The complete surrogates to choose from, in order of preference: "
        "of two with the same score, the earlier one is chosen.",
    )
    metric: RegressionMetricsEnum = Field(
        default=RegressionMetricsEnum.MAE,
        description="Cross-validation metric used to rank the options. Each metric "
        "defines whether lower or higher values are better.",
    )
    folds: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation folds. With fewer experiments than "
        "folds, every experiment is left out once.",
    )
    random_state: Optional[int] = Field(
        default=None,
        description="Seed for splitting the experiments into folds; every option is "
        "scored on the same folds. If not provided, one is drawn at random when the "
        "surrogate is created.",
    )

    @model_validator(mode="after")
    def validate_options(self):
        """Check that every option has the inputs and output of this surrogate.

        Raises:
            ValueError: If an option's inputs or outputs differ from this
                surrogate's, or if there is more than one output.
        """
        for i, option in enumerate(self.options):
            if option.inputs != self.inputs:
                raise ValueError(f"Option {i} has different inputs.")
            if option.outputs != self.outputs:
                raise ValueError(f"Option {i} has a different output.")
        if len(self.outputs) != 1:
            raise ValueError("A selection surrogate predicts exactly one output.")
        return self

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        return issubclass(my_type, ContinuousOutput)
