from typing import Annotated, Any, Literal, Optional

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.enum import RegressionMetricsEnum, UQRegressionMetricsEnum
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective


metrics2objectives = {
    RegressionMetricsEnum.MAE: MinimizeObjective,
    RegressionMetricsEnum.MAPE: MinimizeObjective,
    RegressionMetricsEnum.MSD: MinimizeObjective,
    RegressionMetricsEnum.R2: MaximizeObjective,
    RegressionMetricsEnum.PEARSON: MaximizeObjective,
    RegressionMetricsEnum.SPEARMAN: MaximizeObjective,
    RegressionMetricsEnum.FISHER: MaximizeObjective,
    UQRegressionMetricsEnum.PEARSON_UQ: MaximizeObjective,
    UQRegressionMetricsEnum.SPEARMAN_UQ: MaximizeObjective,
    UQRegressionMetricsEnum.KENDALL_UQ: MaximizeObjective,
    UQRegressionMetricsEnum.MAXIMUMCALIBRATION: MinimizeObjective,
    UQRegressionMetricsEnum.MISCALIBRATIONAREA: MinimizeObjective,
    UQRegressionMetricsEnum.ABSOLUTEMISCALIBRATIONAREA: MinimizeObjective,
}


# reused by every Hyperconfig subclass, which must redeclare these fields to narrow
# their types or fix their defaults
HYPERSTRATEGY_DESCRIPTION = (
    "Strategy used to search the hyperparameters. A fractional factorial covers the "
    "space in a fixed number of runs, random and SOBO take `n_iterations`."
)
HYPERCONFIG_INPUTS_DESCRIPTION = (
    "The hyperparameters to optimize, as input features. Each subclass fixes these to "
    "the hyperparameters of its own surrogate."
)
TARGET_METRIC_DESCRIPTION = (
    "Cross-validation metric the search optimizes. Whether it is maximized or "
    "minimized follows from the metric."
)


class Hyperconfig(BaseModel):
    """Configuration of a hyperparameter optimization for a surrogate.

    The surrogate's hyperparameters become the inputs of a small optimization problem
    whose output is a cross-validation metric, so choosing a kernel or a prior family is
    itself optimized rather than guessed. Carrying the config does not run anything: the
    optimization is run explicitly with `hyperoptimize`, or by a strategy configured
    with `frequency_hyperopt`.
    """

    type: Any
    hyperstrategy: Literal[
        "RandomStrategy", "FractionalFactorialStrategy", "SoboStrategy"
    ] = Field(description=HYPERSTRATEGY_DESCRIPTION)
    inputs: Inputs = Field(description=HYPERCONFIG_INPUTS_DESCRIPTION)
    n_iterations: Optional[Annotated[int, Field(ge=1)]] = Field(
        default=None,
        description="Number of hyperparameter configurations to try. Must be at least "
        "the number of hyperparameters plus two, and must not be set for a fractional "
        "factorial, whose size is fixed. Defaults to the number of hyperparameters "
        "plus ten.",
    )
    target_metric: RegressionMetricsEnum = Field(
        default=RegressionMetricsEnum.MAE, description=TARGET_METRIC_DESCRIPTION
    )

    @field_validator("n_iterations")
    @classmethod
    def validate_n_iterations(cls, v, values):
        if v is None:
            if values.data["hyperstrategy"] == "FractionalFactorialStrategy":
                return v
            return len(values.data["inputs"]) + 10
        if values.data["hyperstrategy"] == "FractionalFactorialStrategy":
            raise ValueError(
                "It is not allowed to specify the number of its for FractionalFactorialStrategy",
            )
        if v < len(values.data["inputs"]) + 2:
            raise ValueError(
                "At least number of hyperparams plus 2 iterations has to be specified",
            )
        return v

    @property
    def domain(self) -> Domain:
        return Domain(
            inputs=self.inputs,
            outputs=Outputs(
                features=[
                    ContinuousOutput(
                        key=self.target_metric.name,
                        objective=metrics2objectives[self.target_metric](),
                    ),
                ],
            ),
        )

    @staticmethod
    def _update_hyperparameters(surrogate_data, hyperparameters: pd.Series):
        raise NotImplementedError(
            "Ideally this would be an abstract method, but this causes problems in pydantic.",
        )


class TrainableSurrogate(BaseModel):
    """Surrogate whose parameters are learned from the experiments.

    Such a surrogate can also carry a config for optimizing its hyperparameters -- the
    settings that the fit itself does not learn.
    """

    hyperconfig: Optional[Hyperconfig] = Field(
        default=None,
        description="Configuration of a hyperparameter optimization for this surrogate. "
        "Carrying it does not run anything; without it, no hyperparameter optimization "
        "is possible.",
    )

    def update_hyperparameters(self, hyperparameters: pd.Series):
        if self.hyperconfig is not None:
            self.hyperconfig.domain.validate_candidates(
                pd.DataFrame(hyperparameters).T,
                only_inputs=True,
                raise_validation_error=True,
            )
            self.hyperconfig._update_hyperparameters(
                self,
                hyperparameters=hyperparameters,
            )
        else:
            raise ValueError("No hyperconfig available.")
