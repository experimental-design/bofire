from typing import Annotated, Literal, Optional

import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain, EngineeredFeatures, Inputs, Outputs
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


class Hyperconfig(BaseModel):
    type: str
    hyperstrategy: Literal[
        "RandomStrategy", "FractionalFactorialStrategy", "SoboStrategy"
    ]
    inputs: Inputs
    n_iterations: Optional[Annotated[int, Field(ge=1)]] = None
    target_metric: RegressionMetricsEnum = RegressionMetricsEnum.MAE

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
    hyperconfig: Optional[Hyperconfig] = None
    engineered_features: EngineeredFeatures = Field(
        default_factory=lambda: EngineeredFeatures()
    )

    @model_validator(mode="after")
    def validate_aggregations(self):
        self.engineered_features.validate_inputs(
            self.inputs
        )  # ty: ignore[unresolved-attribute]
        return self

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
