from abc import abstractmethod
from typing import List, Literal, Optional

import pandas as pd
from pydantic import Field, root_validator, validator
from typing_extensions import Annotated

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


class Aggregation(BaseModel):
    type: str
    features: Annotated[List[str], Field(min_items=2)]
    keep_features: bool = False


class SumAggregation(Aggregation):
    type: Literal["SumAggregation"] = "SumAggregation"


class MeanAggregation(Aggregation):
    type: Literal["MeanAggregation"] = "MeanAggregation"


class Hyperconfig(BaseModel):
    type: str
    hyperstrategy: Literal["RandomStrategy", "FactorialStrategy", "SoboStrategy"]
    inputs: Inputs
    n_iterations: Optional[Annotated[int, Field(ge=1)]] = None
    target_metric: RegressionMetricsEnum = RegressionMetricsEnum.MAE

    @validator("n_iterations", always=True)
    def validate_n_iterations(cls, v, values):
        if v is None:
            if values["hyperstrategy"] == "FactorialStrategy":
                return v
            return len(values["inputs"]) + 10
        else:
            if values["hyperstrategy"] == "FactorialStrategy":
                raise ValueError(
                    "It is not allowed to scpecify the number of its for FactorialStrategy"
                )
            if v < len(values["inputs"]) + 2:
                raise ValueError(
                    "At least number of hyperparams plus 2 iterations has to be specified"
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
                    )
                ]
            ),
        )

    @staticmethod
    @abstractmethod
    def _update_hyperparameters(surrogate_data, hyperparameters: pd.Series):
        pass


class TrainableSurrogate(BaseModel):
    hyperconfig: Optional[Hyperconfig] = None
    aggregations: Optional[Annotated[List[Aggregation], Field(min_items=1)]] = None

    @root_validator
    def validate_aggregations(cls, values):
        if values["aggregations"] is None:
            return values
        for agg in values["aggregations"]:
            if len(set(agg.features) - set(values["inputs"].get_keys())) > 0:
                raise ValueError("Unkown feature keys provided in aggregation(s).")
        return values

    def update_hyperparameters(self, hyperparameters: pd.Series):
        if self.hyperconfig is not None:
            self.hyperconfig.domain.validate_candidates(
                pd.DataFrame(hyperparameters).T,
                only_inputs=True,
                raise_validation_error=True,
            )
            self.hyperconfig._update_hyperparameters(
                self, hyperparameters=hyperparameters
            )
        else:
            raise ValueError("No hyperconfig available.")
