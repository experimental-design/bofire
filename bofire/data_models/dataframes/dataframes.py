from abc import abstractmethod
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.domain import Domain


class Value(BaseModel):
    value: Union[float, str]


class ExperimentOutputValue(Value):
    value: Optional[Union[float, str]] = Field(description="The oberved value.")  # type: ignore
    valid: bool = True


class CandidateOutputValue(Value):
    value: Optional[Union[float, str]] = Field(description="The predicted value.")  # type: ignore
    standard_deviation: float
    objective_value: float


class Row(BaseModel):
    type: str
    inputs: Dict[str, Value]
    outputs: Dict[str, Union[ExperimentOutputValue, CandidateOutputValue]]

    @abstractmethod
    def to_pandas(self) -> pd.Series:
        pass

    @staticmethod
    @abstractmethod
    def from_pandas(row: pd.Series, domain: Domain) -> "Row":
        pass

    def _inputs_to_dict(self) -> Dict[str, Union[float, str]]:
        return {k: v.value for k, v in self.inputs.items()}

    @property
    def input_keys(self):
        return sorted(self.inputs.keys())

    @property
    def output_keys(self):
        return list(self.outputs.keys())

    @property
    def categorical_input_keys(self):
        return sorted([k for k, v in self.inputs.items() if isinstance(v.value, str)])

    @property
    def continuous_input_keys(self):
        return sorted(
            [k for k, v in self.inputs.items() if not isinstance(v.value, str)]
        )

    @property
    def categorical_output_keys(self):
        return sorted([k for k, v in self.outputs.items() if isinstance(v.value, str)])

    @property
    def continuous_output_keys(self):
        return sorted(
            [k for k, v in self.outputs.items() if not isinstance(v.value, str)]
        )


class ExperimentRow(Row):
    type: Literal["ExperimentRow"] = "ExperimentRow"  # type: ignore
    outputs: Dict[str, ExperimentOutputValue]  # type: ignore

    def to_pandas(self) -> pd.Series:
        return pd.Series(
            {
                **self._inputs_to_dict(),
                **{k: v.value for k, v in self.outputs.items()},
                **{f"valid_{k}": v.valid for k, v in self.outputs.items()},
            }
        )

    @staticmethod
    def from_pandas(row: pd.Series, domain: Domain) -> "ExperimentRow":
        inputs = {k: Value(value=row[k]) for k in domain.inputs.get_keys()}
        outputs = {
            k: ExperimentOutputValue(
                value=row[k], valid=row[f"valid_{k}"] if f"valid_{k}" in row else True
            )
            for k in domain.outputs.get_keys()
        }
        return ExperimentRow(inputs=inputs, outputs=outputs)


class CandidateRow(Row):
    type: Literal["CandidateRow"] = "CandidateRow"  # type: ignore
    outputs: Dict[str, CandidateOutputValue]  # type: ignore

    def to_pandas(self) -> pd.Series:
        return pd.Series(
            {
                **self._inputs_to_dict(),
                **{f"{k}_pred": v.value for k, v in self.outputs.items()},
                **{f"{k}_std": v.standard_deviation for k, v in self.outputs.items()},
                **{f"{k}_des": v.objective_value for k, v in self.outputs.items()},
            }
        )

    @staticmethod
    def from_pandas(row: pd.Series, domain: Domain) -> "CandidateRow":
        inputs = {k: Value(value=row[k]) for k in domain.inputs.get_keys()}
        if f"{domain.outputs.get_keys()[0]}_pred" in row.index:
            outputs = {
                k: CandidateOutputValue(
                    value=row[f"{k}_pred"],
                    objective_value=row[f"{k}_des"],
                    standard_deviation=row[f"{k}_std"],
                )
                for k in domain.outputs.get_keys()
            }
        else:
            print(row)
            outputs = {}
        return CandidateRow(inputs=inputs, outputs=outputs)


class DataFrame(BaseModel):
    type: str
    rows: List[Row]

    @field_validator("rows")
    def validate_rows(cls, rows):
        if len({tuple(sorted(row.input_keys)) for row in rows}) > 1:
            raise ValueError("Rows must have the same input keys")
        if len({tuple(sorted(row.output_keys)) for row in rows}) > 1:
            raise ValueError("Rows must have the same output keys")
        if len({tuple(sorted(row.categorical_input_keys)) for row in rows}) > 1:
            raise ValueError("Rows must have the same categorical input keys")
        if len({tuple(sorted(row.continuous_input_keys)) for row in rows}) > 1:
            raise ValueError("Rows must have the same continuous input keys")
        if len({tuple(sorted(row.categorical_output_keys)) for row in rows}) > 1:
            raise ValueError("Rows must have the same categorical output keys")
        if len({tuple(sorted(row.continuous_output_keys)) for row in rows}) > 1:
            raise ValueError("Rows must have the same continuous input keys")
        return rows

    def to_pandas(self):
        return pd.concat([row.to_pandas() for row in self.rows], axis=1).T

    @staticmethod
    @abstractmethod
    def from_pandas(df: pd.DataFrame, domain: Domain) -> "DataFrame":
        pass


class Experiments(DataFrame):
    type: Literal["Experiments"] = "Experiments"  # type: ignore
    rows: List[ExperimentRow]  # type: ignore

    @staticmethod
    def from_pandas(df: pd.DataFrame, domain: Domain) -> "Experiments":
        return Experiments(
            rows=[ExperimentRow.from_pandas(row, domain) for _, row in df.iterrows()]
        )


class Candidates(DataFrame):
    type: Literal["Candidates"] = "Candidates"  # type: ignore
    rows: List[CandidateRow]  # type: ignore

    @staticmethod
    def from_pandas(df: pd.DataFrame, domain: Domain) -> "Candidates":
        return Candidates(
            rows=[CandidateRow.from_pandas(row, domain) for _, row in df.iterrows()]
        )
