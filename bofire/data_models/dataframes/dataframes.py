from abc import abstractmethod
from collections.abc import Sequence
from typing import Dict, Generic, Literal, Optional, TypeVar, Union

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.domain import Domain


def _append_std(s: str):
    return f"{s}_sd"


def _append_pred(s: str):
    return f"{s}_pred"


def _append_des(s: str):
    return f"{s}_des"


Value = Union[str, float]


class OutputValue(BaseModel):
    value: Optional[Value]


TOutputValue = TypeVar("TOutputValue", bound=OutputValue)


class ExperimentOutputValue(OutputValue):
    value: Optional[Value] = Field(description="The observed value.")
    valid: bool = True


class CandidateOutputValue(OutputValue):
    value: Value = Field(description="The predicted value.")
    standard_deviation: float
    objective_value: float


class Row(BaseModel, Generic[TOutputValue]):
    inputs: Dict[str, Value]
    outputs: Dict[str, TOutputValue]

    @abstractmethod
    def to_pandas(self) -> pd.Series:
        pass

    @staticmethod
    @abstractmethod
    def from_pandas(row: pd.Series, domain: Domain) -> "Row":
        pass

    @property
    def input_keys(self):
        return sorted(self.inputs.keys())

    @property
    def output_keys(self):
        return list(self.outputs.keys())

    @property
    def categorical_input_keys(self):
        return sorted([k for k, v in self.inputs.items() if isinstance(v, str)])

    @property
    def continuous_input_keys(self):
        return sorted([k for k, v in self.inputs.items() if not isinstance(v, str)])

    @property
    def categorical_output_keys(self):
        return sorted([k for k, v in self.outputs.items() if isinstance(v.value, str)])

    @property
    def continuous_output_keys(self):
        return sorted(
            [k for k, v in self.outputs.items() if not isinstance(v.value, str)],
        )


class ExperimentRow(Row[ExperimentOutputValue]):
    type: Literal["ExperimentRow"] = "ExperimentRow"

    def to_pandas(self) -> pd.Series:
        return pd.Series(
            {
                **self.inputs,
                **{k: v.value for k, v in self.outputs.items()},
                **{f"valid_{k}": v.valid for k, v in self.outputs.items()},
            },
        )

    @staticmethod
    def from_pandas(row: pd.Series, domain: Domain) -> "ExperimentRow":
        inputs = {k: row[k] for k in domain.inputs.get_keys()}
        outputs = {
            k: ExperimentOutputValue(
                value=row[k],
                valid=row[f"valid_{k}"] if f"valid_{k}" in row else True,
            )
            for k in domain.outputs.get_keys()
        }
        return ExperimentRow(inputs=inputs, outputs=outputs)


class CandidateRow(Row[CandidateOutputValue]):
    type: Literal["CandidateRow"] = "CandidateRow"

    def to_pandas(self) -> pd.Series:
        return pd.Series(
            {
                **self.inputs,
                **{_append_pred(k): v.value for k, v in self.outputs.items()},
                **{
                    _append_std(k): v.standard_deviation
                    for k, v in self.outputs.items()
                },
                **{_append_des(k): v.objective_value for k, v in self.outputs.items()},
            },
        )

    @staticmethod
    def from_pandas(row: pd.Series, domain: Domain) -> "CandidateRow":
        inputs = {k: row[k] for k in domain.inputs.get_keys()}
        if f"{domain.outputs.get_keys()[0]}_pred" in row.index:
            outputs = {
                k: CandidateOutputValue(
                    value=row[_append_pred(k)],
                    objective_value=row[_append_des(k)],
                    standard_deviation=row[_append_std(k)],
                )
                for k in domain.outputs.get_keys()
            }
        else:
            outputs = {}
        return CandidateRow(inputs=inputs, outputs=outputs)


TRow = TypeVar("TRow", bound=Row)


class DataFrame(BaseModel, Generic[TRow]):
    rows: Sequence[TRow]

    def __len__(self):
        return len(self.rows)

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


class Experiments(DataFrame[ExperimentRow]):
    type: Literal["Experiments"] = "Experiments"

    @staticmethod
    def from_pandas(df: pd.DataFrame, domain: Domain) -> "Experiments":
        return Experiments(
            rows=[ExperimentRow.from_pandas(row, domain) for _, row in df.iterrows()],
        )


class Candidates(DataFrame[CandidateRow]):
    type: Literal["Candidates"] = "Candidates"

    @staticmethod
    def from_pandas(df: pd.DataFrame, domain: Domain) -> "Candidates":
        return Candidates(
            rows=[CandidateRow.from_pandas(row, domain) for _, row in df.iterrows()],
        )
