from typing import Union

from bofire.data_models.dataframes.dataframes import (
    CandidateOutputValue,
    CandidateRow,
    Candidates,
    ExperimentOutputValue,
    ExperimentRow,
    Experiments,
    Value,
)


AnyDataFrame = Union[Experiments, Candidates]
AnyRow = Union[ExperimentRow, CandidateRow]
