from typing import Annotated, Union

from pydantic import Field

from bofire.data_models.dataframes.dataframes import (
    CandidateOutputValue,
    CandidateRow,
    Candidates,
    ExperimentOutputValue,
    ExperimentRow,
    Experiments,
    Value,
)


AnyDataFrame = Annotated[Union[Experiments, Candidates], Field(discriminator="type")]
AnyRow = Annotated[Union[ExperimentRow, CandidateRow], Field(discriminator="type")]
