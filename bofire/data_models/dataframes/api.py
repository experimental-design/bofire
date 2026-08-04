from bofire.data_models.dataframes.dataframes import (
    CandidateOutputValue,
    CandidateRow,
    Candidates,
    ExperimentOutputValue,
    ExperimentRow,
    Experiments,
    Value,
)
from bofire.data_models.unions import tagged_union


AnyDataFrame = tagged_union(Experiments, Candidates)
AnyRow = tagged_union(ExperimentRow, CandidateRow)
