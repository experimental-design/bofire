from pydantic import ValidationError

import bofire.data_models.dataframes.api as dataframes
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    dataframes.Experiments,
    lambda: {
        "rows": [
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "alpha": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
        ],
    },
)


specs.add_invalid(
    dataframes.Experiments,
    lambda: {
        "rows": [
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "alpha": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "c": "cat",
                },
                outputs={
                    "alpha": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
        ],
    },
    error=ValueError,
    message="Rows must have the same input keys",
)

specs.add_invalid(
    dataframes.Experiments,
    lambda: {
        "rows": [
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "alpha": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "gamma": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
        ],
    },
    error=ValueError,
    message="Rows must have the same output keys",
)

specs.add_invalid(
    dataframes.Experiments,
    lambda: {
        "rows": [
            dataframes.ExperimentRow(
                inputs={
                    "b": 1,
                    "a": "cat",
                },
                outputs={
                    "alpha": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "alpha": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
        ],
    },
    error=ValueError,
    message="Rows must have the same categorical input keys",
)

specs.add_invalid(
    dataframes.Experiments,
    lambda: {
        "rows": [
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "beta": dataframes.ExperimentOutputValue(value=2),
                    "alpha": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
            dataframes.ExperimentRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "alpha": dataframes.ExperimentOutputValue(value=2),
                    "beta": dataframes.ExperimentOutputValue(value="cat", valid=False),
                },
            ).model_dump(),
        ],
    },
    error=ValueError,
    message="Rows must have the same categorical output keys",
)


specs.add_valid(
    dataframes.Candidates,
    lambda: {
        "rows": [
            dataframes.CandidateRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "alpha": dataframes.CandidateOutputValue(
                        value=2,
                        standard_deviation=0,
                        objective_value=1,
                    ),
                    "beta": dataframes.CandidateOutputValue(
                        value="cat",
                        standard_deviation=0,
                        objective_value=1,
                    ),
                },
            ).model_dump(),
        ],
    },
)

specs.add_invalid(
    dataframes.Candidates,
    lambda: {
        "rows": [
            dataframes.CandidateRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={
                    "alpha": dataframes.CandidateOutputValue(
                        value=2,
                        standard_deviation=0,
                        objective_value=1,
                    ),
                    "beta": dataframes.ExperimentOutputValue(value="cat"),
                },
            ).model_dump(),
        ],
    },
    error=ValidationError,
)

specs.add_valid(
    dataframes.Candidates,
    lambda: {
        "rows": [
            dataframes.CandidateRow(
                inputs={
                    "a": 1,
                    "b": "cat",
                },
                outputs={},
            ).model_dump(),
        ],
    },
)
