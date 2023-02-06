import os

import pytest

from bofire.benchmarks.aspen_benchmark import Aspen_benchmark
from bofire.domain import Domain
from bofire.domain.constraints import Constraints, LinearInequalityConstraint
from bofire.domain.features import (
    CategoricalDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective

input_args = {
    "filename": "aspen_benchmark_test_file.apwz",
    "domain": Domain(
        input_features=InputFeatures(
            features=[
                ContinuousInput(
                    key="A", type="ContinuousInput", lower_bound=0, upper_bound=10.0
                ),
                ContinuousInput(
                    key="B", type="ContinuousInput", lower_bound=-45.0, upper_bound=12.0
                ),
                ContinuousInput(
                    key="C", type="ContinuousInput", lower_bound=0.0, upper_bound=1.0
                ),
                ContinuousInput(
                    key="D", type="ContinuousInput", lower_bound=-10.0, upper_bound=0.0
                ),
                ContinuousInput(
                    key="E", type="ContinuousInput", lower_bound=5.0, upper_bound=6.0
                ),
                ContinuousInput(
                    key="F", type="ContinuousInput", lower_bound=20.0, upper_bound=100.5
                ),
                ContinuousInput(
                    key="G", type="ContinuousInput", lower_bound=1000, upper_bound=1200
                ),
                CategoricalDescriptorInput(
                    key="H",
                    type="CategoricalDescriptorInput",
                    categories=["0", "1"],
                    allowed=[True, True],
                    descriptors=["binary"],
                    values=[[0.0], [1.0]],
                ),
            ]
        ),
        output_features=OutputFeatures(
            features=[
                ContinuousOutput(
                    key="X",
                    type="ContinuousOutput",
                    objective=MaximizeObjective(
                        type="MaximizeObjective", w=1.0, lower_bound=0, upper_bound=1
                    ),
                ),
                ContinuousOutput(
                    key="Y",
                    type="ContinuousOutput",
                    objective=MinimizeObjective(
                        type="MinimizeObjective", w=1.0, lower_bound=0, upper_bound=1
                    ),
                ),
                ContinuousOutput(
                    key="Z",
                    type="ContinuousOutput",
                    objective=MinimizeObjective(
                        type="MinimizeObjective", w=1.0, lower_bound=0, upper_bound=1
                    ),
                ),
            ]
        ),
        constraints=Constraints(
            constraints=[
                LinearInequalityConstraint(
                    type="LinearInequalityConstraint",
                    features=["E", "D"],
                    coefficients=[1.0, 1.0],
                    rhs=10.0,
                ),
                LinearInequalityConstraint(
                    type="LinearInequalityConstraint",
                    features=["C", "A"],
                    coefficients=[-4.0, -5.0],
                    rhs=-2.0,
                ),
                LinearInequalityConstraint(
                    type="LinearInequalityConstraint",
                    features=["A", "D", "C"],
                    coefficients=[-0.6, 7.0, 5.0],
                    rhs=-10.0,
                ),
                LinearInequalityConstraint(
                    type="LinearInequalityConstraint",
                    features=["B", "A"],
                    coefficients=[-1.0, -1.0],
                    rhs=-700.0,
                ),
            ],
            experiments=None,
            candidates=None,
        ),
    ),
    "paths": {
        "A": "\\Data\\Flowsheeting Options\\Calculator\\CA-01\\Input\\FVN_INIT_VAL\\A",
        "B": "\\Data\\Flowsheeting Options\\Calculator\\CA-01\\Input\\FVN_INIT_VAL\\B",
        "C": "\\Data\\Streams\\Stream1\\Input\\TEMP\\C",
        "D": "\\Data\\Blocks\\DUM-01\\Input\\D",
        "E": "\\Data\\Blocks\\DUM-02\\Input\\E",
        "F": "\\Data\\Blocks\\DUM-03\\Input\\F",
        "G": "\\Data\\Blocks\\DUM-4\\Input\\G",
        "H": "\\Data\\Flowsheeting Options\\Calculator\\CA-02\\Input\\FVN_INIT_VAL\\H",
        "X": "\\Data\\Blocks\\DUM-03\\Input\\X",
        "Y": "\\Data\\Blocks\\DUM-4\\Input\\Y",
        "Z": "\\Data\\Flowsheeting Options\\Calculator\\CA-02\\Input\\FVN_INIT_VAL\\Z",
    },
}


@pytest.mark.parametrize(
    "cls_benchmark, kwargs",
    [
        (Aspen_benchmark, input_args),
    ],
)
def test_aspen_benchmark(cls_benchmark: Aspen_benchmark, kwargs: dict):
    """Tests the initializer of Aspen_benchmark and whether the filename, domain and paths are set up correctly.

    Args:
        cls_benchmark (Aspen_benchmark): Aspen_benchmark class
        return_complete (bool): _description_
        kwargs (dict): Arguments to the initializer of Aspen_benchmark. {"filename": , "domain": , "paths": }
    """

    domain = kwargs["domain"]
    filename = kwargs["filename"]
    paths = kwargs["paths"]
    # Create an aspen mockup file, so an object of aspen_benchmark can be created.
    if not os.path.exists(filename):
        file = open(filename, "w")
        file.close()

    # Test, if domain gets set up correctly
    benchmark_function = cls_benchmark(**kwargs)
    assert benchmark_function.domain == domain, (
        "Error during domain set up."
        + "\n Expected: "
        + str(domain)
        + "\n Got: "
        + str(benchmark_function.domain)
    )
    assert benchmark_function.paths == paths, (
        "Error during set up of paths."
        + "\n Expected: "
        + str(paths)
        + "\n Got: "
        + str(benchmark_function.paths)
    )

    # Test, whether ValueError is raised, if a key has no path to aspen provided.
    paths.popitem()
    with pytest.raises(ValueError):
        benchmark_function = cls_benchmark(**kwargs)

    os.remove(kwargs["filename"])

    # Test, if filename error gets thrown for wrong or non-existent filename.
    with pytest.raises(ValueError):
        benchmark_function = cls_benchmark(**kwargs)
