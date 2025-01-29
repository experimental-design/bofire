import os

import pytest

from bofire.benchmarks.aspen_benchmark import Aspen_benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MaximizeObjective


@pytest.mark.parametrize(
    "cls_benchmark",
    [
        (Aspen_benchmark),
    ],
)
def test_aspen_benchmark(cls_benchmark: Aspen_benchmark):
    """Tests the initializer of Aspen_benchmark and whether the filename, domain and paths are set up correctly.

    Args:
        cls_benchmark (Aspen_benchmark): Aspen_benchmark class
        return_complete (bool): _description_
        kwargs (dict): Arguments to the initializer of Aspen_benchmark. {"filename": , "domain": , "paths": }

    """
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="A", type="ContinuousInput", bounds=(0, 10)),
                ContinuousInput(
                    key="B",
                    type="ContinuousInput",
                    bounds=(-45, 20),
                ),
                CategoricalDescriptorInput(
                    key="C",
                    type="CategoricalDescriptorInput",
                    categories=["0", "1"],
                    allowed=[True, True],
                    descriptors=["binary"],
                    values=[[0.0], [1.0]],
                ),
            ],
        ),
        outputs=Outputs(
            features=[
                ContinuousOutput(
                    key="X",
                    type="ContinuousOutput",
                    objective=MaximizeObjective(
                        type="MaximizeObjective",
                        w=1.0,
                    ),
                ),
            ],
        ),
    )
    filename = "aspen_benchmark_test_file.apwz"
    paths = {
        "A": "\\Data\\Flowsheeting Options\\Calculator\\CA-01\\Input\\FVN_INIT_VAL\\A",
        "B": "\\Data\\Flowsheeting Options\\Calculator\\CA-01\\Input\\FVN_INIT_VAL\\B",
        "C": "\\Data\\Streams\\Stream1\\Input\\TEMP\\C",
        "X": "\\Data\\Blocks\\DUM-03\\Input\\X",
    }
    # Create an aspen mockup file, so an object of aspen_benchmark can be created.
    if not os.path.exists(filename):
        file = open(filename, "w")
        file.close()

    # Test, if domain gets set up correctly
    benchmark_function = cls_benchmark(filename=filename, domain=domain, paths=paths)
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
        benchmark_function = cls_benchmark(
            filename=filename,
            domain=domain,
            paths=paths,
        )

    os.remove(filename)

    # Test, if filename error gets thrown for wrong or non-existent filename.
    with pytest.raises(ValueError):
        benchmark_function = cls_benchmark(
            filename=filename,
            domain=domain,
            paths=paths,
        )


test_aspen_benchmark(cls_benchmark=Aspen_benchmark)
