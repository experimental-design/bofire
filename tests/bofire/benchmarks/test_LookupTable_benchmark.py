import numpy as np
import pandas as pd
import pytest

from bofire.benchmarks.LookupTableBenchmark import LookupTableBenchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import CategoricalInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective


@pytest.mark.parametrize(
    "cls_benchmark",
    [
        (LookupTableBenchmark),
    ],
)
def test_LookupTable_benchmark(cls_benchmark: LookupTableBenchmark):
    """Tests the initializer of Aspen_benchmark and whether the filename, domain and paths are set up correctly.

    Args:
        cls_benchmark (Aspen_benchmark): Aspen_benchmark class
        return_complete (bool): _description_
        kwargs (dict): Arguments to the initializer of Aspen_benchmark. {"filename": , "domain": , "paths": }

    """
    look_up = pd.DataFrame(columns=["x1", "x2", "y"])
    look_up["x1"] = np.random.choice(["a", "b", "c", "d"], 10)
    look_up["x2"] = np.random.choice(["e", "b", "f", "d"], 10)
    look_up["y"] = np.random.rand(10)
    look_up = look_up.drop_duplicates(subset=["x1", "x2"])
    input_feature1 = CategoricalInput(
        key="x1",
        categories=list(set(look_up["x1"].to_list())),
    )
    input_feature2 = CategoricalInput(
        key="x2",
        categories=list(set(look_up["x2"].to_list())),
    )
    objective = MaximizeObjective(
        w=1.0,
    )
    inputs = Inputs(features=[input_feature1, input_feature2])
    output_feature = ContinuousOutput(key="y", objective=objective)
    outputs = Outputs(features=[output_feature])
    domain = Domain(inputs=inputs, outputs=outputs)
    benchmark_function = cls_benchmark(domain=domain, lookup_table=look_up)
    # Test, if domain gets set up correctly
    assert benchmark_function.domain == domain
    # Test, if lookup table gets set up correctly
    pd.testing.assert_frame_equal(benchmark_function.lookup_table, look_up)
    # Test, if domain raises error for wrong key
    output_feature = ContinuousOutput(key="y1", objective=objective)
    outputs = Outputs(features=[output_feature])
    domain = Domain(inputs=inputs, outputs=outputs)
    with pytest.raises(ValueError):
        benchmark_function = cls_benchmark(domain=domain, lookup_table=look_up)
    # Test, if the output function works
    inputs = Inputs(features=[input_feature1, input_feature2])
    output_feature = ContinuousOutput(key="y", objective=objective)
    outputs = Outputs(features=[output_feature])
    domain = Domain(inputs=inputs, outputs=outputs)
    benchmark_function = cls_benchmark(domain=domain, lookup_table=look_up)
    sampled = look_up.loc[:2, domain.inputs.get_keys()].copy()
    benchmark_function.f(sampled, return_complete=True)
    # Test, if output function raises error
    sampled.loc[1, "x2"] = "a"
    with pytest.raises(ValueError):
        benchmark_function.f(sampled, return_complete=True)
