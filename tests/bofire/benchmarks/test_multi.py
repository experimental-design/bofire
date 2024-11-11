import pytest

from bofire.benchmarks.benchmark import Benchmark
from bofire.benchmarks.multi import (
    BNH,
    C2DTLZ2,
    DTLZ2,
    TNK,
    ZDT1,
    CrossCoupling,
    SnarBenchmark,
)


@pytest.mark.parametrize(
    "cls_benchmark, return_complete, kwargs",
    [
        (DTLZ2, True, {"dim": 5}),
        (DTLZ2, False, {"dim": 5}),
        (SnarBenchmark, True, {}),
        (SnarBenchmark, False, {}),
        (ZDT1, True, {}),
        (ZDT1, False, {}),
        (CrossCoupling, True, {}),
        (CrossCoupling, False, {}),
        (C2DTLZ2, True, {"dim": 4}),
        (C2DTLZ2, False, {"dim": 4}),
        (BNH, False, {"constraints": True}),
        (BNH, False, {"constraints": False}),
        (TNK, False, {}),
        (TNK, True, {}),
    ],
)
def test_multi_objective_benchmarks(
    cls_benchmark: type[Benchmark],
    return_complete: bool,
    kwargs,
):
    """Test function for multi objective benchmark functions."""
    benchmark_function = cls_benchmark(**kwargs)
    benchmark_function_name = benchmark_function.__class__.__name__

    # Check for correct output dimensions
    n_samples = 1000
    X_samples = benchmark_function.domain.inputs.sample(n=n_samples)
    Y = benchmark_function.f(X_samples, return_complete=return_complete)

    # Define expected number of output variables
    expected_output_variables = len(benchmark_function.domain.outputs) * 2
    print(Y.shape, expected_output_variables)
    # Check, whether expected number of output variables match the actual number
    if return_complete:
        assert Y.shape == (
            n_samples,
            len(benchmark_function.domain.experiment_column_names),
        ), (
            "The shape of the output dataframe of "
            + benchmark_function_name
            + " does not match the expected shape of "
            + "("
            + str(n_samples)
            + ","
            + str(expected_output_variables)
            + ")"
        )
    else:
        assert Y.shape == (n_samples, expected_output_variables), (
            "The shape of the output dataframe of "
            + benchmark_function_name
            + " does not match the expected shape of "
            + "("
            + str(n_samples)
            + ","
            + str(expected_output_variables)
            + ")"
        )

    # Check for correct optima/pareto front for multiobjective functions that return a vector
    # What are the corresponding X vectors to the multiobjective optima space of the input space? -> Not implemented yet in the multiobjective function
