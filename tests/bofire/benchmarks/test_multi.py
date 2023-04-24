import pytest

from bofire.benchmarks.multi import C2DTLZ2, DTLZ2, ZDT1, CrossCoupling, SnarBenchmark


@pytest.mark.parametrize(
    "cls_benchmark, return_complete, kwargs",
    [
        (DTLZ2, True, {"dim": 5}),
        (SnarBenchmark, True, {}),
        (ZDT1, True, {}),
        (
            CrossCoupling,
            True,
            {},
        ),
        (DTLZ2, False, {"dim": 5}),
        (SnarBenchmark, False, {}),
        (ZDT1, False, {}),
        (
            CrossCoupling,
            False,
            {},
        ),
        (C2DTLZ2, True, {"dim": 4}),
        (C2DTLZ2, False, {"dim": 4}),
    ],
)
def test_multi_objective_benchmarks(cls_benchmark, return_complete, kwargs):
    """Test function for multi objective benchmark functions.

    Args:
        cls_benchmark (Benchmark function class): Benchmark function that is supposed to be tested.
        kwargs ({"dim": , "k":}): Optinal arguments for benchmark functions that require additional arguments. DTLZ2 requires "dim" and "k".
    """
    benchmark_function = cls_benchmark(**kwargs)
    benchmark_function_name = benchmark_function.__class__.__name__

    # Check for correct output dimensions
    n_samples = 1000
    X_samples = benchmark_function.domain.inputs.sample(n=n_samples)
    Y = benchmark_function.f(X_samples, return_complete=return_complete)

    # Define expected number of output variables
    expected_output_variables = len(benchmark_function.domain.outputs) * 2
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
