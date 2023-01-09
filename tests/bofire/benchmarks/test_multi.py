import pytest

from bofire.benchmarks.multi import DTLZ2, ZDT1, SnarBenchmark


@pytest.mark.parametrize(
    "cls_benchmark, kwargs",
    [(DTLZ2, {"dim": 5, "k": 1}), (SnarBenchmark, {}), (ZDT1, {})],
)
def test_multi_objective_benchmarks(cls_benchmark, kwargs):
    benchmark_function = cls_benchmark(**kwargs)
    benchmark_function_name = benchmark_function.__class__.__name__
    dim = 0
    if "dim" in kwargs:
        dim = kwargs.get("dim")

    # Check for correct output dimensions
    n_samples = 1000
    X_samples = benchmark_function.domain.inputs.sample(n=n_samples)
    Y = benchmark_function.f(X_samples)
    # Define expected number of output variables
    # Multiobjective functions create a separate DataFrame for output values
    # ZDT has 2 output variables, DTLZ2 returns 4 output variables + its input variables, SnarBenchmark returns 8 variables
    number_of_output_variables_dict = {"DTLZ2": 4 + dim, "SnarBenchmark": 8, "ZDT1": 2}
    number_of_output_variables = number_of_output_variables_dict.get(
        benchmark_function_name, 0  # otherwise 0
    )
    # Check, whether expected number of output variables match the actual number
    assert Y.shape == (n_samples, number_of_output_variables), (
        "The shape of the output dataframe of "
        + benchmark_function_name
        + " does not match the expected shape of "
        + "("
        + str(n_samples)
        + ","
        + str(number_of_output_variables)
        + ")"
    )

    # Check for correct optima/pareto front for multiobjective functions that return a vector
    # What are the corresponding X vectors to the multiobjective optima space of the input space? -> Not implemented yet in the multiobjective function
    # n_samples = 10
    # X_samples = benchmark_function.domain.inputs.sample(n=n_samples)
    # optima = benchmark_function.get_optima(points=n_samples)
    # Y = benchmark_function.f(X_samples)
    # print(optima)
    # print(X_samples)
