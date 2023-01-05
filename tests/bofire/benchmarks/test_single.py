import numpy as np
import pandas as pd
import pytest

from bofire.benchmarks.single import Ackley, Himmelblau


@pytest.mark.parametrize(
    "cls_benchmark, kwargs",
    [
        (Himmelblau, {}),
        (Ackley, {}),
        # TO DO: Implement feature that tests Ackley for categorical and descriptive inputs.
        #(Ackley, {"categorical": True}),
        #(Ackley, {"descriptor": True}),
    ],
)
def test_single_objective_problems(cls_benchmark, kwargs):
    benchmark_function = cls_benchmark(**kwargs)
    benchmark_function_name = benchmark_function.__class__.__name__

    # Check for correct dimensions
    n_samples = 1000
    X_samples = benchmark_function.domain.inputs.sample(n=n_samples)
    # Calculating corresponding y values
    y = benchmark_function.f(X_samples)
    # Check, whether shape of output dataframe matches the expected shape.
    assert y.shape == (n_samples, len(benchmark_function.domain.inputs) + 2,), (
        "The shape of the output dataframe of "
        + benchmark_function_name
        + " does not match the expected shape."
    )

    # Check for correct optima
    # Retrieve optima from benchmark function
    optima = benchmark_function.get_optima()
    # Retrieve variable names
    x_keys = benchmark_function.domain.inputs.get_keys()
    y_keys = benchmark_function.domain.outputs.get_keys()
    px = optima[x_keys]
    py = optima[y_keys]
    Y = benchmark_function.f(px)
    y = Y[y_keys]
    # Check whether the optimum y value at position px equals the expected value at px with tolerance 'atol'.
    assert np.allclose(y, py, atol=1e-04), (
        "Expected optimum value of "
        + benchmark_function_name
        + " does not match calculated optimum value or is out of the tolerance radius."
    )

# test_single_objective_problems(Ackley, {"categorical":True})
