import numpy as np
import pytest

from bofire.benchmarks.single import (
    Ackley,
    Branin,
    Branin30,
    DiscreteHimmelblau,
    Hartmann,
    Himmelblau,
    _CategoricalDiscreteHimmelblau,
)


def test_hartmann():
    with pytest.raises(ValueError):
        Hartmann(8)
    h = Hartmann(dim=6, allowed_k=3)
    assert h.dim == 6
    assert h.domain.constraints[0].max_count == 3
    with pytest.raises(ValueError):
        h.get_optima()
    h = Hartmann(dim=6, allowed_k=None)
    assert len(h.domain.constraints) == 0


@pytest.mark.parametrize(
    "cls_benchmark, return_complete, kwargs",
    [
        (Himmelblau, False, {}),
        (DiscreteHimmelblau, False, {}),
        (DiscreteHimmelblau, True, {}),
        (_CategoricalDiscreteHimmelblau, False, {}),
        (_CategoricalDiscreteHimmelblau, True, {}),
        (Ackley, False, {}),
        (Himmelblau, True, {}),
        (Ackley, True, {}),
        (Hartmann, True, {}),
        (Hartmann, False, {}),
        (Branin, True, {}),
        (Branin, False, {}),
        (Branin30, True, {}),
        (Branin30, False, {})
        # TO DO: Implement feature that tests Ackley for categorical and descriptive inputs.
        # (Ackley, {"categorical": True}),
        # (Ackley, {"descriptor": True}),
    ],
)
def test_single_objective_benchmarks(cls_benchmark, return_complete, kwargs):
    """Test function for single objective benchmark functions.

    Args:
        cls_benchmark (Benchmark function class): Benchmark function that is supposed to be tested.
        kwargs ({"dim": , "k":}): Optinal arguments for benchmark functions that require additional arguments. Ackley can handle categerical and descriptive inputs.
    """
    benchmark_function = cls_benchmark(**kwargs)
    benchmark_function_name = benchmark_function.__class__.__name__

    # Check for correct dimensions
    n_samples = 1000
    X_samples = benchmark_function.domain.inputs.sample(n=n_samples)
    # Calculating corresponding y values
    Y = benchmark_function.f(X_samples, return_complete=return_complete)
    # Check, whether shape of output dataframe matches the expected shape.
    expected_output_variables = len(benchmark_function.domain.outputs) * 2

    if return_complete:
        assert Y.shape == (
            n_samples,
            len(benchmark_function.domain.experiment_column_names),
        ), (
            "The shape of the output dataframe of "
            + benchmark_function_name
            + " does not match the expected shape."
        )
    else:
        assert Y.shape == (n_samples, expected_output_variables), (
            "The shape of the output dataframe of "
            + benchmark_function_name
            + " does not match the expected shape."
        )

    # Check for correct optima
    # Retrieve optima from benchmark function
    try:
        optima = benchmark_function.get_optima()
        # Retrieve variable names
        x_keys = benchmark_function.domain.inputs.get_keys()
        y_keys = benchmark_function.domain.outputs.get_keys()
        px = optima[x_keys]
        py = optima[y_keys]
        Y = benchmark_function.f(px, return_complete=return_complete)
        y = Y[y_keys]
        # Check whether the optimum y value at position px equals the expected value at px with tolerance 'atol'.
        assert np.allclose(y, py, atol=1e-04), (
            "Expected optimum value of "
            + benchmark_function_name
            + " does not match calculated optimum value or is out of the tolerance radius."
        )
    except NotImplementedError:
        pass
