import numpy as np
import pytest

from bofire.benchmarks.benchmark import Benchmark
from bofire.benchmarks.single import (
    Ackley,
    Branin,
    Branin30,
    DiscreteHimmelblau,
    Hartmann,
    Himmelblau,
    Multinormalpdfs,
    MultiTaskHimmelblau,
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
        (Branin30, False, {}),
        (MultiTaskHimmelblau, False, {}),
        (MultiTaskHimmelblau, True, {}),
        (Multinormalpdfs, False, {}),
        (Multinormalpdfs, True, {}),
        (
            Multinormalpdfs,  # user supplies own mean vectors and covariance matrices
            True,
            {
                "means": np.ones(shape=(10, 7)) * 0.5,
                "covmats": [np.diag(np.ones(7))] * 10,
                "dim": 7,
            },
        ),
        # TO DO: Implement feature that tests Ackley for categorical and descriptive inputs.
        # (Ackley, {"categorical": True}),
        # (Ackley, {"descriptor": True}),
    ],
)
def test_single_objective_benchmarks(
    cls_benchmark: type[Benchmark],
    return_complete: bool,
    kwargs,
):
    """Test function for single objective benchmark functions."""
    benchmark_function = cls_benchmark(**kwargs)
    benchmark_function_name = benchmark_function.__class__.__name__

    # Check for correct dimensions
    n_samples = 1
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


@pytest.mark.parametrize(
    "cls_benchmark, return_complete, kwargs1, kwargs2",
    [
        (Multinormalpdfs, False, {"seed": 42}, {"seed": 42}),
        (Multinormalpdfs, True, {"seed": 123}, {"seed": 123}),
    ],
)
def test_single_obj_benchmark_reproducibility(
    cls_benchmark,
    return_complete,
    kwargs1,
    kwargs2,
):
    benchmark_function = cls_benchmark(**kwargs1)
    benchmark_function_rep = cls_benchmark(**kwargs2)
    benchmark_function_name = benchmark_function.__class__.__name__

    # Check for correct dimensions
    n_samples = 27
    X_samples = benchmark_function.domain.inputs.sample(n=n_samples)

    Y = benchmark_function.f(X_samples, return_complete=return_complete)
    Yrep = benchmark_function_rep.f(X_samples, return_complete=return_complete)

    assert np.allclose(Y, Yrep, atol=1e-04), (
        "Attempt to reproduce results of " + benchmark_function_name + " failed."
    )
