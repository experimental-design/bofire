"""Surrogate presets.

A preset is a named configuration of an existing surrogate rather than a surrogate in
its own right. It is a function, not a data model, so what it returns serializes as the
surrogate it configures -- there is no extra type to register, and a caller can always
reach the same result by hand.

The same split applies to the priors, where `THREESIX_LENGTHSCALE_PRIOR` and its
siblings are `partial`s over `GammaPrior` rather than classes.
"""

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import LinearKernel, PolynomialKernel
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, GreaterThan
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate


def LinearSurrogate(
    inputs: Inputs,
    outputs: Outputs,
    **kwargs,
) -> SingleTaskGPSurrogate:
    """Build a single-task GP restricted to linear responses.

    The linear kernel still yields a predicted uncertainty, so the surrogate can be used
    in a Bayesian optimization loop, but it cannot represent curvature. Pick it when the
    response is known to be linear, or when there are too few experiments to support
    anything richer.

    Args:
        inputs: Input features the surrogate acts on.
        outputs: Output feature the surrogate predicts.
        **kwargs: Any other field of `SingleTaskGPSurrogate`. `noise_prior`,
            `noise_constraint` and `hyperconfig` default to values suited to a linear
            kernel rather than to the ones the GP itself defaults to.

    Returns:
        A `SingleTaskGPSurrogate` with a `LinearKernel`.

    Examples:
        >>> surrogate = LinearSurrogate(inputs=inputs, outputs=outputs)
    """
    kwargs.setdefault("noise_prior", THREESIX_NOISE_PRIOR())
    kwargs.setdefault("noise_constraint", GreaterThan(lower_bound=1e-4))
    # the single-task GP search varies over RBF and Matern, which would discard the
    # linear kernel this preset exists to set
    kwargs.setdefault("hyperconfig", None)
    return SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=LinearKernel(),
        **kwargs,
    )


def PolynomialSurrogate(
    inputs: Inputs,
    outputs: Outputs,
    power: int = 2,
    **kwargs,
) -> SingleTaskGPSurrogate:
    """Build a single-task GP restricted to polynomial responses of a fixed degree.

    The polynomial kernel expresses curvature and interactions between inputs, but only
    up to `power`, so the fit stays interpretable where an RBF kernel would take an
    arbitrary shape. Pick it when a response surface of a known order is expected, as in
    a classical DoE.

    Args:
        inputs: Input features the surrogate acts on.
        outputs: Output feature the surrogate predicts.
        power: Degree of the polynomial response.
        **kwargs: Any other field of `SingleTaskGPSurrogate`. `noise_prior`,
            `noise_constraint` and `hyperconfig` default to values suited to a
            polynomial kernel rather than to the ones the GP itself defaults to.

    Returns:
        A `SingleTaskGPSurrogate` with a `PolynomialKernel`.

    Examples:
        >>> surrogate = PolynomialSurrogate(inputs=inputs, outputs=outputs, power=3)
    """
    kwargs.setdefault("noise_prior", THREESIX_NOISE_PRIOR())
    kwargs.setdefault("noise_constraint", GreaterThan(lower_bound=1e-4))
    # the single-task GP search varies over RBF and Matern, which would discard the
    # polynomial kernel this preset exists to set
    kwargs.setdefault("hyperconfig", None)
    return SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=PolynomialKernel(power=power),
        **kwargs,
    )
