"""Surrogate presets.

A preset is a named configuration of an existing surrogate rather than a surrogate in
its own right. It is a function, not a data model, so what it returns serializes as the
surrogate it configures -- there is no extra type to register, and a caller can always
reach the same result by hand.

The same split applies to the priors, where `THREESIX_LENGTHSCALE_PRIOR` and its
siblings are `partial`s over `GammaPrior` rather than classes.

Each preset names explicitly the fields whose default it *changes*, plus the
hyperparameters of the kernel it fixes, so the signature shows what the preset decides.
Everything else is forwarded to the surrogate untouched, which is deliberate: pydantic
deep-copies a field default, while a live object in a function signature would be
shared across every call.

Restricting the fixed kernel to a subset of the inputs is the one thing a preset cannot
express. Build the surrogate directly for that.
"""

from typing import Optional

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import LinearKernel, PolynomialKernel
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, AnyPrior
from bofire.data_models.surrogates.single_task_gp import (
    SingleTaskGPHyperconfig,
    SingleTaskGPSurrogate,
)


def LinearSurrogate(
    inputs: Inputs,
    outputs: Outputs,
    variance_prior: Optional[AnyPrior] = None,
    noise_prior: Optional[AnyPrior] = None,
    hyperconfig: Optional[SingleTaskGPHyperconfig] = None,
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
        variance_prior: Prior over the linear kernel's variance, which sets how large a
            slope the model expects. Defaults to none, leaving it unconstrained.
        noise_prior: Prior over the observation noise. Defaults to the three-six gamma
            prior rather than to the log-normal one a single-task GP would use.
        hyperconfig: Configuration of a hyperparameter optimization. Defaults to none,
            because the single-task GP config varies over RBF and Matern and would
            discard the linear kernel.
        **kwargs: Any remaining field of `SingleTaskGPSurrogate`, forwarded unchanged,
            so its own defaults apply.

    Returns:
        A `SingleTaskGPSurrogate` with a `LinearKernel`.

    Examples:
        >>> surrogate = LinearSurrogate(inputs=inputs, outputs=outputs)
    """
    return SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=LinearKernel(variance_prior=variance_prior),
        noise_prior=noise_prior if noise_prior is not None else THREESIX_NOISE_PRIOR(),
        hyperconfig=hyperconfig,
        **kwargs,
    )


def PolynomialSurrogate(
    inputs: Inputs,
    outputs: Outputs,
    power: int = 2,
    offset_prior: Optional[AnyPrior] = None,
    noise_prior: Optional[AnyPrior] = None,
    hyperconfig: Optional[SingleTaskGPHyperconfig] = None,
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
        offset_prior: Prior over the polynomial kernel's offset, which sets how much
            weight the lower-order terms carry. Defaults to none, leaving it
            unconstrained.
        noise_prior: Prior over the observation noise. Defaults to the three-six gamma
            prior rather than to the log-normal one a single-task GP would use.
        hyperconfig: Configuration of a hyperparameter optimization. Defaults to none,
            because the single-task GP config varies over RBF and Matern and would
            discard the polynomial kernel.
        **kwargs: Any remaining field of `SingleTaskGPSurrogate`, forwarded unchanged,
            so its own defaults apply.

    Returns:
        A `SingleTaskGPSurrogate` with a `PolynomialKernel`.

    Examples:
        >>> surrogate = PolynomialSurrogate(inputs=inputs, outputs=outputs, power=3)
    """
    return SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=PolynomialKernel(power=power, offset_prior=offset_prior),
        noise_prior=noise_prior if noise_prior is not None else THREESIX_NOISE_PRIOR(),
        hyperconfig=hyperconfig,
        **kwargs,
    )
