from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import LinearKernel
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
