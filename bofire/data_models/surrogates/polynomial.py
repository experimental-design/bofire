from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.kernels.api import PolynomialKernel
from bofire.data_models.priors.api import THREESIX_NOISE_PRIOR, GreaterThan
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate


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
