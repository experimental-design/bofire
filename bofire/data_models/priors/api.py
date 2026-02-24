from functools import partial
from typing import List, Type, Union

from bofire.data_models.priors.constraint import (
    GreaterThan,
    LessThan,
    Positive,
    PriorConstraint,
)
from bofire.data_models.priors.gamma import GammaPrior
from bofire.data_models.priors.interval import (
    Interval,
    LogTransformedInterval,
    NonTransformedInterval,
)
from bofire.data_models.priors.lkj import LKJPrior
from bofire.data_models.priors.normal import (
    DimensionalityScaledLogNormalPrior,
    LogNormalPrior,
    NormalPrior,
)
from bofire.data_models.priors.prior import Prior


AbstractPrior = Prior
AbstractPriorConstraint = Union[PriorConstraint, Interval]

_PRIOR_TYPES: List[Type[Prior]] = [
    GammaPrior,
    NormalPrior,
    LKJPrior,
    LogNormalPrior,
    DimensionalityScaledLogNormalPrior,
]

AnyPrior = Union[tuple(_PRIOR_TYPES)]

_PRIOR_CONSTRAINT_TYPES: List[Type] = [
    NonTransformedInterval,
    LogTransformedInterval,
    Positive,
    GreaterThan,
    LessThan,
]

AnyPriorConstraint = Union[tuple(_PRIOR_CONSTRAINT_TYPES)]


def _rebuild_dependent_models() -> None:
    """Rebuild all Pydantic models whose fields reference AnyPrior or AnyPriorConstraint."""
    from bofire.data_models._register_utils import patch_field

    # Lazy imports to avoid circular dependencies
    from bofire.data_models.kernels.aggregation import (
        AdditiveKernel,
        MultiplicativeKernel,
        PolynomialFeatureInteractionKernel,
        ScaleKernel,
    )
    from bofire.data_models.kernels.categorical import (
        HammingDistanceKernel,
        IndexKernel,
        PositiveIndexKernel,
    )
    from bofire.data_models.kernels.conditional import WedgeKernel
    from bofire.data_models.kernels.continuous import (
        MaternKernel,
        RBFKernel,
        SphericalLinearKernel,
    )
    from bofire.data_models.kernels.shape import WassersteinKernel
    from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
    from bofire.data_models.surrogates.linear import LinearSurrogate
    from bofire.data_models.surrogates.mixed_single_task_gp import (
        MixedSingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPSurrogate
    from bofire.data_models.surrogates.polynomial import PolynomialSurrogate
    from bofire.data_models.surrogates.robust_single_task_gp import (
        RobustSingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.shape import PiecewiseLinearGPSurrogate
    from bofire.data_models.surrogates.single_task_gp import (
        SingleTaskGPHyperconfig,
        SingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate

    # Patch AnyPrior fields
    for model_cls, field_name in [
        (RBFKernel, "lengthscale_prior"),
        (MaternKernel, "lengthscale_prior"),
        (SphericalLinearKernel, "lengthscale_prior"),
        (HammingDistanceKernel, "lengthscale_prior"),
        (IndexKernel, "prior"),
        (PositiveIndexKernel, "prior"),
        (PositiveIndexKernel, "task_prior"),
        (PositiveIndexKernel, "diag_prior"),
        (WedgeKernel, "lengthscale_prior"),
        (WedgeKernel, "angle_prior"),
        (WedgeKernel, "radius_prior"),
        (WassersteinKernel, "lengthscale_prior"),
        (SingleTaskGPSurrogate, "noise_prior"),
        (MultiTaskGPSurrogate, "noise_prior"),
        (MixedSingleTaskGPSurrogate, "noise_prior"),
        (TanimotoGPSurrogate, "noise_prior"),
        (PiecewiseLinearGPSurrogate, "outputscale_prior"),
        (PiecewiseLinearGPSurrogate, "noise_prior"),
        (PolynomialSurrogate, "noise_prior"),
        (LinearSurrogate, "noise_prior"),
        (RobustSingleTaskGPSurrogate, "noise_prior"),
    ]:
        patch_field(model_cls, field_name, AnyPrior)

    # Patch AnyPriorConstraint fields
    for model_cls, field_name in [
        (RBFKernel, "lengthscale_constraint"),
        (MaternKernel, "lengthscale_constraint"),
        (SphericalLinearKernel, "lengthscale_constraint"),
        (HammingDistanceKernel, "lengthscale_constraint"),
        (IndexKernel, "var_constraint"),
        (PositiveIndexKernel, "var_constraint"),
        (WedgeKernel, "lengthscale_constraint"),
        (ScaleKernel, "outputscale_constraint"),
        (SingleTaskGPHyperconfig, "lengthscale_constraint"),
        (SingleTaskGPHyperconfig, "outputscale_constraint"),
    ]:
        patch_field(model_cls, field_name, AnyPriorConstraint)

    # Rebuild in dependency order:
    # 1. Leaf kernel models
    for cls in [
        RBFKernel,
        MaternKernel,
        SphericalLinearKernel,
        HammingDistanceKernel,
        IndexKernel,
        PositiveIndexKernel,
        WassersteinKernel,
    ]:
        cls.model_rebuild(force=True)

    # 2. Conditional kernels (reference leaf kernels)
    WedgeKernel.model_rebuild(force=True)

    # 3. Aggregation kernels (embed leaf kernel types)
    for cls in [
        AdditiveKernel,
        MultiplicativeKernel,
        ScaleKernel,
        PolynomialFeatureInteractionKernel,
    ]:
        cls.model_rebuild(force=True)

    # 4. Surrogate models
    SingleTaskGPHyperconfig.model_rebuild(force=True)
    for cls in [
        SingleTaskGPSurrogate,
        MultiTaskGPSurrogate,
        MixedSingleTaskGPSurrogate,
        TanimotoGPSurrogate,
        PiecewiseLinearGPSurrogate,
        PolynomialSurrogate,
        LinearSurrogate,
        RobustSingleTaskGPSurrogate,
    ]:
        cls.model_rebuild(force=True)

    # 5. BotorchSurrogates
    BotorchSurrogates.model_rebuild(force=True)


def register_prior(data_model_cls: Type[Prior]) -> None:
    """Register a custom prior type so it is accepted in AnyPrior fields.

    This appends the type to the internal registry, rebuilds the
    ``AnyPrior`` union, and calls ``model_rebuild`` on all dependent
    Pydantic models (kernels, surrogates) so that the new type is accepted.

    Args:
        data_model_cls: A concrete subclass of ``Prior``.
    """
    global AnyPrior
    if data_model_cls in _PRIOR_TYPES:
        return
    _PRIOR_TYPES.append(data_model_cls)
    AnyPrior = Union[tuple(_PRIOR_TYPES)]
    _rebuild_dependent_models()


def register_prior_constraint(data_model_cls: Type) -> None:
    """Register a custom prior constraint type so it is accepted in AnyPriorConstraint fields.

    This appends the type to the internal registry, rebuilds the
    ``AnyPriorConstraint`` union, and calls ``model_rebuild`` on all
    dependent Pydantic models.

    Args:
        data_model_cls: A concrete subclass of ``PriorConstraint`` or ``Interval``.
    """
    global AnyPriorConstraint
    if data_model_cls in _PRIOR_CONSTRAINT_TYPES:
        return
    _PRIOR_CONSTRAINT_TYPES.append(data_model_cls)
    AnyPriorConstraint = Union[tuple(_PRIOR_CONSTRAINT_TYPES)]
    _rebuild_dependent_models()


# these are priors that are generally applicable
# and do not depend on problem specific extra parameters
AnyGeneralPrior = Union[GammaPrior, NormalPrior, LKJPrior, LogNormalPrior]

# default priors of interest
# botorch defaults
THREESIX_LENGTHSCALE_PRIOR = partial(GammaPrior, concentration=3.0, rate=6.0)
THREESIX_NOISE_PRIOR = partial(GammaPrior, concentration=1.1, rate=0.05)
THREESIX_SCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.15)

# mbo priors
# By default BoTorch places a highly informative prior on the kernel lengthscales,
# which easily leads to overfitting. Here we set a broader prior distribution for the
# lengthscale. The priors for the noise and signal variance are set more tightly.
MBO_LENGTHSCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=0.2)
MBO_NOISE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)
MBO_OUTPUTSCALE_PRIOR = partial(GammaPrior, concentration=2.0, rate=4.0)

# prior for multitask kernel
LKJ_PRIOR = partial(
    LKJPrior,
    shape=2.0,
    sd_prior=GammaPrior(concentration=2.0, rate=0.15),
)

# prior for RobustSingleTaskGPSurrogate
ROBUSTGP_LENGTHSCALE_CONSTRAINT = partial(
    NonTransformedInterval,
    lower_bound=0.05,
    upper_bound=float("inf"),
    initial_value=0.2,
)

ROBUSTGP_OUTPUTSCALE_CONSTRAINT = partial(
    NonTransformedInterval,
    lower_bound=0.01,
    upper_bound=10.0,
    initial_value=0.1,
)

# Hvarfner priors
HVARFNER_NOISE_PRIOR = partial(LogNormalPrior, loc=-4, scale=1)
HVARFNER_LENGTHSCALE_PRIOR = DimensionalityScaledLogNormalPrior

# EDBO priors:
# adapted from the EDBO paper https://github.com/b-shields/edbo/blob/master/edbo/bro.py#L664
# and code https://doi.org/10.1038/s41586-021-03213-y
# EDBO also define starting values for the hyperparameters, which are currently not supported
# in BoFire. We provide it here as a comment behind the prior for reference.
EDBO_MORDRED_LENGTHSCALE_PRIOR = partial(
    GammaPrior, concentration=2.0, rate=0.1
)  # starting value 10.0
EDBO_MORDRED_OUTPUT_SCALE_PRIOR = partial(
    GammaPrior, concentration=2.0, rate=0.1
)  # starting value 10.0
EDBO_MORDRED_NOISE_PRIOR = partial(
    GammaPrior, concentration=1.5, rate=0.1
)  # starting value 5.0

EDBO_DFT_LENGTHSCALE_PRIOR = partial(
    GammaPrior, concentration=2.0, rate=0.2
)  # starting value 5.0
EDBO_DFT_OUTPUT_SCALE_PRIOR = partial(
    GammaPrior, concentration=5.0, rate=0.5
)  # starting value 8.0
EDBO_DFT_NOISE_PRIOR = partial(
    GammaPrior, concentration=1.5, rate=0.1
)  # starting value 5.0

EDBO_OHE_LENGTHSCALE_PRIOR = partial(
    GammaPrior, concentration=3.0, rate=1.0
)  # starting value 2.0
EDBO_OHE_OUTPUT_SCALE_PRIOR = partial(
    GammaPrior, concentration=5.0, rate=0.2
)  # starting value 20.0
EDBO_OHE_NOISE_PRIOR = partial(
    GammaPrior, concentration=1.5, rate=0.1
)  # starting value 5.0
