import math
from typing import Callable, Sequence

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor

from bofire.data_models.constraints.condition import (
    Condition,
    NonZeroCondition,
    SelectionCondition,
    ThresholdCondition,
    threshold_operators,
)
from bofire.data_models.kernels.conditional import (
    ConditionalEmbeddingKernel as ConditionalEmbeddingKernelDataModel,
)


IndicatorFunction = Callable[[Tensor], Tensor]


def _conditional_feature_to_indicator(
    cond_tuple: tuple[str, str, Condition],
    features_to_idx_mapper: Callable[[list[str]], list[int]],
) -> tuple[int, IndicatorFunction]:
    """Get the indicator function from the conditional feature.

    Returns a tuple of (`idx`, `indicator_function`), where `idx` is the feature index
    of the conditional feature.
    """
    feat_key, indicator_feat_key, condition = cond_tuple

    feat_idx, indicator_feat_idx = features_to_idx_mapper(
        [feat_key, indicator_feat_key]
    )

    if isinstance(condition, SelectionCondition):
        # FIXME: this assumes that the values are appropriately encoded.
        # In practice, this will not be the case for CategoricalInputs. However, it
        # isn't easy to pass the indicator feature to this function. A fix may be
        # to create a new EncodedCondition class, that operates in the encoded
        # feature space.
        values_t = torch.tensor(condition.selection)
        return feat_idx, lambda X: torch.isin(X[..., indicator_feat_idx], values_t)

    if isinstance(condition, ThresholdCondition):
        op = threshold_operators[condition.operator]
        return feat_idx, lambda X: op(X[..., indicator_feat_idx], condition.threshold)

    if isinstance(condition, NonZeroCondition):
        return feat_idx, lambda X: X[..., indicator_feat_idx] != 0

    raise ValueError(f"Unrecognised condition {condition.__class__.__name__}.")


def build_indicator_func(
    conditions: Sequence[tuple[str, str, Condition]],
    features_to_idx_mapper: Callable[[list[str]], list[int]] | None,
) -> IndicatorFunction:
    if not conditions:
        return lambda X: torch.ones_like(X, dtype=torch.bool)

    if features_to_idx_mapper is None:
        raise RuntimeError(
            "features_to_idx_mapper must be defined when using a conditional kernel"
        )

    thresholds = [
        _conditional_feature_to_indicator(cond_tuple, features_to_idx_mapper)  # type: ignore
        for cond_tuple in conditions
    ]

    def indicator_func(X: Tensor) -> Tensor:
        mask = torch.ones_like(X, dtype=torch.bool)
        # evaluate threshold constraints
        for idx, indicator in thresholds:
            mask[..., idx] *= indicator(X)
        return mask

    return indicator_func


def compute_base_kernel_active_dims(
    data_model: ConditionalEmbeddingKernelDataModel,
    active_dims: list[int],
    features_to_idx_mapper: Callable[[list[str]], list[int]] | None,
) -> list[int]:
    """Compute the active dimensions for the base_kernel of a conditional kernel.

    The conditional kernel expands the input space by a factor of 2, by embedding each
    input to a point in 2D. We therefore must drop any of the new dimensions that are
    appended but not

    This also removes any variables that were used as indicators."""
    if not active_dims:
        # since we append the embedded features on the right of the input features,
        # we need to know the number of features passed to the kernel
        raise ValueError(
            "`active_dims` must not be empty when using conditional kernels."
        )
    if not data_model.conditions:
        # if there are no conditions, then no embedding takes place - simply return
        # the active dimensions; this check prevents raising an error if there are
        # no conditions, and `feature_to_idx_mapper` is None
        return active_dims

    if features_to_idx_mapper is None:
        raise ValueError(
            "features_to_idx_mapper must be defined when using conditional kernels."
        )

    base_kernel_data_model = data_model.base_kernel
    embedded_idcs = features_to_idx_mapper([tup[0] for tup in data_model.conditions])

    d = len(active_dims)

    if base_kernel_data_model.features:
        active_dims = features_to_idx_mapper(base_kernel_data_model.features)

    embedded_dims = [i + d for i in active_dims if i in embedded_idcs]
    return active_dims + embedded_dims


class WedgeKernel(Kernel):
    r"""Compute the Wedge kernel for conditional features.

    This is similar to the ArcKernel provided by GPyTorch. For an exploration on the
    benefits of the Wedge kernel, see Horn et al. "Surrogates for hierarchical search
    spaces: the wedge-kernel and an automated analysis"
    URL: (https://dl.acm.org/doi/pdf/10.1145/3321707.3321765)

    This implementation differs from the paper above, since we "rotate" the kernel,
    allowing more meaningful priors on the kernel hyperparameters.
    """

    has_lengthscale = True

    def __init__(
        self,
        base_kernel: Kernel,
        indicator_func: Callable[[Tensor], Tensor],
        angle_prior: Prior | None = None,
        radius_prior: Prior | None = None,
        use_rotated_embedding: bool = True,
        **kwargs,
    ):
        """
        Args:
            base_kernel: The Kernel that operates in the embedded space
            indicator_func: A function that indicates which features are active.
                Given a `batch_shape x n x d` tensor, this should return a
                `batch_shape x n x d` binary tensor mask, indicating which features
                are active.
        """
        super().__init__(has_lengthscale=True, **kwargs)

        if self.ard_num_dims is None:
            self.last_dim = 1
        else:
            self.last_dim = self.ard_num_dims

        self.indicator_func = indicator_func

        self.register_parameter(
            name="raw_angle",
            parameter=torch.nn.Parameter(
                torch.zeros(*self.batch_shape, 1, self.last_dim)
            ),
        )
        if angle_prior is not None:
            if not isinstance(angle_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(angle_prior).__name__
                )
            self.register_prior(
                "angle_prior",
                angle_prior,
                lambda m: m.angle,
                lambda m, v: m._set_angle(v),
            )

        self.register_constraint("raw_angle", Interval(1e-4, 1 - 1e-4))

        radii_per_dim = 1 if use_rotated_embedding else 2
        self.use_rotated_embedding = use_rotated_embedding
        self.register_parameter(
            name="raw_radius",
            parameter=torch.nn.Parameter(
                torch.zeros(*self.batch_shape, 1, radii_per_dim * self.last_dim)
            ),
        )

        if radius_prior is not None:
            if not isinstance(radius_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(radius_prior).__name__
                )
            self.register_prior(
                "radius_prior",
                radius_prior,
                lambda m: m.radius,
                lambda m, v: m._set_radius(v),
            )

        self.register_constraint("raw_radius", Positive())

        self.base_kernel = base_kernel
        if self.base_kernel.has_lengthscale:
            self.base_kernel.lengthscale = torch.tensor(1.0)
            self.base_kernel.raw_lengthscale.requires_grad_(False)

    @property
    def angle(self):
        return self.raw_angle_constraint.transform(self.raw_angle)  # type: ignore

    @angle.setter
    def angle(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_angle)  # type: ignore
        self.initialize(raw_angle=self.raw_angle_constraint.inverse_transform(value))  # type: ignore

    @property
    def radius(self):
        return self.raw_radius_constraint.transform(self.raw_radius)  # type: ignore

    @radius.setter
    def radius(self, value):
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_radius)  # type: ignore
        self.initialize(raw_radius=self.raw_radius_constraint.inverse_transform(value))  # type: ignore

    def embedding(self, x):
        # this assumes that x has been normalized to [0, 1]
        mask = self.indicator_func(x)
        x_ = x.div(self.lengthscale)
        l_1 = self.radius[..., 0]
        l_2 = self.radius[..., 1]

        x_1 = (l_1 + x * (l_2 * torch.cos(math.pi * self.angle) - l_1)) * mask
        x_2 = (x * l_2 * torch.sin(math.pi * self.angle)) * mask

        x_ = torch.cat((x_1, x_2), dim=-1)
        return x_

    def rotated_embedding(self, x):
        # we rotate the embedding, which means we can place a prior directly
        # on the lengthscale
        # this assumes that x has been normalized to [0, 1]
        mask = self.indicator_func(x)
        x_ = x.div(self.lengthscale)
        midpoint = 0.5 / self.lengthscale
        ell = self.radius[..., 0]

        x_1 = torch.where(
            mask.bool(), x_, midpoint + torch.cos(math.pi * self.angle) * ell
        )
        x_2 = torch.where(
            mask.bool(), torch.zeros_like(x_), torch.sin(math.pi * self.angle) * ell
        )

        x_ = torch.cat((x_1, x_2), dim=-1)
        return x_

    def forward(
        self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False, **params
    ):
        if self.use_rotated_embedding:
            x1_, x2_ = self.rotated_embedding(x1), self.rotated_embedding(x2)
        else:
            x1_, x2_ = self.embedding(x1), self.embedding(x2)
        return self.base_kernel(
            x1_, x2_, diag=diag, last_dim_is_batch=last_dim_is_batch
        )

    @property
    def is_stationary(self):
        return False
