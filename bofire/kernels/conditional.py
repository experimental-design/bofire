import math
import operator as ops
from typing import Callable

import pandas as pd
import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor

from bofire.data_models.constraints.categorical import (
    SelectionCondition,
    ThresholdCondition,
)
from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import CategoricalInput, DiscreteInput
from bofire.data_models.features.conditional import ConditionalContinuousInput
from bofire.data_models.types import InputTransformSpecs


type IndicatorFunction = Callable[[Tensor], Tensor]

_threshold_operators: dict[str, Callable] = {
    "<": ops.lt,
    "<=": ops.le,
    ">": ops.gt,
    ">=": ops.ge,
}


def _conditional_feature_to_indicator(
    inputs: Inputs, feat: ConditionalContinuousInput
) -> tuple[int, IndicatorFunction]:
    """Get the indicator function from the conditional feature.

    Returns a tuple of (`idx`, `indicator_function`), where `idx` is the feature index
    of the conditional feature.
    """
    condition = feat.indicator_condition
    indicator_feat = inputs.get_by_key(feat.indicator_feature)

    feat_idx, indicator_feat_idx = inputs.get_feature_indices(
        {}, [feat.key, indicator_feat.key]
    )

    if isinstance(condition, SelectionCondition):
        values_t = torch.tensor([])
        if isinstance(indicator_feat, CategoricalInput):
            values = indicator_feat.to_ordinal_encoding(
                pd.Series(indicator_feat.categories)
            )
            values_t = torch.from_numpy(values.to_numpy())
        elif isinstance(indicator_feat, DiscreteInput):
            values_t = torch.as_tensor(indicator_feat.values)
        return feat_idx, lambda X: torch.isin(X[..., indicator_feat_idx], values_t)

    if isinstance(condition, ThresholdCondition):
        op = _threshold_operators[condition.operator]
        return feat_idx, lambda X: op(X[..., indicator_feat_idx], condition.threshold)

    raise ValueError(f"Unrecognised condition {condition.__class__.__name__}.")


def build_indicator_func(
    inputs: Inputs, specs: InputTransformSpecs
) -> IndicatorFunction:
    if (
        CategoricalEncodingEnum.ONE_HOT in specs.values()
        or CategoricalEncodingEnum.DESCRIPTOR in specs.values()
    ):
        # TODO: provide support for one hot and descriptor
        # requires careful thought about how indexing changes
        raise NotImplementedError(
            "Conditional features currently only support ordinal encoding."
        )

    thresholds = [
        _conditional_feature_to_indicator(inputs, feat)
        for feat in inputs.get(includes=ConditionalContinuousInput)
    ]  # type: ignore

    def indicator_func(X: Tensor) -> Tensor:
        mask = torch.ones_like(X, dtype=torch.bool)
        # evaluate threshold constraints
        for idx, indicator in thresholds:
            mask[..., idx] *= indicator(X)
        return mask

    return indicator_func


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
