import math
from typing import Callable

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor


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
