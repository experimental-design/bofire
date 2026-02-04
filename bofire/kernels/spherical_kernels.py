"""
This module implements spherical kernels for Gaussian processes.
The implementation is inspired from: https://github.com/colmont/linear-bo/blob/main/src/kernels/spherical_linear.py
"""

from typing import List, Union

import gpytorch
import torch
from linear_operator.operators import (
    LowRankRootLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
)


def project_onto_unit_sphere(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Project inputs onto sphere after scaling by lengthscale.

    :param x: Input tensor of shape (..., N, D) to project.
    """
    x_sq_norm = x.square().sum(dim=-1, keepdim=True)
    x_ = torch.cat(
        [2 * x, (x_sq_norm - 1.0)], dim=-1
    ).mul(  # inverse stereographic projection
        1.0 / (1.0 + x_sq_norm)
    )
    return x_


def maybe_low_rank_root_lo(root: torch.Tensor) -> RootLinearOperator:
    n, r = root.shape[-2:]
    if r >= n:
        return RootLinearOperator(root)
    else:
        return LowRankRootLinearOperator(root)


class SphericalLinearKernel(gpytorch.kernels.RBFKernel):
    """
    Apply linear kernel after spherical projection.

    :param bounds: The bounds of the input space. If a single (min, max) bound is given, it is used for all dimensions.
    """

    has_lengthscale = True

    def __init__(
        self,
        bounds: Union[tuple[float, float], List[tuple[float, float]]] = (0.0, 1.0),
        **kwargs,
    ):
        # Get ard_num_dims before calling super().__init__
        ard_num_dims = kwargs.get("ard_num_dims", None)
        # Validate that we have at least 2 dimensions for spherical projection
        if ard_num_dims is not None and ard_num_dims < 2:
            raise ValueError(
                f"SphericalLinearKernel requires at least 2 dimensions. Got ard_num_dims={ard_num_dims}."
            )
        super(
            SphericalLinearKernel,
            self,
        ).__init__(**kwargs)
        # Determine number of dimensions from ard_num_dims or infer from bounds
        if self.ard_num_dims is not None:
            num_dims = self.ard_num_dims
        elif isinstance(bounds, list) and len(bounds) > 0:
            num_dims = len(bounds)
        else:
            raise ValueError(
                "Cannot determine number of dimensions. If ard=False then list of bounds should have length equal to the input dimension."
            )
        # Expand bounds if a single tuple is provided
        if isinstance(bounds[0], (int, float)):
            bounds = [(bounds[0], bounds[1])] * num_dims  # type: ignore

        # Create buffer for the center and length of each dimension in the space
        _dtype = self.raw_lengthscale.dtype
        _bounds = torch.tensor(bounds, dtype=_dtype)  # type: ignore
        mins = _bounds[..., 0]
        maxs = _bounds[..., 1]
        centers = (mins + maxs).div(2.0)
        self.register_buffer("_mins", mins)
        self.register_buffer("_maxs", maxs)
        self.register_buffer("_centers", centers)
        self.register_buffer("_num_dims", torch.tensor(num_dims, dtype=torch.long))
        assert torch.all(self._maxs > self._mins), f"Invalid bounds {bounds}."  # type: ignore

        # When ard=False, fix the lengthscales (don't learn them)
        if self.ard_num_dims is None:
            # Make lengthscale non-trainable by removing it from parameters
            # and storing it as a buffer instead
            # Access raw_lengthscale and apply constraint to get initial value
            initial_lengthscale = (
                self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
                .detach()
                .clone()
            )
            delattr(self, "raw_lengthscale")
            if hasattr(self, "raw_lengthscale_constraint"):
                delattr(self, "raw_lengthscale_constraint")
            # Register as buffer (non-trainable)
            self.register_buffer("_fixed_lengthscale", initial_lengthscale)

            # Also fix glob_ls when ard=False - don't learn it
            self.register_buffer("_fixed_glob_ls", torch.ones(1, dtype=_dtype))
        else:
            # ARD case: keep lengthscales trainable, add learnable glob_ls
            glob_ls = torch.ones(1, dtype=_dtype)
            self.register_parameter("raw_glob_ls", torch.nn.Parameter(glob_ls))

        # Learnable coefficients "b_i" for constant and linear terms
        # These are ALWAYS learned (both ARD and non-ARD cases)
        coeffs = torch.zeros(2, dtype=_dtype)
        self.register_parameter("raw_coeffs", torch.nn.Parameter(coeffs))

    @property
    def coeffs(self) -> torch.Tensor:
        """The coefficients for the constant and linear terms"""
        return torch.nn.functional.softmax(self.raw_coeffs, dim=-1)

    @property
    def glob_ls(self) -> torch.Tensor:
        """The global lengthscale"""
        if self.ard_num_dims is None:
            # Non-ARD case: use fixed glob_ls
            return self._fixed_glob_ls
        else:
            # ARD case: use learnable glob_ls
            return torch.sigmoid(self.raw_glob_ls)

    @property
    def lengthscale(self) -> torch.Tensor:
        """Override lengthscale property to handle fixed lengthscale in non-ARD case"""
        if self.ard_num_dims is None:
            # Non-ARD case: return fixed lengthscale
            return self._fixed_lengthscale
        else:
            # ARD case: return learnable lengthscale from parent class
            return super().lengthscale

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):  # noqa: D102
        # Get constants
        lengthscale: torch.Tensor = self.lengthscale

        # Handle both ARD and non-ARD cases
        # If ard_num_dims is None, lengthscale is a scalar, so we need to broadcast it
        if self.ard_num_dims is None:
            # Non-ARD case: lengthscale is scalar, expand to match number of dimensions
            lengthscale = lengthscale.expand(*lengthscale.shape[:-1], self._num_dims)  # type: ignore

        max_sq_norm: torch.Tensor = (
            (self._maxs - self._mins)[..., None, :]  # Shape: (..., 1, D)
            .div(2.0 * lengthscale)
            .square()
            .sum(dim=-1, keepdim=True)  # Shape: (..., 1, 1)
        )
        glob_ls: torch.Tensor = torch.sqrt(
            self.glob_ls * max_sq_norm
        )  # O(\sqrt{D}) init

        # Center and scale inputs
        x1 = x1.sub(self._centers).div(lengthscale)
        x2 = x1 if torch.equal(x1, x2) else x2.sub(self._centers).div(lengthscale)

        # Apply global lengthscale
        x1 = x1.div(glob_ls)
        x2 = x2.div(glob_ls)

        # Project the inputs onto the sphere
        x1_ = project_onto_unit_sphere(x1)
        x2_ = project_onto_unit_sphere(x2)

        # Sum up the (weighted) components for constant and linear terms
        terms = self.coeffs
        term0_sqrt = terms[0].sqrt()
        term1_sqrt = terms[1].sqrt()
        x1_ = torch.cat([x1_ * term1_sqrt, term0_sqrt.expand_as(x1_[..., :1])], dim=-1)
        if torch.equal(x1, x2):
            kernel = maybe_low_rank_root_lo(x1_)
        else:
            x2_ = torch.cat(
                [x2_ * term1_sqrt, term0_sqrt.expand_as(x2_[..., :1])], dim=-1
            )
            kernel = MatmulLinearOperator(x1_, x2_.mT)

        return kernel
