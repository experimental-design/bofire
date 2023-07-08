"""
Tanimoto Kernel. Operates on representations including bit vectors e.g. Morgan/ECFP6 fingerprints count vectors e.g.
RDKit fragment features.
"""
# This code was copied from GAUCHE: https://github.com/leojklarner/gauche/blob/main/gauche/kernels/fingerprint_kernels/tanimoto_kernel.py

import torch

from bofire.kernels.fingerprint_kernels.base_fingerprint_kernel import BitKernel


class TanimotoKernel(BitKernel):
    r"""
    Computes a covariance matrix based on the Tanimoto kernel between inputs `x1` and `x2`:

    Formula:
        .. math::

        \begin{equation*}
        k_{\text{Tanimoto}}(\mathbf{x}, \mathbf{x'}) = \frac{\langle\mathbf{x},
        \mathbf{x'}\rangle}{\left\lVert\mathbf{x}\right\rVert^2 + \left\lVert\mathbf{x'}\right\rVert^2 -
        \langle\mathbf{x}, \mathbf{x'}\rangle}
        \end{equation*}

    This kernel does not have an `outputscale` parameter. To add a scaling parameter,
    decorate this kernel with a `gpytorch.test_kernels.ScaleKernel`.

    Example:
        >>> x = torch.randint(0, 2, (10, 5))
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
        >>>
        >>> batch_x = torch.randint(0, 2, (2, 10, 5))
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """
    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.metric = "tanimoto"

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)
