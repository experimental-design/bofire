"""This module was copied from the GAUCHE library(https://github.com/leojklarner/gauche/blob/main/gauche/kernels/fingerprint_kernels/tanimoto_kernel.py).

GAUCHE was published under the following license (https://github.com/leojklarner/gauche/blob/main/LICENSE):

MIT License

Copyright (c) 2021 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch

from bofire.data_models.features.api import CategoricalMolecularInput
from bofire.kernels.fingerprint_kernels.base_fingerprint_kernel import BitKernel
from bofire.utils.torch_tools import tkwargs

class TanimotoKernel(BitKernel):
    r"""Computes a covariance matrix based on the Tanimoto kernel between inputs `x1` and `x2`:

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

    def __init__(self, pre_compute_distances: bool = False, molecular_inputs: list[CategoricalMolecularInput] = None,
                 computed_mutual_distances: dict[str, list[float]] = None,
                 **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.metric = "tanimoto"

        self.pre_compute_distances = pre_compute_distances

        if self.pre_compute_distances:
            self._molecular_inputs = molecular_inputs
            self.pre_compute_distances = {
                input_.key: self.distance_matrix(input_, computed_mutual_distances[input_.key]) \
                for input_ in molecular_inputs
            }

    def distance_matrix(self, input: CategoricalMolecularInput, distances: list[float]) -> torch.Tensor:
        n = len(input.categories)
        m = n * (n - 1) // 2
        if len(distances) != m:
            raise ValueError(
                f"Expected {m} distances for n={n}, but got {len(distances)}. "
                "Ensure you used itertools.combinations in the same order."
            )

        D = torch.zeros((n, n), **tkwargs)
        rows, cols = torch.triu_indices(n, n, offset=1)  # indices where i < j
        D[rows, cols] = torch.tensor(distances, **tkwargs)
        # Mirror to lower triangle
        D[cols, rows] = D[rows, cols]
        # Diagonal remains 0 (distance of a point to itself)
        return D

    def forward(self, x1, x2, diag=False, **params):
        if self.pre_compute_distances:
            cov = torch.zeros((x1.shape[0], x2.shape[0]))
            for idx, inp_ in enumerate(self._molecular_inputs):
                D = self.pre_compute_distances[inp_.key]
                x1_, x2_ = [x[:, idx].to(torch.long).to(D.device) for x in (x1, x2)]
                D_sub = D[x1_][:, x2_]
                cov += D_sub
            return cov
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2],
                x1.shape[-2],
                dtype=x1.dtype,
                device=x1.device,
            )
        return self.covar_dist(x1, x2, **params)
