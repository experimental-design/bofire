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

from typing import Optional

import torch

from bofire.data_models.features.api import CategoricalMolecularInput
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.kernels.fingerprint_kernels.base_fingerprint_kernel import BitKernel
from bofire.utils.cheminformatics import mutual_tanimoto_similarities
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

    def __init__(
        self,
        pre_compute_similarities: bool = False,
        molecular_inputs: Optional[list[CategoricalMolecularInput]] = None,
        fingerprint_settings: Optional[dict[str, Fingerprints]] = None,
        computed_mutual_similarities: Optional[dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.metric = "tanimoto"

        self.pre_compute_similarities = pre_compute_similarities
        self.fingerprint_settings = fingerprint_settings if fingerprint_settings else {}
        self.molecular_inputs = molecular_inputs if molecular_inputs else []
        self.sim_matrices = (
            computed_mutual_similarities if computed_mutual_similarities else {}
        )

        if self.pre_compute_similarities:
            for inp_ in self.molecular_inputs:
                key = inp_.key
                if key not in list(self.sim_matrices):
                    fingerprint = (
                        self.fingerprint_settings[key]
                        if key in list(self.fingerprint_settings)
                        else Fingerprints()
                    )
                    self.sim_matrices[key] = self.compute_sim_matrix(inp_, fingerprint)

    @staticmethod
    def compute_sim_matrix(
        input: CategoricalMolecularInput,
        fingerprint: Fingerprints,
    ) -> torch.Tensor:
        """loop over combinations of molecules, and put this in a torch 2D array"""

        distances = mutual_tanimoto_similarities(
            input.categories, **fingerprint.model_dump(exclude=["type"])
        )

        n = len(input.categories)
        m = n * (n - 1) // 2
        if len(distances) != m:
            raise ValueError(
                f"Expected {m} distances for n={n}, but got {len(distances)}. "
                "Ensure you used itertools.combinations in the same order."
            )

        D = torch.ones((n, n), **tkwargs)
        rows, cols = torch.triu_indices(n, n, offset=1)  # indices where i < j
        D[rows, cols] = torch.tensor(distances, **tkwargs)
        # Mirror to lower triangle
        D[cols, rows] = D[rows, cols]
        # Diagonal remains 0 (distance of a point to itself)
        return D

    @property
    def re_init_kwargs(self) -> dict:
        if not self.pre_compute_similarities:
            return {}
        return {"computed_mutual_similarities": self.sim_matrices}

    def forward(self, x1, x2, diag=False, **params):
        if self.pre_compute_similarities:
            # Infer shapes
            batch_shape = x1.shape[:-2]
            n1, d = x1.shape[-2], x1.shape[-1]
            n2 = x2.shape[-2]
            assert (
                d == len(self.molecular_inputs)
            ), f"Last dim d={d} must match number of molecular inputs={len(self.molecular_inputs)}"

            cov = torch.zeros((*batch_shape, n1, n2), **tkwargs)

            # Sum contributions for each feature index along the last dim
            for idx, inp_ in enumerate(self.molecular_inputs):
                D = self.sim_matrices[
                    inp_.key
                ]  # [Ni, Ni], precomputed distances for feature idx

                # Gather integer indices for this feature from x1 and x2 (keep batch dims)
                x1_idx = (
                    x1[..., idx].to(torch.long).to(D.device)
                )  # shape: batch_shape × n1
                x2_idx = (
                    x2[..., idx].to(torch.long).to(D.device)
                )  # shape: batch_shape × n2

                # Build submatrix via broadcasting advanced indexing:
                # Result shape: batch_shape × n1 × n2
                D_sub = D[x1_idx.unsqueeze(-1), x2_idx.unsqueeze(-2)]

                cov = cov + D_sub

            if diag:
                # Return diagonal along the last two dims: shape batch_shape × n1
                return cov.diagonal(dim1=-2, dim2=-1)
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
