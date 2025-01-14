from typing import Dict

import torch
from botorch.models.transforms.input import OneHotToNumeric
from gpytorch.kernels.kernel import Kernel
from torch import Tensor


class HammingKernelWithOneHots(Kernel):
    r"""
    A Kernel for one-hot enocded categorical features. The inputs
    may contain more than one categorical feature.

    This kernel mimics the functionality of CategoricalKernel from
    botorch, but assumes categorical features encoded as one-hot variables.
    Computes `exp(-dist(x1, x2) / lengthscale)`, where
    `dist(x1, x2)` is zero if `x1` and `x2` correspond to the
    same category, and one otherwise. If the last dimension
    is not a batch dimension, then the mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def __init__(self, categorical_features: Dict[int, int], *args, **kwargs):
        """
        Initialize.

        Args:
            categorical_features: A dictionary mapping the starting index of each
                categorical feature to its cardinality. This assumes that categoricals
                are one-hot encoded.
            *args, **kwargs: Passed to gpytorch.kernels.kernel.Kernel.__init__
        """
        super().__init__(*args, **kwargs)

        onehot_dim = sum(categorical_features.values())
        self.trx = OneHotToNumeric(
            onehot_dim, categorical_features=categorical_features
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        x1 = self.trx(x1)
        x2 = self.trx(x2)

        delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        if self.ard_num_dims is not None:
            # botorch forces ard_num_dims to be the same as the total size of the of one-hot encoded features
            # however here we just need one length scale per categorical feature
            ls = self.lengthscale[..., : delta.shape[-1]]
        else:
            ls = self.lengthscale

        dists = delta / ls.unsqueeze(-2)
        if last_dim_is_batch:
            dists = dists.transpose(-3, -1)
        else:
            dists = dists.mean(-1)
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res
