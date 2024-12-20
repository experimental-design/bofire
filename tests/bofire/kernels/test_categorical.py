import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import OneHotToNumeric

from bofire.kernels.categorical import HammingKernelWithOneHots


def test_hamming_with_one_hot():
    k1 = CategoricalKernel()
    k2 = HammingKernelWithOneHots()

    xin_oh = torch.eye(3)
    xin_cat = OneHotToNumeric(3, categorical_features={0: 3}).transform(xin_oh)

    z1 = k1(xin_cat).to_dense()
    z2 = k2(xin_oh).to_dense()

    assert torch.allclose(z1, z2)
