import torch
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import OneHotToNumeric

from bofire.kernels.categorical import HammingKernelWithOneHots


def test_hamming_with_one_hot_one_feature():
    cat = {0: 3}

    k1 = CategoricalKernel()
    k2 = HammingKernelWithOneHots(categorical_features=cat)

    xin_oh = torch.eye(3)
    xin_cat = OneHotToNumeric(3, categorical_features=cat).transform(xin_oh)

    z1 = k1(xin_cat).to_dense()
    z2 = k2(xin_oh).to_dense()

    assert z1.shape == z2.shape == (3, 3)
    assert torch.allclose(z1, z2)


def test_hamming_with_one_hot_two_features():
    cat = {0: 2, 2: 4}

    k1 = CategoricalKernel()
    k2 = HammingKernelWithOneHots(categorical_features=cat)

    xin_oh = torch.zeros(4, 6)
    xin_oh[:2, :2] = xin_oh[2:, :2] = torch.eye(2)
    xin_oh[:, 2:] = torch.eye(4)

    xin_cat = OneHotToNumeric(6, categorical_features=cat).transform(xin_oh)

    z1 = k1(xin_cat).to_dense()
    z2 = k2(xin_oh).to_dense()

    assert z1.shape == z2.shape == (4, 4)
    assert torch.allclose(z1, z2)


def test_hamming_with_one_hot_two_features_and_lengthscales():
    cat = {0: 2, 2: 4}

    k1 = CategoricalKernel(ard_num_dims=2)
    k1.lengthscale = torch.tensor([1.5, 3.0])

    # botorch will check that the lengthscale for ARD has the same number of elements as the one-hotted inputs,
    # so we have to specify the ard_num_dims accordingly. The kernel will make sure to only use the right
    # number of elements, corresponding to the number of categorical features.
    k2 = HammingKernelWithOneHots(categorical_features=cat, ard_num_dims=6)
    k2.lengthscale = torch.tensor([1.5, 3.0, 0.0, 0.0, 0.0, 0.0])

    xin_oh = torch.zeros(4, 6)
    xin_oh[:2, :2] = xin_oh[2:, :2] = torch.eye(2)
    xin_oh[:, 2:] = torch.eye(4)

    xin_cat = OneHotToNumeric(6, categorical_features=cat).transform(xin_oh)

    z1 = k1(xin_cat).to_dense()
    z2 = k2(xin_oh).to_dense()

    assert z1.shape == z2.shape == (4, 4)
    assert torch.allclose(z1, z2)


def test_feature_order():
    x1_in = torch.zeros(4, 2)
    x1_in[:2, :] = x1_in[2:, :] = torch.eye(2)
    x2_in = torch.eye(4)

    k1 = HammingKernelWithOneHots(categorical_features={0: 2, 2: 4})
    k2 = HammingKernelWithOneHots(categorical_features={0: 4, 4: 2})

    z1 = k1(torch.cat([x1_in, x2_in], dim=1)).to_dense()
    z2 = k2(torch.cat([x2_in, x1_in], dim=1)).to_dense()

    assert z1.shape == z2.shape == (4, 4)
    assert torch.allclose(z1, z2)
