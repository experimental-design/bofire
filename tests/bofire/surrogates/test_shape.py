import pandas as pd
import torch
from gpytorch.kernels import MaternKernel, ProductKernel, ScaleKernel
from gpytorch.priors import GammaPrior, LogNormalPrior
from pandas.testing import assert_frame_equal
from torch.testing import assert_allclose

import bofire.surrogates.api as surrogates
import tests.bofire.data_models.specs.api as specs
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import MaternKernel as MaternKernelDataModel
from bofire.data_models.kernels.api import RBFKernel as RBFKernelDataModel

# , RBFKernel, ScaleKernel
from bofire.data_models.priors.api import (
    MBO_LENGTHCALE_PRIOR,
    MBO_NOISE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
)
from bofire.data_models.surrogates.api import PiecewiseLinearGPSurrogate
from bofire.kernels.shape import WassersteinKernel


def test_PiecewiseLinearGPSurrogate():
    surrogate_data = specs.surrogates.valid(PiecewiseLinearGPSurrogate).obj()
    surrogate = surrogates.map(surrogate_data)
    assert isinstance(surrogate, surrogates.PiecewiseLinearGPSurrogate)
    assert_allclose(
        surrogate.transform.tf1.idx_x,
        torch.tensor([4, 5], dtype=torch.int64),
    )
    assert_allclose(
        surrogate.transform.tf1.idx_y,
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )
    assert torch.allclose(
        surrogate.transform.tf1.prepend_x,
        torch.tensor([0], dtype=torch.float64),
    )
    assert torch.allclose(
        surrogate.transform.tf1.prepend_y,
        torch.tensor([], dtype=torch.float64),
    )
    assert torch.allclose(
        surrogate.transform.tf1.append_x,
        torch.tensor([1], dtype=torch.float64),
    )
    assert torch.allclose(
        surrogate.transform.tf1.append_y,
        torch.tensor([], dtype=torch.float64),
    )
    assert torch.allclose(
        surrogate.transform.tf2.bounds,
        torch.tensor([[2], [60]], dtype=torch.float64),
    )
    assert torch.allclose(
        surrogate.transform.tf2.indices,
        torch.tensor([1006], dtype=torch.int64),
    )
    experiments = pd.DataFrame.from_dict(
        {
            "phi_0": {0: 0.2508421787324221, 1: 0.4163841440870552},
            "phi_1": {0: 0.433484737895208, 1: 0.6113523601901784},
            "phi_2": {0: 0.5286771157519854, 1: 0.6648468998497129},
            "phi_3": {0: 0.5755781889180437, 1: 0.7274116735640798},
            "t_1": {0: 0.22223684134422478, 1: 0.23491735169040232},
            "t_2": {0: 0.8253013968891976, 1: 0.7838135122442911},
            "t_3": {0: 20.589423292016406, 1: 6.836910327501014},
            "alpha": {0: 7, 1: 3},
        },
    )
    surrogate.fit(experiments)
    assert isinstance(surrogate.model.covar_module, ScaleKernel)
    assert isinstance(surrogate.model.covar_module.outputscale_prior, GammaPrior)
    assert isinstance(surrogate.model.covar_module.base_kernel, ProductKernel)
    assert isinstance(surrogate.model.covar_module.base_kernel.kernels[0], MaternKernel)
    assert isinstance(
        surrogate.model.covar_module.base_kernel.kernels[0].lengthscale_prior,
        GammaPrior,
    )
    assert surrogate.model.covar_module.base_kernel.kernels[
        0
    ].active_dims == torch.tensor([1006], dtype=torch.int64)
    assert isinstance(
        surrogate.model.covar_module.base_kernel.kernels[1],
        WassersteinKernel,
    )
    assert torch.allclose(
        surrogate.model.covar_module.base_kernel.kernels[1].active_dims,
        torch.tensor(list(range(1000)), dtype=torch.int64),
    )
    assert isinstance(
        surrogate.model.covar_module.base_kernel.kernels[1].lengthscale_prior,
        LogNormalPrior,
    )

    preds1 = surrogate.predict(experiments)
    dump = surrogate.dumps()
    surrogate2 = surrogates.map(surrogate_data)
    surrogate2.loads(dump)
    preds2 = surrogate2.predict(experiments)
    assert_frame_equal(preds1, preds2)


def test_PiecewiseLinearGPSurrogate_no_continuous():
    surrogate_data = PiecewiseLinearGPSurrogate(
        inputs=Inputs(
            features=[ContinuousInput(key=f"phi_{i}", bounds=(0, 1)) for i in range(4)]
            + [ContinuousInput(key=f"t_{i+1}", bounds=(0, 1)) for i in range(2)]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="alpha")]),
        interpolation_range=(0, 1),
        n_interpolation_points=1000,
        x_keys=["t_1", "t_2"],
        y_keys=[f"phi_{i}" for i in range(4)],
        continuous_keys=[],
        continuous_kernel=None,
        prepend_x=[0.0],
        append_x=[1.0],
        prepend_y=[],
        append_y=[],
        normalize_y=100.0,
    )
    surrogate = surrogates.map(surrogate_data)
    experiments = pd.DataFrame.from_dict(
        {
            "phi_0": {0: 0.2508421787324221, 1: 0.4163841440870552},
            "phi_1": {0: 0.433484737895208, 1: 0.6113523601901784},
            "phi_2": {0: 0.5286771157519854, 1: 0.6648468998497129},
            "phi_3": {0: 0.5755781889180437, 1: 0.7274116735640798},
            "t_1": {0: 0.22223684134422478, 1: 0.23491735169040232},
            "t_2": {0: 0.8253013968891976, 1: 0.7838135122442911},
            "alpha": {0: 7, 1: 3},
        },
    )
    surrogate.fit(experiments)


def test_PiecewiseLinearGPHyperconfig():
    surrogate_data = specs.surrogates.valid(PiecewiseLinearGPSurrogate).obj()

    candidate = surrogate_data.hyperconfig.inputs.sample(1).loc[0]
    surrogate_data.update_hyperparameters(candidate)
    assert surrogate_data.continuous_kernel.ard == (candidate["ard"] == "True")
    if candidate.continuous_kernel == "matern_1.5":
        assert isinstance(surrogate_data.continuous_kernel, MaternKernelDataModel)
        assert surrogate_data.continuous_kernel.nu == 1.5
    elif candidate.continuous_kernel == "matern_2.5":
        assert isinstance(surrogate_data.continuous_kernel, MaternKernelDataModel)
        assert surrogate_data.continuous_kernel.nu == 2.5
    else:
        assert isinstance(surrogate_data.continuous_kernel, RBFKernelDataModel)
    if candidate.prior == "mbo":
        assert surrogate_data.noise_prior == MBO_NOISE_PRIOR()
        assert (
            surrogate_data.continuous_kernel.lengthscale_prior == MBO_LENGTHCALE_PRIOR()
        )
    else:
        assert surrogate_data.noise_prior == THREESIX_NOISE_PRIOR()
        assert (
            surrogate_data.continuous_kernel.lengthscale_prior
            == THREESIX_LENGTHSCALE_PRIOR()
        )
