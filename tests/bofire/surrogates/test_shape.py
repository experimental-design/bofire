import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from torch.testing import assert_allclose

import bofire.surrogates.api as surrogates
import tests.bofire.data_models.specs.api as specs
from bofire.data_models.surrogates.api import PiecewiseLinearGPSurrogate
from bofire.kernels.shape import WassersteinKernel


def test_PiecewiseLinearGPSurrogate():
    surrogate_data = specs.surrogates.valid(PiecewiseLinearGPSurrogate).obj()
    surrogate = surrogates.map(surrogate_data)
    assert isinstance(surrogate, surrogates.PiecewiseLinearGPSurrogate)
    assert_allclose(
        surrogate.transform.idx_x, torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    )
    assert_allclose(
        surrogate.transform.idx_y, torch.tensor([4, 5, 6, 7], dtype=torch.int64)
    )
    assert surrogate.transform.prepend_x == 0.0
    assert surrogate.transform.prepend_y == 0.0
    assert surrogate.transform.append_x == 60.0
    assert surrogate.transform.append_y == 1.0
    assert surrogate.transform.new_x.shape == (400,)

    experiments = pd.DataFrame(
        {
            "x_0": [10, 10],
            "x_1": [20, 20],
            "x_2": [30, 30],
            "x_3": [40, 40],
            "y_0": [0, 0],
            "y_1": [0, 0],
            "y_2": [1, 0],
            "y_3": [1, 1],
            "alpha": [15, 5],
        }
    )
    surrogate.fit(experiments)

    assert isinstance(surrogate.model.covar_module.base_kernel, WassersteinKernel)
    preds1 = surrogate.predict(experiments)
    dump = surrogate.dumps()
    surrogate2 = surrogates.map(surrogate_data)
    surrogate2.loads(dump)
    preds2 = surrogate2.predict(experiments)
    assert_frame_equal(preds1, preds2)
