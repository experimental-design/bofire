import random

import numpy as np
import pandas as pd
import pytest

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.api import Domain, Inputs
from bofire.data_models.features.api import ContinuousInput

# from tests.bofire.data_models.test_features import (
# )

input_feature = ContinuousInput(key="a", bounds=(10, 20))
xt = pd.Series(np.linspace(0, 1))
expected = np.linspace(-10, 20)


@pytest.mark.parametrize(
    "feature, xt, expected",
    [
        (
            ContinuousInput(key="a", bounds=(0, 10)),
            pd.Series(np.linspace(0, 1, 11)),
            np.linspace(0, 10, 11),
        ),
        (
            ContinuousInput(key="a", bounds=(-10, 20)),
            pd.Series(np.linspace(0, 1)),
            np.linspace(-10, 20),
        ),
    ],
)
def test_continuous_input_feature_from_unit_range(feature, xt, expected):
    x = feature.from_unit_range(xt)
    assert np.allclose(x.values, expected)
