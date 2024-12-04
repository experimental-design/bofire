import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import ContinuousInput
from bofire.strategies.doe.transform import IndentityTransform, MinMaxTransform


def test_IdentityTransform():
    t = IndentityTransform()
    x = np.random.uniform(10, size=(10))
    assert_allclose(x, t(x))
    assert_allclose(np.ones(10), t.jacobian(x))


def test_MinMaxTransform():
    inputs = Inputs(
        features=[
            ContinuousInput(key="a", bounds=(0, 2)),
            ContinuousInput(key="b", bounds=(4, 8)),
        ],
    )
    t = MinMaxTransform(inputs=inputs, feature_range=(-1, 1))
    samples = pd.DataFrame.from_dict({"a": [1, 2], "b": [4, 6]})
    transformed_samples = t(samples.values.flatten())
    assert_allclose(transformed_samples, np.array([0, -1, 1, 0]))
    transformed_jacobian = t.jacobian(samples.values.flatten())
    assert_allclose(transformed_jacobian, np.array([1.0, 0.5, 1.0, 0.5]))
