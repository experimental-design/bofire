import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from bofire.data_models.encodings.api import OrdinalEncoding
from bofire.data_models.features.api import CategoricalInput


def test_categorical_to_label_encoding():
    c = CategoricalInput(key="c", categories=["B", "A", "C"])
    samples = pd.Series(["A", "A", "C", "B"])
    t_samples = c.to_encoding(OrdinalEncoding(), samples)
    assert_series_equal(t_samples["c"], pd.Series([1, 1, 2, 0], name="c"))
    untransformed = c.from_encoding(OrdinalEncoding(), t_samples)
    assert np.all(samples == untransformed)


def test_ordinal_get_bounds():
    feature = CategoricalInput(key="c", categories=["B", "A", "C"])
    lower, upper = feature.get_bounds(transform_type=OrdinalEncoding(), values=None)
    assert np.allclose(lower, 0)
    assert np.allclose(upper, 2)
