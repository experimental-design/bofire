import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import assert_array_equal

from bofire.utils.categoricalDescriptorEncoder import CategoricalDescriptorEncoder

VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT = {
    "categories": ["A", "B", "C"],
    "descriptors": ["d1", "d2"],
    "values": [
        [1.0, 2.0],
        [3.0, 7.0],
        [5.0, 1.0],
    ],
}

data = pd.DataFrame.from_dict({"c1": ["A", "C", "B"]})
X_expected = np.array([[1, 2], [5, 1], [3, 7]])
X_expected_ignore = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 7.0]])

data_unknownCat = pd.DataFrame.from_dict({"c1": ["A", "U", "B"]})

enc = CategoricalDescriptorEncoder(**VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT)


@pytest.mark.parametrize(
    "data, data_unknownCat, expected", [(data, data_unknownCat, X_expected_ignore)]
)
def test_categorical_descriptor_encoder_handle_unknown(data, data_unknownCat, expected):

    # Test that one hot encoder raises error for unknown features
    # present during transform.
    enc = CategoricalDescriptorEncoder(
        **VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT, handle_unknown="error"
    )
    enc.fit(data)
    with pytest.raises(ValueError, match="Found unknown categories"):
        enc.transform(data_unknownCat)

    # Test the ignore option, ignores unknown features (giving all 0's)
    enc = CategoricalDescriptorEncoder(
        **VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT, handle_unknown="ignore"
    )
    enc.fit(data)
    X_passed = data_unknownCat.copy()
    # ignore option allows unknown categories but still warns about
    # coercion to 0
    with pytest.warns(UserWarning):
        assert_array_equal(
            enc.transform(X_passed),
            expected,
        )

    # ensure transformed data was not modified in place
    assert_array_equal(data_unknownCat, X_passed)

    # Raise error if handle_unknown is neither ignore or error.
    enc = CategoricalDescriptorEncoder(
        **VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT, handle_unknown="42"
    )
    with pytest.raises(ValueError, match="handle_unknown should be either"):
        enc.fit(data)


@pytest.mark.parametrize("data, enc", [(data, enc)])
def test_categorical_descriptor_encoder_not_fitted(data, enc):
    msg = (
        "This CategoricalDescriptorEncoder instance is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this "
        "estimator."
    )
    with pytest.raises(NotFittedError, match=msg):
        enc.transform(data)


@pytest.mark.parametrize("output_dtype", [np.int32, np.float32, np.float64])
def test_categorical_descriptor_encoder_transform(output_dtype):

    enc = CategoricalDescriptorEncoder(
        **VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT, dtype=output_dtype
    )
    assert_array_equal(enc.fit_transform(data), X_expected)
    assert_array_equal(enc.fit(data).transform(data), X_expected)


@pytest.mark.parametrize("data, enc", [(data, enc)])
def test_categorical_descriptor_encoder_inverse_transform(data, enc):
    X_tr = enc.fit_transform(data)
    assert_array_equal(enc.inverse_transform(X_tr), data)

    # incorrect shape raises
    X_tr = np.array([[0, 1, 1, 2], [1, 0, 1, 0]])
    msg = re.escape("Shape of the passed X data is not correct")
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_tr)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("X", [[1, 2], np.array([3.0, 4.0])])
@pytest.mark.parametrize("enc", [enc])
def test_X_is_not_1D(enc, X, method):

    msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=msg):
        getattr(enc, method)(X)


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("enc", [enc])
def test_X_is_not_1D_pandas(enc, method):
    pd = pytest.importorskip("pandas")
    X = pd.Series([6, 3, 4, 6])

    msg = "Expected 2D array, got 1D array instead"
    with pytest.raises(ValueError, match=msg):
        getattr(enc, method)(X)


@pytest.mark.parametrize(
    "data, data_unknownCat, X_expected", [(data, data_unknownCat, X_expected)]
)
def test_categorical_descriptor_encoder_categories(data, data_unknownCat, X_expected):

    # auto-generated categories
    enc = CategoricalDescriptorEncoder(
        categories="auto", values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"]
    )
    enc.fit(data)

    assert isinstance(enc.categories_, list)

    cat_exp = ["A", "B", "C"]
    assert_array_equal(enc.categories_[0], cat_exp)

    cats = VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["categories"]
    enc = CategoricalDescriptorEncoder(
        categories=cats, values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"]
    )

    assert_array_equal(enc.fit_transform(data), X_expected)

    assert list(enc.categories[0]) == list(cats)
    assert enc.categories_[0].tolist() == list(cats)

    # when specifying categories manually, unknown categories should already
    # raise when fitting
    enc = CategoricalDescriptorEncoder(
        categories=cats, values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"]
    )
    with pytest.raises(ValueError, match="Found unknown categories"):
        enc.fit(data_unknownCat)

    # Test the ignore option, ignores unknown features (giving all 0's)
    enc = CategoricalDescriptorEncoder(
        categories=cats,
        values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"],
        handle_unknown="ignore",
    )

    # the ignore option allows unknown categories in the input but still triggers a warning
    with pytest.warns(UserWarning):
        assert_array_equal(
            enc.fit(data_unknownCat).transform(data_unknownCat), X_expected_ignore
        )


@pytest.mark.parametrize("data, X_expected", [(data, X_expected)])
def test_categorical_descriptor_encoder_descriptors(data, X_expected):

    enc = CategoricalDescriptorEncoder(
        descriptors="auto", values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"]
    )
    enc.fit(data)

    assert isinstance(enc.descriptors_, list)

    des_exp = ["Descriptor_0_0", "Descriptor_0_1"]
    assert_array_equal(enc.descriptors_[0], des_exp)

    des = VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["descriptors"]
    enc = CategoricalDescriptorEncoder(
        descriptors=des, values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"]
    )

    assert_array_equal(enc.fit_transform(data), X_expected)

    assert list(enc.descriptors[0]) == des
    assert enc.descriptors_[0] == des


@pytest.mark.parametrize(
    "data",
    [(data)],
)
def test_categorical_descriptors_encoder_raise_categories_shape(data):

    cats = [["A", "B"], ["C", "D"]]
    enc = CategoricalDescriptorEncoder(
        categories=cats, values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"]
    )
    msg = "Shape mismatch: if categories is an array,"

    with pytest.raises(ValueError, match=msg):
        enc.fit(data)


@pytest.mark.parametrize(
    "data",
    [(data)],
)
def test_categorical_descriptors_encoder_raise_descriptors_shape(data):

    des = ["Low", "Medium", "High"]
    enc = CategoricalDescriptorEncoder(
        descriptors=des, values=VALID_CATEGORICAL_DESCRIPTOR_ENCODER_INPUT["values"]
    )
    msg = "Shape mismatch: number of descriptors"

    with pytest.raises(ValueError, match=msg):
        enc.fit(data)


@pytest.mark.parametrize(
    "data",
    [(data)],
)
def test_categorical_descriptors_encoder_raise_values_shape(data):

    enc = CategoricalDescriptorEncoder(values=[[1, 2], [2, 3]])
    msg = "Shape mismatch: descriptor values has to be"

    with pytest.raises(ValueError, match=msg):
        enc.fit(data)
