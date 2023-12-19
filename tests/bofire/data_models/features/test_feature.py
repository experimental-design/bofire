"""Generic tests are going here."""

import pytest
from pandas.testing import assert_series_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
    MolecularInput,
)


@pytest.mark.parametrize(
    "spec, n",
    [
        (spec, n)
        for spec in specs.features.valids
        if (spec.cls != ContinuousOutput)
        and (spec.cls != MolecularInput)
        and (spec.cls != CategoricalOutput)
        for n in [1, 5]
    ],
)
def test_sample(spec: specs.Spec, n: int):
    feat = spec.obj()
    samples0 = feat.sample(n=n)
    feat.validate_candidental(samples0)
    samples1 = feat.sample(n=n, seed=42)
    feat.validate_candidental(samples1)
    samples2 = feat.sample(n=n, seed=42)
    feat.validate_candidental(samples2)
    assert_series_equal(samples1, samples2)


cont = specs.features.valid(ContinuousInput).obj()
cat = specs.features.valid(CategoricalInput).obj()
cat_ = specs.features.valid(CategoricalDescriptorInput).obj()
out = specs.features.valid(ContinuousOutput).obj()


@pytest.mark.parametrize(
    "unsorted_list, sorted_list",
    [
        (
            [cont, cat_, cat, out],
            [cont, cat_, cat, out],
        ),
        (
            [cont, cat_, cat, out, cat_, out],
            [cont, cat_, cat_, cat, out, out],
        ),
        (
            [cont, out],
            [cont, out],
        ),
        (
            [out, cont],
            [cont, out],
        ),
    ],
)
def test_feature_sorting(unsorted_list, sorted_list):
    assert sorted(unsorted_list) == sorted_list
