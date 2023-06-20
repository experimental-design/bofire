import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import tests.bofire.data_models.specs.api as Specs
from bofire.data_models.domain.api import Features, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    MolecularInput,
)
from bofire.data_models.objectives.api import  Objective
from bofire.data_models.surrogates.api import ScalerEnum


@pytest.mark.parametrize(
    "spec, n",
    [
        (spec, n)
        for spec in Specs.features.valids
        if (spec.cls != ContinuousOutput)
        and (spec.cls != MolecularInput)
        and (spec.cls != CategoricalOutput)
        for n in [1, 5]
    ],
)
def test_sample(spec: Specs.Spec, n: int):
    feat = spec.obj()
    samples = feat.sample(n=n)
    feat.validate_candidental(samples)


cont = Specs.features.valid(ContinuousInput).obj()
cat = Specs.features.valid(CategoricalInput).obj()
cat_ = Specs.features.valid(CategoricalDescriptorInput).obj()
out = Specs.features.valid(ContinuousOutput).obj()


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


# test features container
if1 = Specs.features.valid(ContinuousInput).obj(key="if1")
if2 = Specs.features.valid(ContinuousInput).obj(key="if2")
if3 = Specs.features.valid(ContinuousInput).obj(key="if3", bounds=(3, 3))
if4 = Specs.features.valid(CategoricalInput).obj(
    key="if4", categories=["a", "b"], allowed=[True, False]
)
if5 = Specs.features.valid(DiscreteInput).obj(key="if5")
if7 = Specs.features.valid(CategoricalInput).obj(
    key="if7",
    categories=["c", "d", "e"],
    allowed=[True, False, False],
)


of1 = Specs.features.valid(ContinuousOutput).obj(key="of1")
of2 = Specs.features.valid(ContinuousOutput).obj(key="of2")
of3 = Specs.features.valid(ContinuousOutput).obj(key="of3", objective=None)

inputs = Inputs(features=[if1, if2])
outputs = Outputs(features=[of1, of2])
features = Features(features=[if1, if2, of1, of2])

@pytest.mark.parametrize(
    "features, expected",
    [
        (Inputs(features=[if1, if2]), []),
        (Inputs(features=[if1, if2, if3, if4, if5]), [if3, if4]),
    ],
)
def test_inputs_get_fixed(features, expected):
    returned = features.get_fixed()
    assert isinstance(features, Inputs)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


@pytest.mark.parametrize(
    "features, expected",
    [
        (Inputs(features=[if1, if2]), [if1, if2]),
        (Inputs(features=[if1, if2, if3, if4, if5]), [if1, if2, if5]),
    ],
)
def test_inputs_get_free(features, expected):
    returned = features.get_free()
    assert isinstance(features, Inputs)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


@pytest.mark.parametrize(
    "features, num_samples, method",
    [
        (features, num_samples, method)
        for features in [
            inputs,
            Inputs(features=[if1, if2, if3, if4, if5, if7]),
        ]
        for num_samples in [1, 2, 1024]
        for method in ["UNIFORM", "SOBOL", "LHS"]
    ],
)
def test_inputs_sample(features: Inputs, num_samples, method):
    samples = features.sample(num_samples, method=method)
    assert samples.shape == (num_samples, len(features))
    assert list(samples.columns) == features.get_keys()


@pytest.mark.parametrize(
    "specs",
    [
        ({"x4": CategoricalEncodingEnum.ONE_HOT}),
        ({"x1": CategoricalEncodingEnum.ONE_HOT}),
        ({"x2": ScalerEnum.NORMALIZE}),
        ({"x2": CategoricalEncodingEnum.DESCRIPTOR}),
    ],
)
def test_inputs_validate_transform_specs_invalid(specs):
    inps = Inputs(
        features=[
            Specs.features.valid(ContinuousInput).obj(key="x1", bounds=(0, 1)),
            Specs.features.valid(CategoricalInput).obj(key="x2", categories=["apple", "banana"]
                ,allowed=[True,True]),
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="x3",
                categories=["apple", "banana"],
                allowed=[True,True],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    with pytest.raises(ValueError):
        inps._validate_transform_specs(specs)


@pytest.mark.parametrize(
    "specs",
    [
        ({"x2": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.DESCRIPTOR}),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            }
        ),
    ],
)
def test_inputs_validate_transform_valid(specs):
    inps = Inputs(
        features=[
            Specs.features.valid(ContinuousInput).obj(key="x1", bounds=(0, 1)),
            Specs.features.valid(CategoricalInput).obj(key="x2", categories=["apple", "banana"]
                ,allowed=[True,True]),
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="x3",
                categories=["apple", "banana"],
                allowed=[True,True],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    inps._validate_transform_specs(specs)


@pytest.mark.parametrize(
    "specs, expected_features2idx, expected_features2names",
    [
        (
            {"x2": CategoricalEncodingEnum.ONE_HOT},
            {"x1": (0,), "x2": (2, 3, 4), "x3": (1,)},
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": ("x3",),
            },
        ),
        (
            {"x2": CategoricalEncodingEnum.DUMMY},
            {"x1": (0,), "x2": (2, 3), "x3": (1,)},
            {"x1": ("x1",), "x2": ("x2_banana", "x2_orange"), "x3": ("x3",)},
        ),
        (
            {"x2": CategoricalEncodingEnum.ORDINAL},
            {"x1": (0,), "x2": (2,), "x3": (1,)},
            {"x1": ("x1",), "x2": ("x2",), "x3": ("x3",)},
        ),
        (
            {"x3": CategoricalEncodingEnum.ONE_HOT},
            {"x1": (0,), "x2": (5,), "x3": (1, 2, 3, 4)},
            {
                "x1": ("x1",),
                "x2": ("x2",),
                "x3": ("x3_apple", "x3_banana", "x3_orange", "x3_cherry"),
            },
        ),
        (
            {"x3": CategoricalEncodingEnum.DESCRIPTOR},
            {"x1": (0,), "x2": (3,), "x3": (1, 2)},
            {
                "x1": ("x1",),
                "x2": ("x2",),
                "x3": (
                    "x3_d1",
                    "x3_d2",
                ),
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            },
            {"x1": (0,), "x2": (3, 4, 5), "x3": (1, 2)},
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": (
                    "x3_d1",
                    "x3_d2",
                ),
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            },
            {"x1": (0,), "x2": (5, 6, 7), "x3": (1, 2, 3, 4)},
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": ("x3_apple", "x3_banana", "x3_orange", "x3_cherry"),
            },
        ),
    ],
)
def test_inputs_get_transform_info(
    specs, expected_features2idx, expected_features2names
):
    inps = Inputs(
        features=[
            Specs.features.valid(ContinuousInput).obj(key="x1", bounds=(0, 1)),
            Specs.features.valid(CategoricalInput).obj(key="x2", categories=["apple", "banana", "orange"]
                ,allowed=[True,True,True]),
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                allowed=[True,True,True,True],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
        ]
    )
    features2idx, features2names = inps._get_transform_info(specs)
    assert features2idx == expected_features2idx
    assert features2names == expected_features2names


@pytest.mark.parametrize(
    "specs",
    [
        ({"x2": CategoricalEncodingEnum.ONE_HOT}),
        ({"x2": CategoricalEncodingEnum.DUMMY}),
        ({"x2": CategoricalEncodingEnum.ORDINAL}),
        ({"x3": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.DESCRIPTOR}),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            }
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            }
        ),
        (
            {
                "x2": CategoricalEncodingEnum.DUMMY,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            }
        ),
    ],
)
def test_inputs_transform(specs):
    inps = Inputs(
        features=[
            Specs.features.valid(ContinuousInput).obj(key="x1", bounds=(0, 1)),
            Specs.features.valid(CategoricalInput).obj(key="x2", categories=["apple", "banana", "orange"]
                ,allowed=[True,True,True]),
            Specs.features.valid(CategoricalDescriptorInput).obj(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                allowed=[True,True,True,True],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
        ]
    )
    samples = inps.sample(n=100)
    samples = samples.sample(40)
    transformed = inps.transform(experiments=samples, specs=specs)
    untransformed = inps.inverse_transform(experiments=transformed, specs=specs)
    assert_frame_equal(samples, untransformed)


if1 = Specs.features.valid(ContinuousInput).obj(key="if1")
if2 = Specs.features.valid(ContinuousInput).obj(key="if2", bounds=(3, 3))
if3 = Specs.features.valid(CategoricalInput).obj(
    key="if3",
    categories=["c1", "c2", "c3"],
    allowed=[True, True, True],
)
if4 = Specs.features.valid(CategoricalInput).obj(
    key="if4",
    categories=["c1", "c2", "c3"],
    allowed=[True, False, False],
)
if5 = Specs.features.valid(CategoricalDescriptorInput).obj(
    key="if5",
    categories=["c1", "c2", "c3"],
    allowed=[True, False, False],
    descriptors=["d1", "d2"],
    values=[
        [1, 2],
        [3, 7],
        [5, 1],
    ],
)
if6 = Specs.features.valid(CategoricalDescriptorInput).obj(
    key="if6",
    categories=["c1", "c2", "c3"],
    allowed=[True, False, False],
    descriptors=["d1", "d2"],
    values=[
        [1, 2],
        [3, 7],
        [5, 1],
    ],
)

of1 = Specs.features.valid(ContinuousOutput).obj(key="of1")

inputs1 = Inputs(features=[if1, if3, if5])

inputs2 = Inputs(
    features=[
        if1,
        if2,
        if3,
        if4,
        if5,
        if6,
    ]
)


@pytest.mark.parametrize(
    "inputs, specs, expected_bounds",
    [
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 2, 0, 0, 0], [5.3, 1, 2, 1, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.DUMMY,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 2, 0, 0], [5.3, 1, 2, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.DUMMY,
                "if5": CategoricalEncodingEnum.DUMMY,
            },
            [[3, 0, 0, 0, 0], [5.3, 1, 1, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ONE_HOT,
            },
            [[3, 0, 0, 0, 0, 0, 0], [5.3, 1, 0, 0, 1, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 2, 0], [5.3, 1, 2, 2]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.ORDINAL,
            },
            [[3, 0, 0], [5.3, 2, 2]],
        ),
        # new domain
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [3, 3, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0],
                [
                    5.3,
                    3,
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                ],
            ],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ONE_HOT,
                "if6": CategoricalEncodingEnum.ONE_HOT,
            },
            [
                [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5.3, 3, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            ],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [
                    3,
                    3,
                    1,
                    2,
                    1,
                    2,
                    0,
                    0,
                ],
                [5.3, 3, 1, 2, 1, 2, 2, 2],
            ],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.ORDINAL,
                "if6": CategoricalEncodingEnum.ORDINAL,
            },
            [[3, 3, 0, 0, 0, 0], [5.3, 3, 2, 2, 2, 2]],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ORDINAL,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [3.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [
                    5.3,
                    3.0,
                    2.0,
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            ],
        ),
    ],
)
def test_inputs_get_bounds(inputs, specs, expected_bounds):
    lower, upper = inputs.get_bounds(specs=specs)
    assert np.allclose(expected_bounds[0], lower)
    assert np.allclose(expected_bounds[1], upper)


def test_inputs_get_bounds_fit():
    # at first the fix on the continuous ones is tested
    inputs = Inputs(features=[if1, if2])
    experiments = inputs.sample(100)
    experiments["if1"] += [random.uniform(-2, 2) for _ in range(100)]
    experiments["if2"] += [random.uniform(-2, 2) for _ in range(100)]
    opt_bounds = inputs.get_bounds(specs={})
    fit_bounds = inputs.get_bounds(specs={}, experiments=experiments)
    for i, key in enumerate(inputs.get_keys(ContinuousInput)):
        assert fit_bounds[0][i] < opt_bounds[0][i]
        assert fit_bounds[1][i] > opt_bounds[1][i]
        assert fit_bounds[0][i] == experiments[key].min()
        assert fit_bounds[1][i] == experiments[key].max()
    # next test the fix for the CategoricalDescriptor feature
    inputs = Inputs(
        features=[
            if1,
            if2,
            if3,
            if4,
            if5,
            if6,
        ]
    )
    experiments = inputs.sample(100)
    experiments["if4"] = [random.choice(if4.categories) for _ in range(100)]
    experiments["if6"] = [random.choice(if6.categories) for _ in range(100)]
    opt_bounds = inputs.get_bounds(
        specs={
            "if3": CategoricalEncodingEnum.ONE_HOT,
            "if4": CategoricalEncodingEnum.ONE_HOT,
            "if5": CategoricalEncodingEnum.DESCRIPTOR,
            "if6": CategoricalEncodingEnum.DESCRIPTOR,
        }
    )
    fit_bounds = inputs.get_bounds(
        {
            "if3": CategoricalEncodingEnum.ONE_HOT,
            "if4": CategoricalEncodingEnum.ONE_HOT,
            "if5": CategoricalEncodingEnum.DESCRIPTOR,
            "if6": CategoricalEncodingEnum.DESCRIPTOR,
        },
        experiments=experiments,
    )
    # check difference in descriptors
    assert opt_bounds[0][-8] == 1
    assert opt_bounds[1][-8] == 1
    assert opt_bounds[0][-7] == 2
    assert opt_bounds[1][-7] == 2
    assert fit_bounds[0][-8] == 1
    assert fit_bounds[0][-7] == 1
    assert fit_bounds[1][-8] == 5
    assert fit_bounds[1][-7] == 7
    # check difference in onehots
    assert opt_bounds[1][-1] == 0
    assert opt_bounds[1][-2] == 0
    assert fit_bounds[1][-1] == 1
    assert fit_bounds[1][-2] == 1


mixed_data = pd.DataFrame(
    columns=["of1", "of2", "of3"],
    index=range(5),
    data=np.random.uniform(size=(5, 3)),
)
mixed_data["of4"] = ["a", "a", "b", "b", "a"]


@pytest.mark.parametrize(
    "features, samples",
    [
        (
            outputs,
            pd.DataFrame(
                columns=["of1", "of2"],
                index=range(5),
                data=np.random.uniform(size=(5, 2)),
            ),
        ),
        (
            Outputs(features=[of1, of2, of3]),
            pd.DataFrame(
                columns=["of1", "of2", "of3"],
                index=range(5),
                data=np.random.uniform(size=(5, 3)),
            ),
        ),
        (
            Outputs(
                features=[
                    of1,
                    of2,
                    of3,
                    Specs.features.valid(CategoricalOutput).obj(
                        key="of4", categories=["a", "b"], objective=[1.0, 0.0]
                    ),
                ]
            ),
            mixed_data,
        ),
    ],
)
def test_outputs_call(features, samples):
    o = features(samples)
    assert o.shape == (
        len(samples),
        len(features.get_keys_by_objective(Objective))
        + len(features.get_keys(CategoricalOutput)),
    )
    assert list(o.columns) == [
        f"{key}_des"
        for key in features.get_keys_by_objective(Objective)
        + features.get_keys(CategoricalOutput)
    ]


