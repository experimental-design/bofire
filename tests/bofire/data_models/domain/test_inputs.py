import itertools

import pytest

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import CategoricalDescriptorInput, CategoricalInput


@pytest.mark.parametrize(
    "inputs, data",
    [
        (
            Inputs(),
            [],
        ),
        (
            Inputs(
                features=[
                    CategoricalInput(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                ]
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
            ],
        ),
        (
            Inputs(
                features=[
                    CategoricalInput(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalInput(
                        key="f2",
                        categories=["c21", "c22", "c23"],
                    ),
                    CategoricalInput(
                        key="f3",
                        categories=["c31", "c32"],
                    ),
                ]
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
                [("f2", "c21"), ("f2", "c22"), ("f2", "c23")],
                [("f3", "c31"), ("f3", "c32")],
            ],
        ),
    ],
)
def test_inputs_get_categorical_combinations(inputs, data):
    expected = list(itertools.product(*data))
    assert inputs.get_categorical_combinations() == expected


@pytest.mark.parametrize(
    "inputs, data, include, exclude",
    [
        (
            Inputs(
                features=[
                    CategoricalInput(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInput(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ]
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
            ],
            CategoricalInput,
            CategoricalDescriptorInput,
        ),
        (
            Inputs(
                features=[
                    CategoricalInput(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInput(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ]
            ),
            [
                [("f2", "c21"), ("f2", "c22")],
                [("f1", "c11"), ("f1", "c12")],
            ],
            CategoricalInput,
            None,
        ),
        (
            Inputs(
                features=[
                    CategoricalInput(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInput(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ]
            ),
            [
                [("f2", "c21"), ("f2", "c22")],
            ],
            CategoricalDescriptorInput,
            None,
        ),
        (
            Inputs(
                features=[
                    CategoricalInput(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInput(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ]
            ),
            [],
            CategoricalDescriptorInput,
            CategoricalInput,
        ),
    ],
)
def test_categorical_combinations_of_domain_filtered(inputs, data, include, exclude):
    expected = list(itertools.product(*data))
    assert (
        inputs.get_categorical_combinations(include=include, exclude=exclude)
        == expected
    )
