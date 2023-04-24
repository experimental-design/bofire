import pytest

from bofire.data_models.domain.api import Inputs
from bofire.data_models.domain.features import ContinuousInput
from bofire.utils.doe import get_confounding_matrix

inputs = Inputs(
    features=[ContinuousInput(key=i, bounds=(0, 10)) for i in ["a", "b", "c"]]
)


design = inputs.sample(20)


@pytest.mark.parametrize(
    "powers, interactions",
    [([-1], [2]), ([2], [1]), ([2], [4])],
)
def test_get_confounding_matrix_invalid(powers, interactions):
    with pytest.raises(AssertionError):
        get_confounding_matrix(
            inputs=inputs,
            design=design,
            powers=powers,
            interactions=interactions,
        )


@pytest.mark.parametrize(
    "powers, interactions",
    [([2], [2]), ([3], [2]), ([3], [2, 3])],
)
def test_get_confounding_matrix_valid(powers, interactions):
    get_confounding_matrix(
        inputs=inputs,
        design=design,
        powers=powers,
        interactions=interactions,
    )
