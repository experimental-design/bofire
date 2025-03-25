import pytest
from numpy.testing import assert_array_equal

from bofire.data_models.domain.api import Inputs
from bofire.data_models.domain.features import ContinuousInput
from bofire.utils.default_fracfac_generators import default_fracfac_generators
from bofire.utils.doe import (
    apply_block_generator,
    compute_generator,
    ff2n,
    fracfact,
    get_alias_structure,
    get_block_generator,
    get_confounding_matrix,
    get_default_generator,
    get_generator,
    get_n_blocks,
    validate_generator,
)


inputs = Inputs(
    features=[ContinuousInput(key=i, bounds=(0, 10)) for i in ["a", "b", "c"]],
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


def test_ff2n():
    design = ff2n(1)
    assert_array_equal(design, [[-1], [1]])
    design = ff2n(2)
    assert_array_equal(design, [[-1, -1], [-1, 1], [1, -1], [1, 1]])
    design = ff2n(3)
    assert_array_equal(
        design,
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
    )


def test_fracfact():
    design = fracfact("a b c")
    assert_array_equal(design, ff2n(3))
    design = fracfact("a b ab")
    assert_array_equal(
        design,
        [
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, 1, 1],
        ],
    )
    design = fracfact("a b -ab")
    assert_array_equal(
        design,
        [
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
    )
    design = fracfact("a b c ab ac")
    assert_array_equal(
        design,
        [
            [-1, -1, -1, 1, 1],
            [-1, -1, 1, 1, -1],
            [-1, 1, -1, -1, 1],
            [-1, 1, 1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1],
            [1, 1, -1, 1, -1],
            [1, 1, 1, 1, 1],
        ],
    )


def test_get_alias_structure():
    alias_structure = get_alias_structure("a b c")
    assert sorted(alias_structure) == sorted(
        ["a", "b", "c", "I", "ab", "ac", "bc", "abc"],
    )
    alias_structure = get_alias_structure("a b ab")
    assert sorted(alias_structure) == sorted(["I = abc", "a = bc", "b = ac", "c = ab"])


@pytest.mark.parametrize(
    "n_factors, generator, message",
    [
        (2, "a b c", "Generator does not match the number of factors."),
        (2, "a a", "Main factors are confounded with each other."),
        (2, "a c", "Use the letters `a b` for the main factors."),
        (5, "a b c ab ab", "Generators are not unique."),
        (5, "a b c ab ad", "Generators are not valid."),
        (2, "ab ac", "At least one unconfounded main factor is needed."),
    ],
)
def test_validate_generator_invalid(n_factors: int, generator: str, message: str):
    with pytest.raises(ValueError, match=message):
        validate_generator(n_factors, generator)


@pytest.mark.parametrize(
    "n_factors, n_generators, expected",
    [
        (1, 0, "a"),
        (2, 0, "a b"),
        (3, 0, "a b c"),
        (3, 1, "a b ab"),
        (4, 0, "a b c d"),
        (4, 1, "a b c abc"),
        (5, 1, "a b c d abcd"),
        (5, 2, "a b c ab ac"),
        (6, 0, "a b c d e f"),
        (6, 1, "a b c d e abcde"),
        (6, 2, "a b c d abc abd"),
        (6, 3, "a b c ab ac bc"),
        (7, 0, "a b c d e f g"),
        (7, 1, "a b c d e f abcdef"),
        (7, 2, "a b c d e abcd abce"),
        (7, 3, "a b c d abc abd acd"),
        (7, 4, "a b c ab ac bc abc"),
        (8, 0, "a b c d e f g h"),
        (8, 1, "a b c d e f g abcdefg"),
        (8, 2, "a b c d e f abcde abcdf"),  # minitab is giving here abcd, abef
        (8, 3, "a b c d e abcd abce abde"),  # minitab is giving here abc abd bcde
        (8, 4, "a b c d abc abd acd bcd"),
    ],
)
def test_compute_generator(n_factors, n_generators, expected):
    assert compute_generator(n_factors, n_generators) == expected


@pytest.mark.parametrize(
    "n_factors, n_generators",
    [(2, 1), (3, 2), (4, 3), (4, 2), (5, 3), (6, 4), (7, 5), (8, 5)],
)
def test_compute_generator_invalid(n_factors, n_generators):
    with pytest.raises(
        ValueError,
        match="Design not possible, as main factors are confounded with each other.",
    ):
        compute_generator(n_factors, n_generators)


def test_get_default_generator():
    for _, row in default_fracfac_generators.iterrows():
        n_factors = row["n_factors"]
        n_generators = row["n_generators"]
        g = get_default_generator(n_factors, n_generators)
        validate_generator(n_factors, g)
    with pytest.raises(
        ValueError,
        match="No generator available for the requested combination.",
    ):
        get_default_generator(100, 1)


def test_get_generator():
    assert get_generator(6, 2) != compute_generator(6, 2)
    assert get_generator(16, 1) == compute_generator(16, 1)


@pytest.mark.parametrize(
    "n_factors, n_generators, n_repetitions, expected",
    [
        (3, 0, 1, [2, 4]),
        (3, 0, 2, [2, 4, 8]),
        (3, 0, 3, [2, 3, 4, 6, 12]),
        (4, 0, 1, [2, 4, 8]),
        (4, 0, 2, [2, 4, 8, 16]),
        (4, 0, 3, [2, 3, 4, 6, 8, 12, 24]),
    ],
)
def test_get_n_blocks(n_factors, n_generators, n_repetitions, expected):
    n_blocks = get_n_blocks(
        n_factors=n_factors, n_generators=n_generators, n_repetitions=n_repetitions
    )
    assert n_blocks == expected


@pytest.mark.parametrize(
    "n_factors, n_generators, n_repetitions, n_blocks, expected",
    [(3, 0, 1, 2, "ABC"), (3, 0, 1, 4, "AB; AC; BC"), (3, 0, 2, 8, "AB; AC; BC")],
)
def test_get_block_generator(
    n_factors,
    n_generators,
    n_repetitions,
    n_blocks,
    expected,
):
    block_generator = get_block_generator(
        n_factors=n_factors,
        n_blocks=n_blocks,
        n_repetitions=n_repetitions,
        n_generators=n_generators,
    )
    assert block_generator == expected


def test_get_block_generator_invalid():
    with pytest.raises(
        ValueError, match="Blocking can be reached by repetitions only."
    ):
        get_block_generator(
            n_factors=3,
            n_blocks=2,
            n_repetitions=2,
            n_generators=0,
        )
    with pytest.raises(
        ValueError, match="No block generator available for the requested combination."
    ):
        get_block_generator(
            n_factors=3,
            n_blocks=27,
            n_repetitions=2,
            n_generators=0,
        )


@pytest.mark.parametrize(
    "design, block_generator, expected",
    [
        (fracfact("a b c"), "AB; AC; BC", [0, 1, 2, 3, 3, 2, 1, 0]),
        (fracfact("a b c"), "ABC", [0, 1, 1, 0, 1, 0, 0, 1]),
    ],
)
def test_apply_block_generator(design, block_generator, expected):
    apply_block_generator(
        design=fracfact("a b c"),
        gen="AB; AC; BC",
    )
