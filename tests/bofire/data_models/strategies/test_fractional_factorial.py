import pytest

from bofire.data_models.strategies.api import FractionalFactorialStrategy


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
def test_get_generator(n_factors, n_generators, expected):
    assert (
        FractionalFactorialStrategy.get_generator(n_factors, n_generators) == expected
    )


@pytest.mark.parametrize(
    "n_factors, n_generators",
    [(2, 1), (3, 2), (4, 3), (4, 2), (5, 3), (6, 4), (7, 5), (8, 5)],
)
def test_get_generator_invalid(n_factors, n_generators):
    with pytest.raises(
        ValueError,
        match="Design not possible, as main factors are confounded with each other.",
    ):
        FractionalFactorialStrategy.get_generator(n_factors, n_generators)


@pytest.mark.parametrize(
    "n_factors, generator, message",
    [
        (2, "a b c", "Generator does not match the number of factors."),
        (2, "a a", "Main factors are confounded with each other."),
        (2, "a c", "Main factors are not in alphabetical order."),
        (5, "a b c ab ab", "Generators are not unique."),
        (5, "a b c ab ad", "Generators are not valid."),
        (2, "ab ac", "Generators are not valid."),
    ],
)
def test_validate_generator_invalid(n_factors: int, generator: str, message: str):
    with pytest.raises(ValueError, match=message):
        FractionalFactorialStrategy.validate_generator(n_factors, generator)
