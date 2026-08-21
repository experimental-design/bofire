import pytest

from bofire.data_models.introspection.paths import (
    PathError,
    PathSegment,
    SegmentKind,
    parse_path,
    unparse,
)


def name(value):
    return PathSegment(SegmentKind.NAME, value)


def index(value):
    return PathSegment(SegmentKind.INDEX, value)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("", ()),
        ("a", (name("a"),)),
        ("a.b", (name("a"), name("b"))),
        (
            "domain.inputs.temperature",
            (name("domain"), name("inputs"), name("temperature")),
        ),
        ('a["b.c"].d', (name("a"), name("b.c"), name("d"))),
        ("a.b\\.c.d", (name("a"), name("b.c"), name("d"))),
        ("x[0]", (name("x"), index(0))),
        ('x["0"]', (name("x"), name("0"))),
        ("x[0].y", (name("x"), index(0), name("y"))),
        ('["a.b"]', (name("a.b"),)),
        ("a.0", (name("a"), name("0"))),
        ('a["say \\"hi\\""]', (name("a"), name('say "hi"'))),
    ],
)
def test_parse_path(path, expected):
    assert parse_path(path) == expected


@pytest.mark.parametrize(
    "path",
    ["", "a", "a.b", 'a["b.c"].d', "x[0]", 'x["0"]', "x[0].y", '["a.b"]', "a.0"],
)
def test_unparse_round_trips(path):
    segments = parse_path(path)
    assert parse_path(unparse(segments)) == segments


def test_bracket_and_backslash_forms_are_equivalent():
    assert parse_path('a["b.c"].d') == parse_path("a.b\\.c.d")


def test_bracket_form_is_emitted_for_keys_needing_escape():
    assert unparse(parse_path("a.b\\.c")) == 'a["b.c"]'


def test_integer_index_differs_from_string_key():
    # A feature keyed "0" would be unreachable if these collapsed.
    assert parse_path("x[0]") != parse_path('x["0"]')
    assert parse_path("x[0]")[-1].is_index
    assert not parse_path('x["0"]')[-1].is_index


@pytest.mark.parametrize(
    "path",
    [
        "a..b",
        "a.",
        ".a",
        "a[",
        "a[1",
        "a]",
        "a[b]",
        'a["b',
        "a\\",
        'a["b"]c',
    ],
)
def test_malformed_paths_raise(path):
    with pytest.raises(PathError):
        parse_path(path)


def test_unquoted_non_integer_index_suggests_the_quoted_form():
    with pytest.raises(PathError, match=r'\["b"\]'):
        parse_path("a[b]")
