"""Path grammar for addressing locations inside a data model.

A path names one location in a schema or in a serialized spec, e.g.
``domain.inputs.temperature.bounds`` or ``acquisition_function.beta``. The
grammar is::

    path    := <empty> | segment (("." segment) | index)*
    index   := "[" (integer | quoted) "]"

``""`` addresses the root. Element keys that contain a ``.`` are written either
bracket-quoted (``domain.inputs["a.b"]``) or backslash-escaped
(``domain.inputs.a\\.b``); both parse identically and :func:`unparse` emits the
bracket form. ``[0]`` is a positional index while ``["0"]`` is the string key
``"0"`` -- a feature keyed ``"0"`` would otherwise be unreachable.

This module is deliberately free of any pydantic or BoFire imports: it maps
strings to segments and back, and nothing else.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union


__all__ = ["PathError", "PathSegment", "SegmentKind", "parse_path", "unparse"]

#: Characters that force a name segment to be emitted in bracket-quoted form.
_UNSAFE_IN_NAME = frozenset('.[]\\"')


class PathError(ValueError):
    """Raised when a path is malformed or cannot be resolved."""


class SegmentKind(str, Enum):
    """Whether a segment addresses something by name or by position."""

    NAME = "name"
    INDEX = "index"


@dataclass(frozen=True)
class PathSegment:
    """One step of a path.

    ``NAME`` segments address a model field or a container element key;
    ``INDEX`` segments address a list position.
    """

    kind: SegmentKind
    value: Union[str, int]

    @property
    def is_index(self) -> bool:
        return self.kind is SegmentKind.INDEX


def _parse_bracket(path: str, start: int) -> Tuple[PathSegment, int]:
    """Parse a ``[...]`` group beginning at ``start``.

    Returns the segment and the index just past the closing bracket.
    """
    i = start + 1
    if i < len(path) and path[i] == '"':
        # Quoted key: ["a.b"]. Backslash escapes the next character.
        i += 1
        chars: List[str] = []
        while i < len(path) and path[i] != '"':
            if path[i] == "\\":
                i += 1
                if i >= len(path):
                    raise PathError(f"{path!r}: trailing backslash inside quoted key")
            chars.append(path[i])
            i += 1
        if i >= len(path):
            raise PathError(f"{path!r}: unterminated quoted key")
        i += 1  # past the closing quote
        if i >= len(path) or path[i] != "]":
            raise PathError(f"{path!r}: expected ']' after quoted key")
        return PathSegment(SegmentKind.NAME, "".join(chars)), i + 1

    # Unquoted: must be an integer index.
    close = path.find("]", i)
    if close == -1:
        raise PathError(f"{path!r}: unterminated '['")
    raw = path[i:close]
    try:
        return PathSegment(SegmentKind.INDEX, int(raw)), close + 1
    except ValueError:
        raise PathError(
            f"{path!r}: '[{raw}]' is not an integer index. "
            f'Use ["{raw}"] to address it as a key.'
        ) from None


def parse_path(path: str) -> Tuple[PathSegment, ...]:
    """Parse a path string into segments.

    Args:
        path: The path to parse. ``""`` yields an empty tuple (the root).

    Returns:
        The parsed segments.

    Raises:
        PathError: If the path is malformed.

    Examples:
        >>> parse_path("domain.inputs.temperature")[-1].value
        'temperature'
        >>> parse_path('domain.inputs["a.b"]')[-1].value
        'a.b'
        >>> parse_path("steps[0]")[-1].kind is SegmentKind.INDEX
        True
    """
    if path == "":
        return ()

    segments: List[PathSegment] = []
    chars: List[str] = []
    i = 0
    # ``started`` distinguishes "no name accumulated yet" from "empty name",
    # so that a bracket directly after a dot is a clear error.
    started = False

    def flush() -> None:
        nonlocal started
        if started:
            segments.append(PathSegment(SegmentKind.NAME, "".join(chars)))
            chars.clear()
            started = False

    while i < len(path):
        ch = path[i]
        if ch == "\\":
            i += 1
            if i >= len(path):
                raise PathError(f"{path!r}: trailing backslash")
            chars.append(path[i])
            started = True
            i += 1
        elif ch == ".":
            if not started:
                raise PathError(f"{path!r}: empty segment before position {i}")
            flush()
            i += 1
            if i >= len(path):
                raise PathError(f"{path!r}: trailing '.'")
        elif ch == "[":
            flush()
            segment, i = _parse_bracket(path, i)
            segments.append(segment)
            # A following '.' is a separator, not part of a name. Anything other
            # than a separator, another bracket, or the end is a missing '.'.
            if i < len(path):
                if path[i] == ".":
                    i += 1
                    if i >= len(path):
                        raise PathError(f"{path!r}: trailing '.'")
                elif path[i] != "[":
                    raise PathError(
                        f"{path!r}: expected '.' or '[' after ']' at position {i}"
                    )
        elif ch == "]":
            raise PathError(f"{path!r}: unmatched ']' at position {i}")
        else:
            chars.append(ch)
            started = True
            i += 1

    flush()
    return tuple(segments)


def _emit_name(value: str, first: bool) -> str:
    """Render a name segment, bracket-quoting it if it needs escaping."""
    if value and not (_UNSAFE_IN_NAME & set(value)):
        return value if first else f".{value}"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'["{escaped}"]'


def unparse(segments: Tuple[PathSegment, ...]) -> str:
    """Render segments back into a path string.

    ``unparse(parse_path(p))`` is stable: it re-parses to the same segments,
    though it normalises backslash-escaped keys to the bracket-quoted form.
    """
    parts: List[str] = []
    for position, segment in enumerate(segments):
        if segment.is_index:
            parts.append(f"[{segment.value}]")
        else:
            parts.append(_emit_name(str(segment.value), first=position == 0))
    return "".join(parts)
