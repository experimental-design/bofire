from typing import Callable, Dict, Tuple


Normalizer = Callable[[dict], dict]

NORMALIZERS: Dict[Tuple[str, str], Normalizer] = {}


def normalizer(step: str, *type_tags: str):
    """Register a per-type normalizer for a given migration step.

    A normalizer is a pure dict->dict function that mutates only what would
    fail Pydantic validation. The walker dispatches on the dict's `type` tag.
    """

    def deco(fn: Normalizer) -> Normalizer:
        for tag in type_tags:
            NORMALIZERS[(step, tag)] = fn
        return fn

    return deco


def get_normalizer(step: str, type_tag: str) -> Normalizer:
    return NORMALIZERS.get((step, type_tag), _identity)


def _identity(p: dict) -> dict:
    return p
