from bofire.data_models.migration.steps.pre_to_0_3_3 import (  # noqa: F401
    domain,
    kernels,
    strategies,
    surrogates,
)
from bofire.data_models.migration.version import Kind
from bofire.data_models.migration.walker import walk


STEP = "pre_to_0_3_3"

_KIND_TOP_TYPES = {
    # If a payload lacks a top-level "type" tag, the kind argument tells us
    # what to expect. We don't currently need to invent one because every
    # observed strategy/surrogate carries `type`, but bare Domain payloads do
    # not always carry it (it was added recently).
    "domain": "Domain",
}


def run(payload: dict, kind: Kind) -> dict:
    if "type" not in payload and kind in _KIND_TOP_TYPES:
        payload["type"] = _KIND_TOP_TYPES[kind]
    return walk(payload, STEP)
