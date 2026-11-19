from typing import Optional

from bofire.data_models.migration.errors import UnknownVersionError
from bofire.data_models.migration.steps import STEPS
from bofire.data_models.migration.version import BASELINE, SOURCE_PRE, Kind


def migrate(
    payload: dict,
    source: str = SOURCE_PRE,
    target: Optional[str] = None,
    kind: Kind = "strategy",
) -> dict:
    """Migrate a legacy BoFire payload to a newer schema version.

    Pure dict -> dict; never calls model_validate internally. The result is the
    minimum-essential dict that Pydantic will accept; default-valued fields are
    left absent for Pydantic to fill in on validation.

    Args:
        payload: Serialized BoFire data model.
        source: Source version. ``"pre"`` is the sentinel for any pre-baseline
            dump. Future steps will accept explicit version strings.
        target: Target version. Defaults to the migration tool's baseline
            (currently ``0.3.3``).
        kind: One of ``"strategy"``, ``"surrogate"``, ``"domain"``. Used to set
            the top-level discriminator if absent.

    Returns:
        Migrated payload.
    """
    if target is None:
        target = BASELINE
    chain = _resolve_chain(source, target)
    payload = dict(payload)
    for step_fn in chain:
        payload = step_fn(payload, kind)
    return payload


def _resolve_chain(source: str, target: str):
    chain = []
    cursor = source
    while cursor != target:
        for s, t, fn in STEPS:
            if s == cursor:
                chain.append(fn)
                cursor = t
                break
        else:
            raise UnknownVersionError(
                f"No migration step from {cursor!r} towards {target!r}. "
                f"Known steps: {[(s, t) for s, t, _ in STEPS]}"
            )
    return chain
