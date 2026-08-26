"""Patching a serialized data model spec at addressed paths.

``describe`` lets a caller find out what a location accepts; this writes to the
same locations, using the same path grammar, so a path reported by ``describe``
can be handed straight back here.

The contract is JSON in, JSON out: a caller -- typically an LLM agent -- passes a
serialized spec and a list of ``{"path", "value"}`` changes and gets a new,
validated, canonical spec back. Pydantic is an internal detail; no model object
crosses the boundary.

Validation is a single stage. Writes go into a working copy and the whole root
model is then rebuilt, which is authoritative in a way that per-field validation
is not: pydantic's ``validate_assignment`` re-runs the validators of the model
whose field is assigned, but not those of its ancestors, so assigning two levels
down leaves cross-field rules unchecked. The rebuild also produces better errors
than validating a value in isolation, because the root's fields carry their
discriminators -- a bad tag yields *"Input tag 'qFoo' found using 'type' does not
match any of the expected tags: ..."* rather than one error per union member.
"""

import copy
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bofire.data_models.introspection.paths import PathError, parse_path
from bofire.data_models.introspection.registry import lookup
from bofire.data_models.introspection.render import summarize
from bofire.data_models.introspection.resolver import resolve
from bofire.data_models.introspection.resync import ResyncError, resync


__all__ = ["PatchError", "ResyncError", "reconfigure"]


class PatchError(ValueError):
    """Raised when a change cannot be applied to the spec."""


def _normalise(changes: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Any]]:
    """Validate the shape of the change list before touching anything."""
    normalised = []
    for index, change in enumerate(changes):
        if not isinstance(change, Mapping):
            raise PatchError(
                f"changes[{index}] must be a mapping, got {type(change).__name__}."
            )
        unknown = set(change) - {"path", "value"}
        if unknown:
            raise PatchError(
                f"changes[{index}] has unexpected keys {sorted(unknown)}; "
                f"expected 'path' and 'value'."
            )
        if "path" not in change:
            raise PatchError(f"changes[{index}] is missing 'path'.")
        if "value" not in change:
            raise PatchError(f"changes[{index}] is missing 'value'.")
        if not isinstance(change["path"], str):
            raise PatchError(f"changes[{index}]['path'] must be a string.")
        normalised.append((change["path"], change["value"]))
    return normalised


def _write(working: dict, path: str, value: Any) -> dict:
    """Apply one change to *working*, returning the (possibly replaced) root."""
    if parse_path(path) == ():
        if not isinstance(value, dict):
            raise PatchError("Replacing the root requires a spec object.")
        return copy.deepcopy(value)

    cursor = resolve(path, instance=working)
    if cursor.view is None:
        raise PathError(f"{path}: does not address a value in this spec.")
    parent, accessor = cursor.view.target()
    if parent is None or accessor is None:
        raise PathError(f"{path}: cannot be written to.")
    parent[accessor] = copy.deepcopy(value)
    return working


def reconfigure(
    spec: Mapping[str, Any],
    changes: Sequence[Mapping[str, Any]],
    rationale: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply *changes* to a serialized spec and return a new validated one.

    Args:
        spec: A serialized data model carrying a ``type`` discriminator, as
            produced by ``model_dump(mode="json")``.
        changes: ``[{"path": ..., "value": ...}, ...]``. Each replaces the whole
            value at its path; paths use the grammar ``describe`` reports. They
            are applied in order against the evolving copy, and all-or-nothing --
            on any failure *spec* is left untouched.
        rationale: Why this reconfiguration is being made. Echoed back in the
            result so a campaign log records the reasoning alongside the edit.

    Returns:
        ``{"spec", "summary", "applied", "propagated", "reset", "rationale"}``.
        ``propagated`` and ``reset`` name the derived copies that had to be
        touched to keep the spec self-consistent, so nothing happens silently.

    Raises:
        PatchError: If a change is malformed.
        PathError: If a path does not resolve.
        ResyncError: If a changed input/output key set cannot be reconciled with
            customised derived state.
        pydantic.ValidationError: If the result is not a valid spec. The error's
            ``loc`` is in dump space, e.g.
            ``('domain', 'inputs', 'features', 0, 'ContinuousInput', 'bounds')``.

    Note:
        The first result is *canonicalised*: values are coerced to their declared
        types, so a ``bounds`` tuple comes back as a list and ``w: 1`` as ``1.0``.
        Repeated calls are stable -- dump, rebuild, dump is a fixpoint.

    Example:
        >>> out = reconfigure(                                    # doctest: +SKIP
        ...     spec,
        ...     [{"path": "acquisition_function",
        ...       "value": {"type": "qUCB", "beta": 2.0}}],
        ...     "posterior untrustworthy at 8 trials; widen exploration",
        ... )
        >>> out["summary"]["acquisition_function"]                # doctest: +SKIP
        'qUCB'
    """
    if not isinstance(spec, Mapping) or not isinstance(spec.get("type"), str):
        raise PatchError(
            "spec must be a serialized data model carrying a 'type' discriminator."
        )
    normalised = _normalise(changes)
    root_cls = lookup(spec["type"])

    working = copy.deepcopy(dict(spec))
    pre_spec = copy.deepcopy(dict(spec))

    for path, value in normalised:
        working = _write(working, path, value)

    report = None
    if working.get("domain") != pre_spec.get("domain"):
        report = resync(working, pre_spec)

    rebuilt = root_cls(**working)
    return {
        "spec": rebuilt.model_dump(mode="json"),
        "summary": summarize(rebuilt),
        "applied": [path for path, _ in normalised],
        "propagated": report.propagated if report else [],
        "reset": report.reset if report else [],
        "rationale": rationale,
    }


def reconfigure_json(spec_json: str, changes_json: str, rationale: str = "") -> str:
    """String-in, string-out wrapper for transports that speak raw JSON."""
    return json.dumps(
        reconfigure(json.loads(spec_json), json.loads(changes_json), rationale)
    )
