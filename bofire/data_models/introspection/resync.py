"""Keeping a spec's derived copies consistent with a patched domain.

A serialized strategy spec duplicates domain data. ``surrogate_specs`` holds a
copy of every feature it models, so on a three-objective problem a single input
key appears four times. Patching only ``domain.inputs.temperature.bounds`` and
rebuilding therefore raises ``Features with key temperature are incompatible.``
-- ``BotorchSurrogates._check_compability`` compares inputs by full pydantic
value equality, not by key.

That strictness is what makes the fix simple: a stale copy needs no field-by-field
merge, only wholesale replacement. The rule is

    replace any nested value that still equals the *pre-patch* domain feature
    with the post-patch one.

Copies that were in sync track the change; copies that had deliberately
diverged -- a ``StepwiseStrategy`` step legitimately searching narrower bounds,
which ``validate_domain_compatibility`` explicitly permits -- are left alone.
Hyperparameter search spaces such as ``SingleTaskGPHyperconfig.inputs`` are
untouched for the same reason: their features never equalled a domain feature.

Structural edits (a changed input or output *key set*) cannot be repaired by
replacement, because copies must gain or lose entries. For those, a block that
is provably auto-derived is dropped so BoFire's own validator regenerates it;
anything the caller customised is refused rather than silently rebuilt.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


__all__ = ["ResyncError", "ResyncReport", "resync"]


class ResyncError(ValueError):
    """Raised when a patched domain leaves derived state that cannot be repaired."""


@dataclass
class ResyncReport:
    """What the resync pass had to touch beyond the caller's own changes."""

    propagated: List[str] = field(default_factory=list)
    reset: List[str] = field(default_factory=list)


def _shape(domain: dict) -> Tuple[Any, ...]:
    """The part of a domain that derived blocks are keyed on.

    Not just the feature key sets: ``MoboStrategy.ref_point`` is derived from the
    outputs that carry a *maximise/minimise/close-to-target* objective, so
    promoting an outcome from a constraint-shaped objective to a real one changes
    what must be regenerated even though no key was added or removed.
    """
    inputs = sorted(_features_by_key(domain.get("inputs")))
    objectives = {
        key: (feature.get("objective") or {}).get("type")
        for key, feature in _features_by_key(domain.get("outputs")).items()
    }
    return (tuple(inputs), tuple(sorted(objectives.items())))


def _features_by_key(container: Any) -> Dict[str, dict]:
    if not isinstance(container, dict):
        return {}
    return {
        f["key"]: f
        for f in container.get("features", [])
        if isinstance(f, dict) and "key" in f
    }


def _replace_equal(
    node: Any,
    replacements: List[Tuple[dict, dict]],
    path: str,
    out: List[str],
) -> None:
    """Replace, in place, every nested value equal to a pre-patch feature.

    Equality is the same test ``_check_compability`` applies, so a copy that
    would have been rejected is exactly the copy that gets updated.
    """
    if isinstance(node, list):
        for index, item in enumerate(node):
            for old, new in replacements:
                if item == old:
                    node[index] = copy.deepcopy(new)
                    out.append(f"{path}[{index}]")
                    break
            else:
                _replace_equal(item, replacements, f"{path}[{index}]", out)
    elif isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            for old, new in replacements:
                if value == old:
                    node[key] = copy.deepcopy(new)
                    out.append(child)
                    break
            else:
                _replace_equal(value, replacements, child, out)


def _derive_surrogate_specs(root_cls: type, domain_dict: dict) -> Optional[dict]:
    """What ``surrogate_specs`` would be if BoFire derived it from this domain.

    Uses *root_cls* rather than ``BotorchStrategy`` because subclasses override
    the per-output factory -- ``MultiFidelityVarianceBasedStrategy`` forces a
    ``MultiTaskGPSurrogate``, for instance.
    """
    from bofire.data_models.domain.api import Domain
    from bofire.data_models.surrogates.api import BotorchSurrogates

    generate = getattr(root_cls, "_generate_surrogate_specs", None)
    if generate is None:
        return None
    try:
        empty = BotorchSurrogates(surrogates=[])
        generate(Domain(**domain_dict), empty)
        return empty.model_dump(mode="json")
    except Exception:  # noqa: BLE001 - an underivable block is simply not auto-derived
        return None


def _derive_ref_point(domain_dict: dict) -> Optional[dict]:
    """What ``MoboStrategy.ref_point`` would be if derived from this domain."""
    from bofire.data_models.domain.api import Domain
    from bofire.data_models.objectives.api import (
        CloseToTargetObjective,
        MaximizeObjective,
        MinimizeObjective,
    )
    from bofire.data_models.strategies.predictives.mobo import (
        AbsoluteMovingReferenceValue,
        ExplicitReferencePoint,
    )

    try:
        domain = Domain(**domain_dict)
        keys = domain.outputs.get_keys_by_objective(
            [MaximizeObjective, MinimizeObjective, CloseToTargetObjective],
        )
        return ExplicitReferencePoint(
            values={
                k: AbsoluteMovingReferenceValue(orient_at_best=False, offset=0.0)
                for k in keys
            }
        ).model_dump(mode="json")
    except Exception:  # noqa: BLE001
        return None


#: Blocks that BoFire regenerates from the domain when absent. Each entry maps a
#: field name to a function producing what auto-derivation *would* yield, so a
#: customised block can be told apart from a derived one and refused instead of
#: silently rebuilt. ``None`` means "no derivation exists": the block is reset and
#: reported, because a structural change invalidates it outright.
_DERIVED_BLOCKS: Dict[str, Optional[Callable[[type, dict], Optional[dict]]]] = {
    "surrogate_specs": lambda cls, dom: _derive_surrogate_specs(cls, dom),
    "ref_point": lambda cls, dom: _derive_ref_point(dom),
    "fidelity_cost_model_spec": None,
    "sampling": None,
}


def _resync_structural(
    spec: dict,
    pre_spec: dict,
    pre_domain: dict,
    post_domain: dict,
    root_cls: type,
    path: str,
    report: ResyncReport,
) -> None:
    """Regenerate derived blocks that the new domain shape has invalidated.

    A block is only touched when it is genuinely stale -- replacement may already
    have brought it into line -- and only reset when it was derived rather than
    chosen. "Derived" is judged against the *pre-patch* spec, since replacement
    has since edited the working copy.
    """
    refusals = []
    for name, derive in _DERIVED_BLOCKS.items():
        current = spec.get(name)
        if current is None:
            continue
        if derive is None:
            spec.pop(name)
            report.reset.append(f"{path}{name}" if path else name)
            continue

        if current == derive(root_cls, post_domain):
            continue  # already consistent with the new domain

        was_derived = pre_spec.get(name) == derive(root_cls, pre_domain)
        if was_derived:
            spec.pop(name)
            report.reset.append(f"{path}{name}" if path else name)
        else:
            refusals.append(f"{path}{name}" if path else name)

    if refusals:
        raise ResyncError(
            "The domain changed shape in a way that cannot be reconciled with "
            "customised derived state. Patch these alongside the domain, or "
            "reset them to their defaults first: " + ", ".join(sorted(refusals))
        )


def _resync_node(
    spec: Any,
    pre_spec: Any,
    path: str,
    report: ResyncReport,
) -> None:
    """Resync one strategy spec, then recurse into any nested strategies.

    ``StepwiseStrategy`` needs the recursion: ``validate_domain_compatibility``
    ignores bounds and allowed categories and compares only the key intersection,
    so a stale sub-strategy domain validates cleanly and silently optimises the
    wrong problem.

    Each nested strategy is compared against *its own* pre-patch domain, not the
    outer one. A step is allowed to search a different space than its parent, and
    judging it by the parent's key set would read a deliberate difference as a
    structural change.
    """
    if not isinstance(spec, dict) or not isinstance(pre_spec, dict):
        return
    pre_domain = pre_spec.get("domain")
    post_domain = spec.get("domain")
    if not isinstance(pre_domain, dict) or not isinstance(post_domain, dict):
        return

    from bofire.data_models.introspection.registry import UnknownNameError, lookup

    try:
        root_cls = lookup(spec["type"]) if isinstance(spec.get("type"), str) else None
    except UnknownNameError:
        root_cls = None

    # Replacement first: it repairs every copy that was still in sync, and is
    # correct whether or not the shape also changed.
    replacements = []
    for side in ("inputs", "outputs"):
        before = _features_by_key(pre_domain.get(side))
        after = _features_by_key(post_domain.get(side))
        for key, old in before.items():
            new = after.get(key)
            if new is not None and new != old:
                replacements.append((old, new))
    if replacements:
        for name, value in spec.items():
            if name == "domain":
                continue
            _replace_equal(value, replacements, f"{path}{name}", report.propagated)

    # Copies cannot gain or lose entries by replacement, so a changed shape needs
    # the derived blocks regenerated instead.
    if _shape(pre_domain) != _shape(post_domain) and root_cls is not None:
        _resync_structural(
            spec, pre_spec, pre_domain, post_domain, root_cls, path, report
        )

    # Recurse into nested strategies (``StepwiseStrategy.steps[].strategy_data``).
    # Only where the step lists still line up; if the patch added or removed a
    # step there is no pre-patch counterpart to compare against, and the root
    # rebuild remains the backstop.
    post_steps = spec.get("steps") or []
    pre_steps = pre_spec.get("steps") or []
    if len(post_steps) == len(pre_steps):
        for index, (post_step, pre_step) in enumerate(zip(post_steps, pre_steps)):
            if not (isinstance(post_step, dict) and isinstance(pre_step, dict)):
                continue
            _resync_node(
                post_step.get("strategy_data"),
                pre_step.get("strategy_data"),
                f"{path}steps[{index}].strategy_data.",
                report,
            )


def resync(spec: dict, pre_spec: dict) -> ResyncReport:
    """Reconcile *spec*'s derived copies with its patched domain, in place.

    Args:
        spec: The working spec, already carrying the caller's changes.
        pre_spec: The spec as it was *before* those changes.

    Returns:
        What was touched beyond the caller's own changes.

    Raises:
        ResyncError: If the input/output key set changed and the spec carries
            customised derived state that cannot be regenerated safely.
    """
    report = ResyncReport()
    _resync_node(spec, pre_spec, "", report)
    return report
