import math
from typing import Optional

import torch
from gpytorch.constraints import Interval
from gpytorch.module import Module


def set_starting_values(model: Module) -> None:
    """Set where the fit of each hyperparameter starts, unless its constraint says so.

    A hyperparameter with a prior starts at the prior's mode, moved into its bounds. One
    without a prior starts where GPyTorch starts it, which for a bound without a
    transform is moved into the bound: a lower bound plus ln 2, an upper bound minus
    ln 2, or the middle of an interval. A starting value given on the constraint is
    kept, and so is a prior mode the constraint cannot take on (e.g. 0 for a positive
    parameter).

    Args:
        model: The model, before it is fitted.
    """
    for name, module, prior, closure, setting_closure in model.named_priors():
        if setting_closure is None:
            continue
        constraint = _constraint_of(module, name.rsplit(".", 1)[-1])
        if constraint is not None and constraint.initial_value is not None:
            continue
        try:
            mode = prior.mode
        except (NotImplementedError, AttributeError):
            continue
        current = closure(module)
        try:
            start = mode.to(current).expand_as(current).clone()
        except RuntimeError:
            continue
        if constraint is not None:
            lower = constraint.lower_bound.to(current)
            upper = constraint.upper_bound.to(current)
            if constraint.enforced:
                if not bool(((start > lower) & (start < upper)).all()):
                    continue
            else:
                start = torch.maximum(torch.minimum(start, upper), lower)
        setting_closure(module, start)

    for name, constraint in model.named_constraints():
        if constraint.enforced or constraint.initial_value is not None:
            continue
        raw = model.get_parameter(name.removesuffix("_constraint"))
        lower = constraint.lower_bound.to(raw)
        upper = constraint.upper_bound.to(raw)
        if bool(((raw >= lower) & (raw <= upper)).all()):
            continue
        start = torch.where(
            torch.isfinite(lower) & torch.isfinite(upper),
            (lower + upper) / 2,
            torch.where(
                torch.isfinite(lower), lower + math.log(2), upper - math.log(2)
            ),
        )
        with torch.no_grad():
            raw.copy_(start.expand_as(raw))


def _constraint_of(module: Module, prior_name: str) -> Optional[Interval]:
    """The constraint of the hyperparameter a prior is placed on, if there is one."""
    parameter = prior_name.removesuffix("_prior")
    # ConstantMean registers the prior on its constant as "mean_prior"
    for candidate in (parameter, "constant" if parameter == "mean" else None):
        if candidate is None:
            continue
        constraint = getattr(module, f"raw_{candidate}_constraint", None)
        if constraint is not None:
            return constraint
    return None
