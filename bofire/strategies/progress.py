from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class AskProgressData:
    """Snapshot reported while `Strategy.ask` optimizes an acquisition function
    or, for design of experiments, a design criterion.

    Attributes:
        step: 1-based counter of the reports within one `ask`. For the genetic
            algorithm it is the generation number. For BoTorch it is the running
            count of iterations of the inner optimizer across all optimizer runs.
            Depending on the installed scipy version, restarts that are
            optimized together report once per iteration of the group or once
            each, so the count may differ between environments. An
            exhaustive search over a purely categorical domain has no
            iterations: it counts the batches of up to 2048 choices that were
            scored, over all requested candidates. For design of experiments it
            is the solver iteration.
        value: The acquisition value, or for design of experiments the design
            criterion (higher is better), or None if unavailable. For the
            genetic algorithm it is the best value in the current population.
            For BoTorch it is the value at the iterate just produced, or in an
            exhaustive search the best value scored so far for the candidate
            being selected; when several restarts are optimized jointly it is
            the sum over those restarts. For design of experiments it is the
            design criterion at the current iterate. It is None whenever the
            inner optimizer reports no value, which can happen under linear
            or nonlinear constraints depending on the installed scipy version.
    """

    step: int
    value: Optional[float]


AskProgressCallback = Callable[[AskProgressData], None]


def scipy_progress_callback(callback: AskProgressCallback) -> Callable[[Any], None]:
    """Adapts `callback` to the `callback` argument of `scipy.optimize.minimize`.

    The scipy problem minimizes the negated value that is reported.
    """
    steps = count(1)

    # The parameter MUST be named `intermediate_result`: scipy (and botorch's batched
    # L-BFGS-B) inspect the signature and only then pass an `OptimizeResult`. SLSQP
    # passes the bare iterate on older scipy versions, which carries no value.
    def on_iteration(intermediate_result: Any) -> None:
        fun = getattr(intermediate_result, "fun", None)
        callback(
            AskProgressData(
                step=next(steps),
                value=None if fun is None else -float(fun),
            )
        )

    return on_iteration
