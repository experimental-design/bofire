from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Literal, Optional


@dataclass(frozen=True)
class AskOptimizationProgress:
    """Snapshot reported while `Strategy.ask` optimizes an acquisition function
    or, for design of experiments, a design criterion.

    Attributes:
        optimizer: Which optimizer produced the snapshot. "ipopt" and "scipy"
            are the two solvers of the design of experiments strategy.
        step: 1-based counter. For "genetic_algorithm" it is the generation
            number. For "botorch" it is the running count of iterations of the
            inner optimizer across all optimizer runs within one `ask`;
            restarts that are optimized in one batch share an iteration. An
            exhaustive search over a purely categorical domain has no
            iterations: it counts the batches of up to 2048 choices that were
            scored, over all requested candidates. For "ipopt" and "scipy" it
            is the solver iteration.
        max_steps: Upper bound on `step`, constant within one `ask`, so that
            `step / max_steps` is the fraction of the budget used. The
            optimizer stops earlier once it converges, so the last step
            usually stays below it.
        value: The acquisition value, or for design of experiments the design
            criterion (higher is better), or None if unavailable. For
            "genetic_algorithm" it is the best value in the
            current population. For "botorch" it is the value at the iterate
            just produced, or in an exhaustive search the best value scored so
            far for the candidate being selected; when several restarts are
            optimized jointly it is the sum over those restarts. For "ipopt"
            and "scipy" it is the design criterion at the current iterate,
            negated because the solver minimizes it. It is None whenever the
            inner optimizer reports no value, which can happen under linear
            or nonlinear constraints depending on the installed scipy version.
    """

    optimizer: Literal["botorch", "genetic_algorithm", "ipopt", "scipy"]
    step: int
    max_steps: int
    value: Optional[float]


AskProgressCallback = Callable[[AskOptimizationProgress], None]


def scipy_progress_callback(
    callback: AskProgressCallback,
    optimizer: Literal["botorch", "scipy"],
    max_steps: int,
) -> Callable[[Any], None]:
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
            AskOptimizationProgress(
                optimizer=optimizer,
                step=next(steps),
                max_steps=max_steps,
                value=None if fun is None else -float(fun),
            )
        )

    return on_iteration
