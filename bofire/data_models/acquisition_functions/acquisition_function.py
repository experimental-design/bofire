from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field, PositiveFloat

from bofire.data_models.base import BaseModel
from bofire.data_models.types import IntPowerOfTwo


class AcquisitionFunction(BaseModel):
    """Scores how worthwhile a candidate is to run next.

    A strategy proposes the candidates that maximize this score, so the choice of
    acquisition function is the choice of what "worthwhile" means: pure improvement over
    the best result so far, information about the model, coverage of a Pareto front, and
    so on.
    """

    type: Any


class MCAcquisitionFunction(AcquisitionFunction):
    """Acquisition function whose value is approximated by Monte Carlo sampling.

    Sampling is what lets these score a whole batch of candidates jointly rather than one
    point at a time, at the cost of an approximation whose noise falls as the sample
    count rises.
    """

    type: Any
    n_mc_samples: IntPowerOfTwo = Field(
        default=512,
        description="Number of Monte Carlo samples drawn to approximate the "
        "acquisition value. Higher values reduce the approximation noise and make "
        "candidate selection more reproducible, at proportionally higher cost per "
        "optimization step. Must be a power of two.",
    )


class SingleObjectiveAcquisitionFunction(AcquisitionFunction):
    """Acquisition function for optimizing one output."""

    type: Any


class MultiObjectiveAcquisitionFunction(AcquisitionFunction):
    """Acquisition function for trading off several outputs against each other.

    These score a candidate by how much it would enlarge the dominated hypervolume, so
    they need a reference point and they reward filling gaps in the Pareto front rather
    than improving any single output.
    """

    type: Any
    alpha: Annotated[float, Field(ge=0)] = Field(
        default=0.0,
        description="Tolerance for approximating the partitioning of the objective "
        "space used in the hypervolume computation. The default of 0 partitions "
        "exactly; larger values trade accuracy for speed, which matters as the number "
        "of objectives grows.",
    )


class qNEI(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Noisy expected improvement over the best observed result.

    Unlike `qEI`, it does not treat the best observation as exact: it integrates over
    the model's belief about the already-observed points, which is the right choice when
    the measurements are noisy. Prefer `qLogNEI`, whose reformulation optimizes more
    reliably.
    """

    type: Literal["qNEI"] = "qNEI"
    prune_baseline: bool = Field(
        default=True,
        description="Whether to drop already-observed points that are unlikely to be "
        "the best one before evaluating. This shrinks the problem and usually speeds "
        "optimization up without changing the result.",
    )


class qLogNEI(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Noisy expected improvement, evaluated in log space.

    Numerically the best-behaved of the improvement-based acquisition functions and the
    usual default: the log formulation keeps a useful gradient in regions where `qNEI`
    underflows to zero and the optimizer would stall.
    """

    type: Literal["qLogNEI"] = "qLogNEI"
    prune_baseline: bool = Field(
        default=True,
        description="Whether to drop already-observed points that are unlikely to be "
        "the best one before evaluating. This shrinks the problem and usually speeds "
        "optimization up without changing the result.",
    )


class pTS(SingleObjectiveAcquisitionFunction):
    """Pathwise Thompson Sampling acquisition function.

    Draws a sample function from the model and proposes its optimum, so exploration
    comes from the randomness of the draw rather than from an explicit trade-off
    parameter. It takes no settings, and being sample-path based it needs no Monte Carlo
    sample count.
    """

    type: Literal["pTS"] = "pTS"


class qEI(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Expected improvement over the best observed result.

    Assumes the best observation is exact, so it suits noise-free evaluations; use
    `qNEI` when measurements are noisy. Prefer `qLogEI`, whose reformulation optimizes
    more reliably.
    """

    type: Literal["qEI"] = "qEI"


class qLogEI(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Expected improvement over the best observed result, evaluated in log space.

    The log formulation keeps a useful gradient where `qEI` underflows to zero, which is
    common once the model is confident and improvements become small.
    """

    type: Literal["qLogEI"] = "qLogEI"


class qSR(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Simple regret: the expected value of the output itself.

    Purely exploitative — it ignores uncertainty and heads for wherever the model
    predicts the best mean, which makes it a reasonable final-step choice but prone to
    getting stuck if used throughout a campaign. It cannot handle output constraints
    separately and falls back to a constrained Monte Carlo objective.
    """

    type: Literal["qSR"] = "qSR"


class qUCB(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Upper confidence bound: predicted mean plus a multiple of the uncertainty.

    The one acquisition function whose exploration is set explicitly rather than implied,
    which makes it the natural choice when you want direct control over that trade-off.
    It cannot handle output constraints separately and falls back to a constrained Monte
    Carlo objective.
    """

    type: Literal["qUCB"] = "qUCB"
    beta: Annotated[float, Field(ge=0)] = Field(
        default=0.2,
        description="How strongly model uncertainty is rewarded relative to the "
        "predicted mean. Zero is purely exploitative; larger values push candidates "
        "into regions the model knows least about.",
    )


class qPI(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Probability that a candidate improves on the best observed result.

    Scores only whether an improvement happens, not how large it would be, so it tends
    to favour small safe gains over rarer large ones.
    """

    type: Literal["qPI"] = "qPI"
    tau: PositiveFloat = Field(
        default=1e-3,
        description="Temperature used to smooth the improvement indicator so that it "
        "can be differentiated. Smaller values approximate the true probability more "
        "closely but give the optimizer a harsher surface to work on.",
    )


class qEHVI(MCAcquisitionFunction, MultiObjectiveAcquisitionFunction):
    """Expected hypervolume improvement over the current Pareto front.

    Assumes the observed front is exact, so it suits noise-free evaluations; use
    `qNEHVI` when measurements are noisy. Prefer `qLogEHVI`, whose reformulation
    optimizes more reliably.
    """

    type: Literal["qEHVI"] = "qEHVI"


class qLogEHVI(MCAcquisitionFunction, MultiObjectiveAcquisitionFunction):
    """Expected hypervolume improvement, evaluated in log space.

    The log formulation keeps a useful gradient where `qEHVI` underflows to zero, which
    happens readily once the front is well populated and each further gain is small.
    """

    type: Literal["qLogEHVI"] = "qLogEHVI"


class qNEHVI(MCAcquisitionFunction, MultiObjectiveAcquisitionFunction):
    """Noisy expected hypervolume improvement over the current Pareto front.

    Integrates over the model's belief about the already-observed points instead of
    taking the observed front as exact, which is what makes it suitable for noisy
    measurements. Prefer `qLogNEHVI`, whose reformulation optimizes more reliably.
    """

    type: Literal["qNEHVI"] = "qNEHVI"
    prune_baseline: bool = Field(
        default=True,
        description="Whether to drop already-observed points that are unlikely to lie "
        "on the Pareto front before evaluating. This shrinks the problem and usually "
        "speeds optimization up without changing the result.",
    )


class qLogNEHVI(MCAcquisitionFunction, MultiObjectiveAcquisitionFunction):
    """Noisy expected hypervolume improvement, evaluated in log space.

    Numerically the best-behaved of the hypervolume acquisition functions and the usual
    default for multi-objective problems.
    """

    type: Literal["qLogNEHVI"] = "qLogNEHVI"
    prune_baseline: bool = Field(
        default=True,
        description="Whether to drop already-observed points that are unlikely to lie "
        "on the Pareto front before evaluating. This shrinks the problem and usually "
        "speeds optimization up without changing the result.",
    )


class qMFHVKG(MCAcquisitionFunction, MultiObjectiveAcquisitionFunction):
    """Multi-fidelity hypervolume knowledge gradient.

    Scores how much running a candidate would improve the Pareto front you could report
    afterwards, rather than how good the candidate itself looks. That look-ahead is what
    lets it decide to spend a cheap low-fidelity evaluation for information alone, so it
    is the multi-objective choice for a domain with a task or fidelity input.
    """

    type: Literal["qMFHVKG"] = "qMFHVKG"
    num_fantasies: int = Field(
        default=8,
        description="Number of fantasy outcomes simulated per candidate when looking "
        "ahead. Higher values estimate the knowledge gain better and cost "
        "proportionally more, since each fantasy conditions its own model.",
    )
    num_pareto: int = Field(
        default=10,
        description="Number of points used to represent the Pareto front when scoring "
        "a candidate's look-ahead value.",
    )
    n_mc_samples: IntPowerOfTwo = Field(
        default=32,
        description="Number of Monte Carlo samples drawn to approximate the "
        "acquisition value. Lower than for the other acquisition functions because "
        "each sample carries the cost of the look-ahead. Must be a power of two.",
    )


class qNegIntPosVar(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Negative integrated posterior variance, for active learning.

    Scores a candidate by how much it would reduce the model's overall uncertainty, not
    by how good its outcome is likely to be. Use it when the goal is a model that is
    accurate everywhere rather than a single optimum.
    """

    type: Literal["qNegIntPosVar"] = "qNegIntPosVar"
    weights: Optional[Dict[str, PositiveFloat]] = Field(
        default=None,
        description="How much reducing the uncertainty of each output counts, keyed by "
        "output feature key. If not provided, all outputs count equally. The values are "
        "normalized, so only their ratios matter.",
    )


class qLogPF(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """MC based batch LogProbability of Feasibility acquisition function.

    It is used to select the next batch of experiments to maximize the
    probability of finding feasible solutions with respect to output
    constraints in the next batch. It can be only used in the SoboStrategy
    and is especially useful in combination with the FeasibleExperimentCondition
    within the StepwiseStrategy.

    It optimizes for feasibility alone and ignores the objectives entirely, so it is a
    way out of an infeasible region rather than a way to a good result.
    """

    type: Literal["qLogPF"] = "qLogPF"
