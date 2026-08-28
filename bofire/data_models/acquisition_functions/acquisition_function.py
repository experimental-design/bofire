import warnings
from typing import Annotated, Any, ClassVar, Dict, Literal, Optional

from pydantic import Field, PositiveFloat, model_validator

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


class MCAcquisitionFunction(BaseModel):
    """Mixin for an acquisition function approximated by Monte Carlo sampling.

    Sampling is what lets such a function score a whole batch of candidates jointly
    rather than one point at a time, at the cost of an approximation whose noise falls
    as the sample count rises.
    """

    n_mc_samples: IntPowerOfTwo = Field(
        default=512,
        description="Number of Monte Carlo samples drawn to approximate the "
        "acquisition value. Higher values reduce the approximation noise and make "
        "candidate selection more reproducible, at proportionally higher cost per "
        "optimization step. Must be a power of two.",
    )


class SupersededByLogVariant(BaseModel):
    """Mixin for an acquisition function whose log-space variant should be used instead.

    The plain formulations underflow to zero once improvements become small, leaving the
    optimizer without a gradient to follow, which the log variants fix. Subclasses name
    their replacement in `log_variant`.
    """

    log_variant: ClassVar[str]

    @model_validator(mode="after")
    def warn_superseded_by_log_variant(self):
        warnings.warn(
            f"{type(self).__name__} is deprecated and will be removed in a future "
            f"release. Use {self.log_variant} instead: it optimizes the same quantity "
            "in log space, which is numerically better behaved.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self


class NoisyAcquisitionFunction(BaseModel):
    """Mixin for an acquisition function that treats the observed results as noisy.

    Rather than taking the best result so far at face value, it integrates over the
    model's belief about the points already observed. Doing so is never wrong — with
    noise-free data it simply costs more compute for the same answer — so the choice is
    one of efficiency rather than correctness.
    """

    prune_baseline: bool = Field(
        default=True,
        description="Whether to drop already-observed points that cannot plausibly be "
        "optimal before evaluating. This shrinks the problem and usually speeds "
        "optimization up without changing the result.",
    )


class SingleObjectiveAcquisitionFunction(AcquisitionFunction):
    """Acquisition function for optimizing one output."""

    type: Any


class MultiObjectiveAcquisitionFunction(AcquisitionFunction):
    """Acquisition function for trading off several outputs against each other.

    It scores a candidate by how much that candidate would enlarge the dominated
    hypervolume, so it needs a reference point and it rewards filling gaps in the Pareto
    front rather than improving any single output.
    """

    type: Any
    alpha: Annotated[float, Field(ge=0)] = Field(
        default=0.0,
        description="Tolerance for approximating the partitioning of the objective "
        "space used in the hypervolume computation. The default of 0 partitions "
        "exactly; larger values trade accuracy for speed, which matters as the number "
        "of objectives grows.",
    )


class qNEI(
    MCAcquisitionFunction,
    NoisyAcquisitionFunction,
    SupersededByLogVariant,
    SingleObjectiveAcquisitionFunction,
):
    """Noisy expected improvement over the best observed result.

    Deprecated in favour of `qLogNEI`. Unlike `qEI`, it does not treat the best
    observation as exact.
    """

    type: Literal["qNEI"] = "qNEI"
    log_variant: ClassVar[str] = "qLogNEI"


class qLogNEI(
    MCAcquisitionFunction, NoisyAcquisitionFunction, SingleObjectiveAcquisitionFunction
):
    """Noisy expected improvement, evaluated in log space.

    Numerically the best-behaved of the improvement-based acquisition functions and the
    usual default: the log formulation keeps a useful gradient in regions where `qNEI`
    underflows to zero and the optimizer would stall.
    """

    type: Literal["qLogNEI"] = "qLogNEI"


class pTS(SingleObjectiveAcquisitionFunction):
    """Pathwise Thompson Sampling acquisition function.

    Draws a sample function from the model and proposes its optimum, so exploration
    comes from the randomness of the draw rather than from an explicit trade-off
    parameter.
    """

    type: Literal["pTS"] = "pTS"


class qEI(
    MCAcquisitionFunction, SupersededByLogVariant, SingleObjectiveAcquisitionFunction
):
    """Expected improvement over the best observed result.

    Deprecated in favour of `qLogEI`. Assumes the best observation is exact, so it suits
    noise-free evaluations; use `qNEI` when measurements are noisy.
    """

    type: Literal["qEI"] = "qEI"
    log_variant: ClassVar[str] = "qLogEI"


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
    getting stuck if used throughout a campaign.
    """

    type: Literal["qSR"] = "qSR"


class qUCB(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """Upper confidence bound: predicted mean plus a multiple of the uncertainty.

    The exploration of this acquisition function is set explicitly, by `beta`, rather
    than following from the form of the score. That parameter is not easy to tune, so
    the explicitness is worth having mainly when a particular trade-off is being
    imposed deliberately.
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


class qEHVI(
    MCAcquisitionFunction, SupersededByLogVariant, MultiObjectiveAcquisitionFunction
):
    """Expected hypervolume improvement over the current Pareto front.

    Deprecated in favour of `qLogEHVI`. Assumes the observed front is exact, so it suits
    noise-free evaluations; use `qNEHVI` when measurements are noisy.
    """

    type: Literal["qEHVI"] = "qEHVI"
    log_variant: ClassVar[str] = "qLogEHVI"


class qLogEHVI(MCAcquisitionFunction, MultiObjectiveAcquisitionFunction):
    """Expected hypervolume improvement, evaluated in log space.

    The log formulation keeps a useful gradient where `qEHVI` underflows to zero, which
    happens readily once the front is well populated and each further gain is small.
    """

    type: Literal["qLogEHVI"] = "qLogEHVI"


class qNEHVI(
    MCAcquisitionFunction,
    NoisyAcquisitionFunction,
    SupersededByLogVariant,
    MultiObjectiveAcquisitionFunction,
):
    """Noisy expected hypervolume improvement over the current Pareto front.

    Deprecated in favour of `qLogNEHVI`. Does not take the observed front as exact,
    which is what makes it suitable for noisy measurements.
    """

    type: Literal["qNEHVI"] = "qNEHVI"
    log_variant: ClassVar[str] = "qLogNEHVI"


class qLogNEHVI(
    MCAcquisitionFunction, NoisyAcquisitionFunction, MultiObjectiveAcquisitionFunction
):
    """Noisy expected hypervolume improvement, evaluated in log space.

    Numerically the best-behaved of the hypervolume acquisition functions and the usual
    default for multi-objective problems.
    """

    type: Literal["qLogNEHVI"] = "qLogNEHVI"


class qMFHVKG(MCAcquisitionFunction, MultiObjectiveAcquisitionFunction):
    """Multi-fidelity hypervolume knowledge gradient.

    Proposes the candidate that maximizes the *global reward* — the maximum of the
    surrogate's mean prediction after the experiment at that candidate has been
    observed. This encourages proposing cheap, low-fidelity candidates now in order to
    know more about the best experiment next time. Use it for a domain with a task or
    fidelity input.
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
        "output feature key. The outputs are combined additively, as a weighted sum of "
        "their variances, so only the ratios between the weights affect which candidate "
        "wins. If not provided, all outputs count equally.",
    )


class qLogPF(MCAcquisitionFunction, SingleObjectiveAcquisitionFunction):
    """MC based batch LogProbability of Feasibility acquisition function.

    It is used to select the next batch of experiments to maximize the
    probability of finding feasible solutions with respect to output
    constraints in the next batch. It can be only used in the SoboStrategy
    and is especially useful in combination with the FeasibleExperimentCondition
    within the StepwiseStrategy.

    It is evaluated against all of the outputs carrying a *constrained* objective. It
    does not account for the optimization objectives, so it provides a way out of an
    infeasible region rather than a high objective value.
    """

    type: Literal["qLogPF"] = "qLogPF"
