import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    SampleReducingMCAcquisitionFunction,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.sampling.base import MCSampler

from bofire.data_models.acquisition_functions.api import qMFMES, qMFVariance
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import TaskInput
from bofire.data_models.objectives.api import MaximizeObjective, Objective
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityStrategy as DataModel,
)
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.predictives.sobo import SoboStrategy
from bofire.strategies.random import RandomStrategy
from bofire.utils.torch_tools import tkwargs


class qMultiFidelityVariance(SampleReducingMCAcquisitionFunction):
    r"""MC-based Variance Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].) Since we only consider the variance, we get the following
    expression.

    `qVariance = E(max(|Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.


    """

    def __init__(
        self,
        model: Model,
        beta: float,
        fidelity_thresholds: torch.Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[torch.Tensor] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.beta_prime = self._get_beta_prime(beta=beta)
        self.fidelity_thresholds = fidelity_thresholds

    def _get_beta_prime(self, beta: float) -> float:
        return math.sqrt(beta * math.pi / 2)

    def _sample_forward(self, obj: torch.Tensor) -> torch.Tensor:
        r"""Evaluate qMultiFidelityVariance per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of acquisition values.
        """
        mean = obj.mean(dim=0)
        return self.beta_prime * (obj - mean).abs()

    def forward(self, X: torch.Tensor):
        r"""Compute acquisition values for a batch of a design point with different fidelities.

        Since the acquisition function depends on other fidelities, we need to
        share information across a batch of samples across fidelities. We therefore
        need to override the forward method to handle this.

        We return a simplified acquisition function, that is simply 1 / (m+1) if
        the fidelity is above the variance threshold, and 0 otherwise. Maximizing
        this will give the smallest fidelity that is above the threshold.

        Args:
            X: A `batch_shape x q=1 x d`-dim Tensor. X must be ordered from lowest
                to highest fidelity.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values.
        """
        acqf_values = super().forward(X)

        fidelity_threshold_scale = self.model.outcome_transform.stdvs.item()
        fidelity_thresholds = self.fidelity_thresholds * fidelity_threshold_scale
        fidelity_thresholds = fidelity_thresholds.view(
            *([1] * (acqf_values.ndim - 1)), -1
        )
        above_threshold = acqf_values > fidelity_thresholds
        above_threshold[..., -1] = True  # selecting highest fidelity is always allowed

        acqf_indicator = (
            1 / (1 + torch.arange(above_threshold.size(-1))) * above_threshold.float()
        )
        return acqf_indicator


def _gen_candidate_set(
    domain: Domain,
    transform_specs: InputTransformSpecs,
    num_candidates: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a candidate set for Gumbel sampling."""
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(
            domain=domain,
            fallback_sampling_method=SamplingMethodEnum.SOBOL,
            seed=seed,
        ),
    )
    candidate_df = random_strategy.ask(num_candidates)
    candidate_set = domain.inputs.transform(
        experiments=candidate_df,
        specs=transform_specs,
    )
    return torch.from_numpy(candidate_set.to_numpy()).to(**tkwargs)


def get_mf_acquisition_function(
    acquisition_function_name: str,
    model: Model,
    target_fidelities: dict[int, float],
    objective: MCAcquisitionObjective,
    maximize: bool = True,
    X_pending: Optional[torch.Tensor] = None,
    mc_samples: int = 512,
    seed: Optional[int] = None,
    *,
    beta: Optional[float] = None,
    fidelity_thresholds: Optional[torch.Tensor] = None,
    fidelity_costs: Optional[list[float]] = None,
    candidate_set: Optional[torch.Tensor] = None,
):
    """Convenience function for initialiing multi-fidelity acquisition functions.

    Mirrors the signature of botorch.acquisition.factory.get_acquisition_function.
    """

    # we require a posterior transform since the MultiTaskGP model has
    # model.num_outputs > 1, even though it is in fact a single output model.
    posterior_transform = ScalarizedPosteriorTransform(weights=torch.tensor([1.0]))
    # TODO: use proper cost model
    fidelity_task_idx = list(target_fidelities.keys())[0]

    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    if acquisition_function_name == "qMFMES":
        if candidate_set is None:
            raise ValueError("`candidate_set` must not be None for qMFMES.")
        if fidelity_costs is None:
            raise ValueError("`fidelity_costs` must not be None for qMFMES.")
        fidelity_fixed, fidelity_gradient = (
            fidelity_costs[0],
            fidelity_costs[1] - fidelity_costs[0],
        )
        cost_model = AffineFidelityCostModel(
            fidelity_weights={fidelity_task_idx: fidelity_gradient},
            fixed_cost=fidelity_fixed,
        )
        cost_aware_utility = InverseCostWeightedUtility(cost_model)

        return qMultiFidelityMaxValueEntropy(
            model=model,
            candidate_set=candidate_set,  # type: ignore
            project=project,
            posterior_transform=posterior_transform,
            cost_aware_utility=cost_aware_utility,
            X_pending=X_pending,
            maximize=maximize,
        )

    elif acquisition_function_name == "qMFVariance":
        if beta is None:
            raise ValueError("`beta` must not be None for qMFVariance.")
        if fidelity_thresholds is None:
            raise ValueError("`fidelity_thresholds` must not be None for qMFVariance.")
        return qMultiFidelityVariance(
            model=model,
            beta=beta,
            fidelity_thresholds=fidelity_thresholds,
            posterior_transform=posterior_transform,
            objective=objective,
            X_pending=X_pending,
        )

    raise NotImplementedError(
        f"Unknown acquisition function {acquisition_function_name}"
    )


class MultiFidelityStrategy(SoboStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.task_feature_key = self.domain.inputs.get_keys(TaskInput)[0]
        self.fidelity_acquisition_function = data_model.fidelity_acquisition_function

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        """Generate new candidates (x, m).

        This is a greedy optimization of the acquisition function. We first
        optimize the acqf for the target fidelity to generate a candidate x,
        then select a target fidelity.

        We do this procedure greedily in line with [Folch et al. 2023]. This has
        the advantage of being simpler and faster, as we only need to evaluate
        the fidelity acquisition function M times. It also allows more freedom
        in the choice of design-space acquisition function, as well as enabling a
        more flexible choice of surrogate models.

        Args:
            candidate_count (int): number of candidates to be generated

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        if candidate_count > 1:
            raise NotImplementedError("Batch optimization is not yet implemented")

        self._verify_all_fidelities_observed()

        task_feature: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        # only optimize the input x on the target fidelity
        # we fix the fidelity by setting all other fidelities to 'not allowed'
        prev_allowed = task_feature.allowed
        task_feature.allowed = [fidelity == 0 for fidelity in task_feature.fidelities]
        x = super()._ask(candidate_count)
        task_feature.allowed = prev_allowed
        fidelity_cand = self.select_fidelity_candidate(x)
        pred = self.predict(fidelity_cand)
        return pd.concat((fidelity_cand, pred), axis=1)

    def _get_fidelity_acqf(
        self, fidelity_input: TaskInput
    ) -> qMultiFidelityVariance | qMultiFidelityMaxValueEntropy:
        _, X_pending = self.get_acqf_input_tensors()
        assert self.model is not None and self.experiments is not None

        fidelity_input_idx = self.domain.inputs.get_keys().index(fidelity_input.key)
        # TODO: target fidelity is not necessarily at index 0

        # determine sense of optimization (max/min)
        # qMFMES doesn't take an `objective` argument, so we need `maximize`
        (
            objective_callable,
            _,
            _,
        ) = self._get_objective_and_constraints()
        target_feature = self.domain.outputs.get_by_objective(includes=Objective)[0]
        maximize = isinstance(target_feature.objective, MaximizeObjective)  # type: ignore

        fidelity_acqf = get_mf_acquisition_function(
            acquisition_function_name=self.fidelity_acquisition_function.__class__.__name__,
            model=self.model,
            target_fidelities={fidelity_input_idx: 0.0},
            objective=objective_callable,
            maximize=maximize,
            X_pending=X_pending,
            beta=(
                self.fidelity_acquisition_function.beta
                if isinstance(self.fidelity_acquisition_function, qMFVariance)
                else 0.2
            ),
            fidelity_thresholds=(
                torch.atleast_1d(
                    torch.tensor(
                        self.fidelity_acquisition_function.fidelity_thresholds,
                        **tkwargs,
                    )
                )
                if isinstance(self.fidelity_acquisition_function, qMFVariance)
                else None
            ),
            candidate_set=_gen_candidate_set(
                domain=self.domain,
                transform_specs=self.input_preprocessing_specs,
                num_candidates=1000,
            )
            if isinstance(self.fidelity_acquisition_function, qMFMES)
            else None,
            fidelity_costs=self.fidelity_acquisition_function.fidelity_costs
            if isinstance(self.fidelity_acquisition_function, qMFMES)
            else None,
        )

        return fidelity_acqf

    def select_fidelity_candidate(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        """Select the fidelity for a given input.

        Uses the variance based approach (see [Kandasamy et al. 2016,
        Folch et al. 2023]) to select the lowest fidelity that has a variance
        exceeding a threshold. If no such fidelity exists, pick the target fidelity

        Args:
            X (pd.DataFrame): optimum input of target fidelity

        Returns:
            pd.DataFrame: selected fidelity and prediction
        """
        fidelity_input: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        assert fidelity_input.allowed is not None

        sorted_fidelities = np.argsort(fidelity_input.fidelities)[::-1]
        num_fidelities = len(fidelity_input.fidelities)

        fidelity_acqf = self._get_fidelity_acqf(fidelity_input)

        X_fidelity_batched = X.loc[
            X.index.repeat(num_fidelities),
            self.domain.inputs.get_keys(excludes=TaskInput),
        ]
        sorted_fidelity_labels = [
            fidelity_input.categories[f] for f in sorted_fidelities
        ]
        X_fidelity_batched[self.task_feature_key] = sorted_fidelity_labels * len(X)
        X_fidelity_batched_transformed = self.domain.inputs.transform(
            experiments=X_fidelity_batched, specs=self.input_preprocessing_specs
        )
        X_fidelity_batched_tensor = (
            torch.from_numpy(X_fidelity_batched_transformed.to_numpy())
            .to(**tkwargs)
            .unsqueeze(-2)
        )
        with torch.no_grad():
            # since we optimize over a discrete set of fidelities, there is
            # no need to compute gradients
            acqf_values = fidelity_acqf(X_fidelity_batched_tensor)
        chosen_fidelity_idx = int(torch.argmax(acqf_values).item())
        candidate = X_fidelity_batched.iloc[[chosen_fidelity_idx]]
        return candidate

    def _verify_all_fidelities_observed(self) -> None:
        """Get all fidelities that have at least one observation.

        We use this instead of overriding `has_sufficient_experiments` to provide
        a more descriptive error message."""
        assert self.experiments is not None
        observed_fidelities = set(self.experiments[self.task_feature_key].unique())
        allowed_fidelities = set(
            self.domain.inputs.get_by_key(
                self.task_feature_key
            ).get_allowed_categories()  # type: ignore
        )
        missing_fidelities = allowed_fidelities - observed_fidelities
        if missing_fidelities:
            raise ValueError(f"Some tasks have no experiments: {missing_fidelities}")
