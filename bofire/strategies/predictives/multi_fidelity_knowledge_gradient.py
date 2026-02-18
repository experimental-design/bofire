from collections.abc import Callable
from typing import List

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.cost_aware import CostAwareUtility, InverseCostWeightedUtility
from botorch.acquisition.multi_objective import (
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
)
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.get_sampler import get_sampler
from botorch.sampling.list_sampler import ListSampler
from pydantic import PositiveInt
from torch import Tensor
from typing_extensions import Self

from bofire.data_models.acquisition_functions.api import qMFHVKG
from bofire.data_models.api import Domain
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousTaskInput, DiscreteTaskInput
from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.outlier_detection.outlier_detections import OutlierDetections
from bofire.data_models.strategies.api import ExplicitReferencePoint
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.strategies.predictives.multi_fidelity_knowledge_gradient import (
    MultiFidelityHVKGStrategy as DataModel,
)
from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.strategies.strategy import make_strategy
from bofire.utils.multiobjective import get_ref_point_mask, infer_ref_point
from bofire.utils.torch_tools import get_multiobjective_objective, tkwargs


def _map_cost_model_to_botorch(
    inputs: Inputs, features2idx: dict[str, tuple[int]]
) -> tuple[AffineFidelityCostModel, dict[int, float]]:
    """Convert the task inputs to a cost model for the cost-weighted utility."""

    def _target_fidelity(
        task_input: ContinuousTaskInput | DiscreteTaskInput, weight: float
    ) -> float:
        # if the cost is positive, then the upper bound is the highest fidelity.
        return task_input.upper_bound if weight > 0 else task_input.lower_bound

    task_inputs = inputs.get([ContinuousTaskInput, DiscreteTaskInput])

    fidelity_weights = {
        features2idx[task_input.key][0]: task_input.fidelity_cost_weight
        for task_input in task_inputs
    }

    total_fixed_cost = sum(task_input.fidelity_fixed_cost for task_input in task_inputs)

    cost_function = AffineFidelityCostModel(
        fidelity_weights=fidelity_weights,
        fixed_cost=total_fixed_cost,
    )
    target_fidelities = {
        (idx := features2idx[task_input.key][0]): _target_fidelity(
            task_input, fidelity_weights[idx]
        )
        for task_input in task_inputs
    }

    return cost_function, target_fidelities


class MultiFidelityHVKGStrategy(BotorchStrategy):
    """Use the MFHVKG AF for a multi-objective, multi-fidelity problem.

    This does not inherit from MoboStrategy since the acqf requires some additional
    custom setup.

    Implementation based on https://botorch.org/docs/tutorials/Multi_objective_multi_fidelity_BO/"""

    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.task_feature_keys = self.domain.inputs.get_keys(
            [ContinuousTaskInput, DiscreteTaskInput]
        )
        self.acquisition_function = data_model.acquisition_function

        # assert isinstance(data_model.ref_point, ExplicitReferencePoint)
        assert not isinstance(data_model.ref_point, dict)
        self.ref_point: ExplicitReferencePoint | None = data_model.ref_point
        self.ref_point_mask = get_ref_point_mask(self.domain)

    def _get_acqfs(self, n: PositiveInt) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."
        assert self.experiments is not None, "No experiments available."

        X_train, X_pending = self.get_acqf_input_tensors()

        objective = self._get_objective()
        assert self.model is not None

        features2idx, _ = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs
        )

        cost_function, target_fidelities = _map_cost_model_to_botorch(
            self.domain.inputs, features2idx
        )

        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_function)

        current_value = self.get_current_value(target_fidelities)

        acqf = get_acquisition_function_qMFHVKG(
            self.model,
            ref_point=self.get_adjusted_refpoint(),
            current_value=current_value,
            objective=objective,
            target_fidelities=target_fidelities,
            X_observed=X_train,
            X_pending=X_pending,
            mc_samples=self.acquisition_function.n_mc_samples,
            num_fantasies=self.acquisition_function.num_fantasies,
            num_pareto=self.acquisition_function.num_pareto,
            cost_aware_utility=cost_aware_utility,
        )
        return [acqf]

    def _get_objective(self) -> GenericMCMultiOutputObjective:
        assert self.experiments is not None
        objective = get_multiobjective_objective(
            outputs=self.domain.outputs,
            experiments=self.experiments,
        )
        return GenericMCMultiOutputObjective(objective=objective)

    def get_adjusted_refpoint(self) -> List[float]:
        assert self.experiments is not None, "No experiments available."
        assert (
            isinstance(self.ref_point, ExplicitReferencePoint) or self.ref_point is None
        )
        df = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments,
        )
        ref_point = infer_ref_point(
            self.domain,
            experiments=df,
            return_masked=False,
            reference_point=self.ref_point,
        )

        return (
            self.ref_point_mask
            * np.array(
                [
                    ref_point[feat]
                    for feat in self.domain.outputs.get_keys_by_objective(
                        excludes=ConstrainedObjective,
                    )
                ],
            )
        ).tolist()

    def get_current_value(self, target_fidelities: dict[int, float]):
        """Compute the hypervolume of the current HV maximizing set."""
        assert self.model is not None
        curr_val_acqf = _get_hv_value_function(
            model=self.model,
            ref_point=torch.as_tensor(self.get_adjusted_refpoint(), **tkwargs),
            use_posterior_mean=True,
        )

        # fix the task feature
        # we only want the HV at the highest fidelity
        features2idx, _ = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs
        )
        prev_bounds: dict[str, tuple[float, float]] = {}

        for key in self.task_feature_keys:
            task_feature = self.inputs.get_by_key(key)
            assert isinstance(task_feature, ContinuousTaskInput)
            prev_bounds[key] = (task_feature.lower_bound, task_feature.upper_bound)
            tgt = target_fidelities[features2idx[key][0]]
            task_feature.bounds = (tgt, tgt)

        candidates = self.acqf_optimizer.optimize(
            candidate_count=self.acquisition_function.num_pareto,
            acqfs=[curr_val_acqf],
            domain=self.domain,
            experiments=self.experiments,
        )

        # calculate the acquisition function value
        transformed = self.domain.inputs.transform(
            candidates,
            self.input_preprocessing_specs,
        )
        X = torch.from_numpy(transformed.values).to(**tkwargs)

        with torch.no_grad():
            val = curr_val_acqf(X).cpu().detach()

        for key in self.task_feature_keys:
            task_feature = self.inputs.get_by_key(key)
            assert isinstance(task_feature, ContinuousTaskInput)
            task_feature.bounds = prev_bounds[key]

        return val

    @classmethod
    def make(
        cls,
        domain: Domain,
        ref_point: ExplicitReferencePoint | dict[str, float] | None = None,
        cost_aware_utility: CostAwareUtility | None = None,
        acquisition_function: qMFHVKG | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
        include_infeasible_exps_in_acqf_calc: bool | None = False,
    ) -> Self:
        return make_strategy(cls, DataModel, locals())


def get_project(target_fidelities: dict[int, float]) -> Callable[[Tensor], Tensor]:
    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    return project


def get_acquisition_function_qMFHVKG(
    model: GPyTorchModel,
    objective: MCMultiOutputObjective,
    X_observed: Tensor,
    target_fidelities: dict[int, float],
    current_value: Tensor,
    X_pending: Tensor | None = None,
    mc_samples: int = 512,
    seed: int | None = None,
    *,
    # parameters that are only needed for qMFHVKG
    num_fantasies: int = 8,
    num_pareto: int = 10,
    ref_point: None | list[float] | Tensor = None,
    cost_aware_utility: CostAwareUtility | None = None,
) -> qMultiFidelityHypervolumeKnowledgeGradient:
    if ref_point is None:
        raise ValueError("`ref_point` must be a Tensor for qMFHVKG")

    seed_sequence = np.random.SeedSequence(seed)
    seeds = seed_sequence.generate_state(1 + model.num_outputs)
    inner_sampler = get_sampler(
        posterior=model.posterior(X_observed[:1]),  # type: ignore
        sample_shape=torch.Size([mc_samples]),
        seed=int(seeds[0]),
    )

    sampler = ListSampler(
        *[
            get_sampler(
                posterior=model.posterior(X_observed[:1]),
                sample_shape=torch.Size([num_fantasies]),
                seed=int(seed),
            )
            for seed in seeds[1:]
        ]
    )

    return qMultiFidelityHypervolumeKnowledgeGradient(
        model=model,
        ref_point=torch.as_tensor(
            ref_point, device=X_observed.device, dtype=X_observed.dtype
        ),
        target_fidelities=target_fidelities,
        num_fantasies=num_fantasies,
        num_pareto=num_pareto,
        sampler=sampler,
        objective=objective,
        inner_sampler=inner_sampler,
        X_pending=X_pending,
        X_evaluation_mask=None,
        X_pending_evaluation_mask=None,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=get_project(target_fidelities),
    )
