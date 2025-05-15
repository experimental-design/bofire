from typing import Dict, List, Optional, cast

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, get_acquisition_function
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.models.gpytorch import GPyTorchModel
from pydantic import PositiveInt
from typing_extensions import Self

from bofire.data_models.acquisition_functions.api import (
    AnyMultiObjectiveAcquisitionFunction,
    qEHVI,
    qLogEHVI,
    qLogNEHVI,
    qNEHVI,
)
from bofire.data_models.domain.domain import Domain
from bofire.data_models.objectives.api import ConstrainedObjective
from bofire.data_models.outlier_detection.outlier_detections import OutlierDetections
from bofire.data_models.strategies.api import ExplicitReferencePoint
from bofire.data_models.strategies.api import MoboStrategy as DataModel
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.strategies.strategy import make_strategy
from bofire.utils.multiobjective import get_ref_point_mask, infer_ref_point
from bofire.utils.torch_tools import (
    get_multiobjective_objective,
    get_output_constraints,
    tkwargs,
)


class MoboStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        assert isinstance(data_model.ref_point, ExplicitReferencePoint)
        self.ref_point: ExplicitReferencePoint = data_model.ref_point
        self.ref_point_mask = get_ref_point_mask(self.domain)
        self.acquisition_function = data_model.acquisition_function

    objective: Optional[MCMultiOutputObjective] = None

    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."
        assert self.experiments is not None, "No experiments available."

        X_train, X_pending = self.get_acqf_input_tensors()

        # get etas and constraints
        constraints, etas = get_output_constraints(
            self.domain.outputs,
            experiments=self.experiments,
        )
        if len(constraints) == 0:
            constraints, etas = None, 1e-3
        else:
            etas = torch.tensor(etas).to(**tkwargs)

        objective = self._get_objective()
        # in case that qehvi, qlogehvi is used we need also y
        if isinstance(self.acquisition_function, (qLogEHVI, qEHVI)):
            Y = torch.from_numpy(
                self.domain.outputs.preprocess_experiments_all_valid_outputs(
                    self.experiments,
                )[self.domain.outputs.get_keys()].values,
            ).to(**tkwargs)
        else:
            Y = None

        assert self.model is not None

        acqf = get_acquisition_function(
            self.acquisition_function.__class__.__name__,
            self.model,
            ref_point=self.get_adjusted_refpoint(),
            objective=objective,
            X_observed=X_train,
            X_pending=X_pending,
            constraints=constraints,
            eta=etas,
            mc_samples=self.acquisition_function.n_mc_samples,
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            alpha=self.acquisition_function.alpha,
            prune_baseline=(
                self.acquisition_function.prune_baseline
                if isinstance(self.acquisition_function, (qLogNEHVI, qNEHVI))
                else True
            ),
            Y=Y,
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

    @classmethod
    def make(
        cls,
        domain: Domain,
        ref_point: ExplicitReferencePoint | Dict[str, float] | None = None,
        acquisition_function: AnyMultiObjectiveAcquisitionFunction | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
    ):
        """
        Creates an instance of a multi-objective strategy based on expected hypervolume improvement.

        S. Daulton, M. Balandat, and E. Bakshy.
        Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement.
        Advances in Neural Information Processing Systems 34, 2021.

        Parameters:
            domain: The domain specifying the search space.
            ref_point: Reference point for hypervolume computation.
            acquisition_function: Acquisition function.
            acquisition_optimizer: Optimizer for the acquisition function.
            surrogate_specs: Surrogate model specifications.
            outlier_detection_specs: Outlier detection configuration.
            min_experiments_before_outlier_check: Minimum number of experiments before performing outlier detection.
            frequency_check: Frequency at which to perform outlier checks.
            frequency_hyperopt: Frequency at which to perform hyperparameter optimization.
            folds: Number of folds for cross-validation for hyperparameter optimization.
            seed: Random seed for reproducibility.
        Returns:
            An instance of the strategy configured with the specified parameters.
        """
        return cast(Self, make_strategy(cls, DataModel, locals()))
