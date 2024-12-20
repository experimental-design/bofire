from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_list,
    optimize_acqf_mixed,
)
from torch import Tensor

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.predictives import acqf_optimizers as data_models
from bofire.data_models.strategies.predictives.acqf_optimizers import (
    AcquisitionFunctionOptimizer as BaseDataModel,
)
from bofire.data_models.strategies.predictives.acqf_optimizers import (
    BotorchAcqfOptimizer as BotorchDataModel,
)
from bofire.utils.torch_tools import get_interpoint_constraints, get_linear_constraints


class AcquisitionFunctionOptimizer(ABC):
    """Base class for all acquisition function optimizers.

    Attributes:
    """

    def __init__(
        self,
        data_model: BaseDataModel,
    ):
        pass

    @classmethod
    def from_spec(cls, data_model: BaseDataModel) -> "AcquisitionFunctionOptimizer":
        """Used by the mapper to map from data model to functional strategy."""
        return cls(data_model=data_model)

    @abstractmethod
    def optimize(
        self,
        domain: Domain,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        bounds: Tensor,
        ic_generator: Callable,
        ic_gen_kwargs: Dict,
        nonlinear_constraints: List[Callable[[Tensor], float]],
        fixed_features: Optional[Dict[int, float]],
        fixed_features_list: Optional[List[Dict[int, float]]],
    ) -> Tuple[Tensor, Tensor]:
        """Abstract method to optimize the acquisition function.

        Args:
            domain (bofire.data_models.domain.api.domain): Search domain.
            candidate_count (int): Number of candidates to produce.
            acqfs (List[AcquisitionFunction]): List of acquisition functions. If there is more than
                one element, the acquisition functions are optimized in sequence with previous candidates
                set as pending. This is also known as sequential greedy optimization.
            bounds (Tensor): A `2 x d` tensor of lower and upper bounds for each factor. If suitable
                constraints are provided, these bounds can be `-inf` and `+inf`, respectively.
            ic_generator (Callable): Function for generating initial conditions. Must be specified
                for nonlinear inequality constraints.
            ic_gen_kwargs: Additional keyword arguments passed to function specified by
                `ic_generator`.
            nonlinear_constraints (List[Callable[[Tensor], float]]): A list of tuples representing
                the nonlinear inequality constraints. The first element in the tuple is a callable
                representing a constraint of the form `callable(x) >= 0`. In case of an
                intra-point constraint, `callable()` takes in a one-dimensional tensor of
                shape `d` and returns a scalar. In case of an inter-point constraint,
                `callable()` takes a two dimensional tensor of shape `candidate_count x d` and again
                returns a scalar. The second element is a boolean, indicating if it is an
                intra-point (`True`) or inter-point (`False`) constraint.
            fixed_features (Optional[Dict[int, float]]): A map `{feature_index: value}` for features
                that should be fixed to a particular value during generation.
            fixed_features_list (Optional[List[Dict[int, float]]]): A list of maps `{feature_index: value}`.
                The i-th item represents the fixed_feature for the i-th optimization. If
                `fixed_features_list` is provided, `optimize_acqf_mixed` is invoked.
        Returns:
            Tuple[Tensor, Tensor]: A two-element tuple containing
                - a `candidate_count x d`-dim tensor of generated candidates.
                - a `candidate_count`-dim tensor of associated acquisition values.
        """
        pass


class BotorchAcqfOptimizer(AcquisitionFunctionOptimizer):
    def __init__(
        self,
        data_model: BotorchDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model)

        self.num_restarts = data_model.num_restarts
        self.num_raw_samples = data_model.num_raw_samples
        self.maxiter = data_model.maxiter
        self.batch_limit = data_model.batch_limit

    def _get_optimizer_options(self, domain: Domain) -> Dict[str, int]:
        """Returns a dictionary of settings passed to `optimize_acqf` controlling
        the behavior of the optimizer.

        Args:
            domain (bofire.data_models.domain.api.Domain): Search domain.

        Returns:
            Dict[str, int]: The dictionary with the settings.
        """
        return {
            "batch_limit": (  # type: ignore
                self.batch_limit
                if len(domain.constraints.get([NChooseKConstraint, ProductConstraint]))
                == 0
                else 1
            ),
            "maxiter": self.maxiter,
        }

    def optimize(
        self,
        domain: Domain,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        bounds: Tensor,
        ic_generator: Callable,
        ic_gen_kwargs: Dict,
        nonlinear_constraints: List[Callable[[Tensor], float]],
        fixed_features: Optional[Dict[int, float]],
        fixed_features_list: Optional[List[Dict[int, float]]],
    ) -> Tuple[Tensor, Tensor]:
        """Uses Botorch (and subsequently scipy) to optimize the acquisition function.

        Args:
            domain (bofire.data_models.domain.api.domain): Search domain.
            candidate_count (int): Number of candidates to produce.
            acqfs (List[AcquisitionFunction]): List of acquisition functions. If there is more than
                one element, the acquisition functions are optimized in sequence with previous candidates
                set as pending. This is also known as sequential greedy optimization.
            bounds (Tensor): A `2 x d` tensor of lower and upper bounds for each factor. If suitable
                constraints are provided, these bounds can be `-inf` and `+inf`, respectively.
            ic_generator (Callable): Function for generating initial conditions. Must be specified
                for nonlinear inequality constraints.
            ic_gen_kwargs: Additional keyword arguments passed to function specified by
                `ic_generator`.
            nonlinear_constraints (List[Callable[[Tensor], float]]): A list of tuples representing
                the nonlinear inequality constraints. The first element in the tuple is a callable
                representing a constraint of the form `callable(x) >= 0`. In case of an
                intra-point constraint, `callable()` takes in a one-dimensional tensor of
                shape `d` and returns a scalar. In case of an inter-point constraint,
                `callable()` takes a two dimensional tensor of shape `candidate_count x d` and again
                returns a scalar. The second element is a boolean, indicating if it is an
                intra-point (`True`) or inter-point (`False`) constraint.
            fixed_features (Optional[Dict[int, float]]): A map `{feature_index: value}` for features
                that should be fixed to a particular value during generation.
            fixed_features_list (Optional[List[Dict[int, float]]]): A list of maps `{feature_index: value}`.
                The i-th item represents the fixed_feature for the i-th optimization. If
                `fixed_features_list` is provided, `optimize_acqf_mixed` is invoked.
        Returns:
            Tuple[Tensor, Tensor]: A two-element tuple containing
                - a `candidate_count x d`-dim tensor of generated candidates.
                - a `candidate_count`-dim tensor of associated acquisition values.
        """
        if len(acqfs) > 1:
            candidates, acqf_vals = optimize_acqf_list(
                acq_function_list=acqfs,
                bounds=bounds,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearEqualityConstraint,
                ),
                inequality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearInequalityConstraint,
                ),
                nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                fixed_features=fixed_features,
                fixed_features_list=fixed_features_list,
                ic_gen_kwargs=ic_gen_kwargs,
                ic_generator=ic_generator,
                options=self._get_optimizer_options(domain),  # type: ignore
            )
        else:
            if fixed_features_list:
                candidates, acqf_vals = optimize_acqf_mixed(
                    acq_function=acqfs[0],
                    bounds=bounds,
                    q=candidate_count,
                    num_restarts=self.num_restarts,
                    raw_samples=self.num_raw_samples,
                    equality_constraints=get_linear_constraints(
                        domain=domain,
                        constraint=LinearEqualityConstraint,
                    ),
                    inequality_constraints=get_linear_constraints(
                        domain=domain,
                        constraint=LinearInequalityConstraint,
                    ),
                    nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                    fixed_features_list=fixed_features_list,
                    ic_generator=ic_generator,
                    ic_gen_kwargs=ic_gen_kwargs,
                    options=self._get_optimizer_options(domain),  # type: ignore
                )
            else:
                interpoints = get_interpoint_constraints(
                    domain=domain, n_candidates=candidate_count
                )
                candidates, acqf_vals = optimize_acqf(
                    acq_function=acqfs[0],
                    bounds=bounds,
                    q=candidate_count,
                    num_restarts=self.num_restarts,
                    raw_samples=self.num_raw_samples,
                    equality_constraints=get_linear_constraints(
                        domain=domain,
                        constraint=LinearEqualityConstraint,
                    )
                    + interpoints,
                    inequality_constraints=get_linear_constraints(
                        domain=domain,
                        constraint=LinearInequalityConstraint,
                    ),
                    fixed_features=fixed_features,
                    nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                    return_best_only=True,
                    options=self._get_optimizer_options(domain),  # type: ignore
                    ic_generator=ic_generator,
                    **ic_gen_kwargs,
                )
        return candidates, acqf_vals


ACQFOPT_MAP: Dict[
    Type[data_models.AcquisitionFunctionOptimizer], Type[AcquisitionFunctionOptimizer]
] = {
    data_models.BotorchAcqfOptimizer: BotorchAcqfOptimizer,
}


def map(
    data_model: data_models.AcquisitionFunctionOptimizer,
) -> AcquisitionFunctionOptimizer:
    cls = ACQFOPT_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
