import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_list,
    optimize_acqf_mixed,
)
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating
from pydantic import BaseModel, model_validator
from torch import Tensor

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
    Input,
)
from bofire.data_models.strategies.api import (
    AcquisitionOptimizer as AcquisitionOptimizerDataModel,
)
from bofire.data_models.strategies.api import (
    BotorchOptimizer as BotorchOptimizerDataModel,
)
from bofire.data_models.strategies.api import (
    GeneticAlgorithmOptimizer as GeneticAlgorithmDataModel,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import (
    ShortestPathStrategy as ShortestPathStrategyDataModel,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies import utils
from bofire.strategies.random import RandomStrategy
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.utils.torch_tools import (
    get_initial_conditions_generator,
    get_interpoint_constraints,
    get_linear_constraints,
    get_nonlinear_constraints,
    tkwargs,
)


class OptimizerEnum(str, Enum):
    OPTIMIZE_ACQF_LIST = "OPTIMIZE_ACQF_LIST"
    OPTIMIZE_ACQF = "OPTIMIZE_ACQF"
    OPTIMIZE_ACQF_MIXED = "OPTIMIZE_ACQF_MIXED"
    OPTIMIZE_ACQF_MIXED_ALTERNATING = "OPTIMIZE_ACQF_MIXED_ALTERNATING"


# Threshold for switching between optimizers optimize_acqf_mixed
# and optimize_acqf_mixed_alternating. Threshold copied from Ax.
ALTERNATING_OPTIMIZER_THRESHOLD = 10


class AcquisitionOptimizer(ABC):
    def __init__(self, data_model: AcquisitionOptimizerDataModel):
        self.prefer_exhaustive_search_for_purely_categorical_domains = (
            data_model.prefer_exhaustive_search_for_purely_categorical_domains
        )

    def optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,
        experiments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Optimizes the acquisition function(s) for the given domain and input preprocessing specs.

        Args:
            candidate_count: Number of candidates that should be returned.
            acqfs: List of acquisition functions that should be optimized.
            domain: The domain of the optimization problem.
            experiments: The experiments that have been conducted so far.

        Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.

        """
        # we check here if we have a fully combinatorial search space
        # and use _optimize_acqf_discrete in this case
        if self.prefer_exhaustive_search_for_purely_categorical_domains:
            if len(
                domain.inputs.get(includes=[DiscreteInput, CategoricalInput]),
            ) == len(domain.inputs):
                if len(acqfs) > 1:
                    raise NotImplementedError(
                        "Multiple Acqfs are currently not supported for purely combinatorial search spaces.",
                    )
                return self._optimize_acqf_discrete(
                    candidate_count=candidate_count,
                    acqf=acqfs[0],
                    domain=domain,
                    experiments=experiments,  # type: ignore
                )

        return self._optimize(
            candidate_count=candidate_count,
            acqfs=acqfs,
            domain=domain,
            experiments=experiments,
        )

    @abstractmethod
    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,
        experiments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Optimizes the acquisition function(s) for the given domain and input preprocessing specs.

        Args:
            candidate_count (int): Number of candidates that should be returned.
            acqfs (List[AcquisitionFunction]): List of acquisition functions that should be optimized.
            domain (Domain): The domain of the optimization problem.

        Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.

        """
        pass

    @staticmethod
    def _features2idx(
        domain: Domain,
    ) -> Dict[str, Tuple[int]]:
        features2idx, _ = domain.inputs._get_transform_info(
            {},
        )
        return features2idx

    @staticmethod
    def _input_preprocessing_specs(
        domain: Domain,
    ) -> InputTransformSpecs:
        return {
            key: CategoricalEncodingEnum.ORDINAL
            for key in domain.inputs.get_keys(CategoricalInput)
        }

    @staticmethod
    def _candidates_tensor_to_dataframe(
        candidates: Tensor,
        domain: Domain,
    ) -> pd.DataFrame:
        """Converts a tensor of candidates to a pandas Dataframe.

        Args:
            candidates (Tensor): Tensor of candidates returned from `optimize_acqf`.

        Returns:
            pd.DataFrame: Dataframe of candidates.
        """
        # This method is needed here as we use a botorch method to optimize over
        # purely categorical spaces

        df_candidates = pd.DataFrame(
            data=candidates.detach().numpy(),
            columns=domain.inputs.get_keys(),
        )

        df_candidates = domain.inputs.inverse_transform(
            df_candidates,
            AcquisitionOptimizer._input_preprocessing_specs(domain),
        )
        return df_candidates

    @staticmethod
    def get_fixed_features(
        domain: Domain,
    ) -> Dict[int, float]:
        """Provides the values of all fixed features

        Returns:
            fixed_features (dict): Dictionary of fixed features, keys are the feature indices, values the transformed feature values

        """

        fixed_features = {}
        features2idx = AcquisitionOptimizer._features2idx(domain)
        input_preprocessing_specs = AcquisitionOptimizer._input_preprocessing_specs(
            domain
        )

        for _, feat in enumerate(domain.inputs.get(Input)):
            assert isinstance(feat, Input)
            if feat.fixed_value() is not None:
                fixed_values = feat.fixed_value(
                    transform_type=input_preprocessing_specs.get(feat.key),  # type: ignore
                )
                for j, idx in enumerate(features2idx[feat.key]):
                    fixed_features[idx] = fixed_values[j]  # type: ignore

        return fixed_features

    @staticmethod
    def _optimize_acqf_discrete(
        candidate_count: int,
        acqf: AcquisitionFunction,
        domain: Domain,
        experiments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Optimizes the acquisition function for a discrete search space.

        Args:
            candidate_count: Number of candidates that should be returned.
            acqf: Acquisition function that should be optimized.

        Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
        """
        choices = pd.DataFrame.from_dict(
            [  # type: ignore
                {e[0]: e[1] for e in combi}
                for combi in domain.inputs.get_categorical_combinations()
            ],
        )
        # adding categorical features that are fixed
        for feat in domain.inputs.get_fixed():
            choices[feat.key] = feat.fixed_value()[0]  # type: ignore
        # compare the choices with the training data and remove all that are also part
        # of the training data
        merged = choices.merge(
            experiments[domain.inputs.get_keys()],
            on=list(choices.columns),
            how="left",
            indicator=True,
        )
        filtered_choices = merged[merged["_merge"] == "left_only"].copy()
        filtered_choices.drop(columns=["_merge"], inplace=True)

        # remove here everything that falls under a CategoricalExcludeConstraint
        filtered_choices = filtered_choices[
            domain.constraints.is_fulfilled(filtered_choices)
        ].copy()

        # translate the filtered choice to torch
        t_choices = torch.from_numpy(
            domain.inputs.transform(
                filtered_choices,
                specs=AcquisitionOptimizer._input_preprocessing_specs(domain),
            ).values,
        ).to(**tkwargs)
        candidates, _ = optimize_acqf_discrete(
            acq_function=acqf,
            q=candidate_count,
            unique=True,
            choices=t_choices,
        )
        return AcquisitionOptimizer._candidates_tensor_to_dataframe(
            candidates=candidates,
            domain=domain,
        )


class _OptimizeAcqfInputBase(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"  # Forbid arguments not defined in the schema


class _OptimizeAcqfInput(_OptimizeAcqfInputBase):
    acq_function: Callable
    bounds: Tensor
    q: int
    num_restarts: int
    raw_samples: int
    options: dict[str, bool | float | int | str] | None
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None
    nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None
    fixed_features: dict[int, float] | None
    sequential: bool
    ic_generator: Callable | None
    generator: Any


class _OptimizeAcqfMixedInput(_OptimizeAcqfInputBase):
    acq_function: Callable
    bounds: Tensor
    q: int
    num_restarts: int
    fixed_features_list: List[Dict[int, float]]  # it has to have more than two items
    raw_samples: int
    options: dict[str, bool | float | int | str] | None
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None
    nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None
    ic_generator: Callable | None
    ic_gen_kwargs: Dict


class _OptimizeAcqfListInput(_OptimizeAcqfInputBase):
    acq_function_list: List[Callable]
    bounds: Tensor
    num_restarts: int
    raw_samples: int
    options: dict[str, bool | float | int | str] | None
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None
    nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None
    fixed_features: dict[int, float] | None
    fixed_features_list: List[Dict[int, float]] | None
    ic_generator: Callable | None
    ic_gen_kwargs: Dict

    @model_validator(mode="after")
    def validate_fixed_features(self):
        if self.fixed_features and self.fixed_features_list:
            raise ValueError(
                "Only one of fixed_features and fixed_features_list can be provided.",
            )
        return self


class _OptimizeAcqfMixedAlternatingInput(_OptimizeAcqfInputBase):
    acq_function: Callable
    bounds: Tensor
    discrete_dims: Dict[int, List[float]] | None
    cat_dims: Dict[int, List[int]] | None
    options: dict[str, bool | float | int | str] | None
    q: int
    num_restarts: int
    raw_samples: int
    fixed_features: dict[int, float] | None
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None


class BotorchOptimizer(AcquisitionOptimizer):
    def __init__(self, data_model: BotorchOptimizerDataModel):
        self.n_restarts = data_model.n_restarts
        self.n_raw_samples = data_model.n_raw_samples
        self.maxiter = data_model.maxiter
        self.batch_limit = data_model.batch_limit
        self.sequential = data_model.sequential

        self.local_search_config = data_model.local_search_config

        super().__init__(data_model)

    def _setup(self):
        pass

    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        domain: Domain,
        experiments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        input_preprocessing_specs = self._input_preprocessing_specs(domain)
        bounds = utils.get_torch_bounds_from_domain(domain, input_preprocessing_specs)

        # setup local bounds
        assert experiments is not None
        local_lower, local_upper = domain.inputs.get_bounds(
            specs=input_preprocessing_specs,
            reference_experiment=experiments.iloc[-1],
        )
        local_bounds = torch.tensor([local_lower, local_upper]).to(**tkwargs)

        # do the global opt
        candidates, global_acqf_val = self._optimize_acqf_continuous(
            domain=domain,
            candidate_count=candidate_count,
            acqfs=acqfs,
            bounds=bounds,
        )

        candidates = self._candidates_tensor_to_dataframe(candidates, domain)

        if (
            self.local_search_config is not None
            and has_local_search_region(domain)
            and candidate_count == 1
        ):
            local_candidates, local_acqf_val = self._optimize_acqf_continuous(
                domain=domain,
                candidate_count=candidate_count,
                acqfs=acqfs,
                bounds=local_bounds,
            )
            if self.local_search_config.is_local_step(
                local_acqf_val.item(),
                global_acqf_val.item(),
            ):
                return self._candidates_tensor_to_dataframe(
                    local_candidates,
                    domain,
                )

            assert experiments is not None
            sp = ShortestPathStrategy(
                data_model=ShortestPathStrategyDataModel(
                    domain=domain,
                    start=experiments.iloc[-1].to_dict(),
                    end=candidates.iloc[-1].to_dict(),
                ),
            )

            step = pd.DataFrame(sp.step(sp.start)).T
            return step

        return candidates

    def _optimize_acqf_continuous(
        self,
        domain: Domain,
        bounds: Tensor,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
    ) -> Tuple[Tensor, Tensor]:
        optimizer = self._determine_optimizer(domain=domain, n_acqfs=len(acqfs))
        optimizer_input = self._get_arguments_for_optimizer(
            bounds=bounds,
            optimizer=optimizer,
            acqfs=acqfs,
            domain=domain,
            candidate_count=candidate_count,
        )
        optimizer_mapping = {
            OptimizerEnum.OPTIMIZE_ACQF_LIST: optimize_acqf_list,
            OptimizerEnum.OPTIMIZE_ACQF: optimize_acqf,
            OptimizerEnum.OPTIMIZE_ACQF_MIXED: optimize_acqf_mixed,
            OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING: optimize_acqf_mixed_alternating,
        }
        candidates, acqf_vals = optimizer_mapping[optimizer](
            **optimizer_input.model_dump()
        )  # type: ignore
        return candidates, acqf_vals

    def _get_optimizer_options(self, domain: Domain) -> Dict[str, int]:
        """Returns a dictionary of settings passed to `optimize_acqf` controlling
        the behavior of the optimizer.

        Returns:
            Dict[str, int]: The dictionary with the settings.

        """
        return {
            "batch_limit": (  # type: ignore
                self.batch_limit
                if len(
                    domain.constraints.get([NChooseKConstraint, ProductConstraint]),
                )
                == 0
                else 1
            ),
            "maxiter": self.maxiter,
        }

    def _determine_optimizer(self, domain: Domain, n_acqfs) -> OptimizerEnum:
        if n_acqfs > 1:
            return OptimizerEnum.OPTIMIZE_ACQF_LIST
        n_categorical_combinations = (
            domain.inputs.get_number_of_categorical_combinations()
        )
        if n_categorical_combinations == 1:
            return OptimizerEnum.OPTIMIZE_ACQF
        if (
            n_categorical_combinations <= ALTERNATING_OPTIMIZER_THRESHOLD
            or len(get_nonlinear_constraints(domain)) > 0
        ):
            return OptimizerEnum.OPTIMIZE_ACQF_MIXED
        return OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING

    def _get_arguments_for_optimizer(
        self,
        optimizer: OptimizerEnum,
        acqfs: List[AcquisitionFunction],
        bounds: Tensor,
        candidate_count: int,
        domain: Domain,
    ) -> (
        _OptimizeAcqfInput
        | _OptimizeAcqfMixedInput
        | _OptimizeAcqfListInput
        | _OptimizeAcqfMixedAlternatingInput
    ):
        input_preprocessing_specs = self._input_preprocessing_specs(domain)
        features2idx = self._features2idx(domain)
        inequality_constraints = get_linear_constraints(
            domain, constraint=LinearInequalityConstraint
        )
        equality_constraints = get_linear_constraints(
            domain, constraint=LinearEqualityConstraint
        )
        if len(nonlinear_constraints := get_nonlinear_constraints(domain)) == 0:
            ic_generator = None
            ic_gen_kwargs = {}
        else:
            # TODO: implement LSR-BO also for constraints --> use local bounds
            ic_generator = gen_batch_initial_conditions
            ic_gen_kwargs = {
                "generator": get_initial_conditions_generator(
                    strategy=RandomStrategy(
                        data_model=RandomStrategyDataModel(domain=domain),
                    ),
                    transform_specs=input_preprocessing_specs,
                ),
            }
        nonlinear_constraints = (
            nonlinear_constraints if len(nonlinear_constraints) > 0 else None
        )
        # now do it for optimize_acqf
        if optimizer == OptimizerEnum.OPTIMIZE_ACQF:
            interpoints = get_interpoint_constraints(
                domain=domain,
                n_candidates=candidate_count,
            )
            return _OptimizeAcqfInput(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                options=self._get_optimizer_options(domain),  # type: ignore
                sequential=self.sequential,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints + interpoints,
                nonlinear_inequality_constraints=nonlinear_constraints,
                ic_generator=ic_generator,
                generator=ic_gen_kwargs["generator"]
                if ic_generator is not None
                else None,
                fixed_features=self.get_fixed_features(domain=domain),
            )
        elif optimizer == OptimizerEnum.OPTIMIZE_ACQF_MIXED:
            return _OptimizeAcqfMixedInput(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                options=self._get_optimizer_options(domain),  # type: ignore
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_constraints,
                ic_generator=ic_generator,
                ic_gen_kwargs=ic_gen_kwargs,
                fixed_features_list=self.get_categorical_combinations(domain),
            )
        elif optimizer == OptimizerEnum.OPTIMIZE_ACQF_LIST:
            n_combos = domain.inputs.get_number_of_categorical_combinations()
            return _OptimizeAcqfListInput(
                acq_function_list=acqfs,
                bounds=bounds,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                options=self._get_optimizer_options(domain),  # type: ignore
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_constraints,
                ic_generator=ic_generator,
                ic_gen_kwargs=ic_gen_kwargs,
                fixed_features_list=self.get_categorical_combinations(domain)
                if n_combos > 1
                else None,
                fixed_features=self.get_fixed_features(domain=domain)
                if n_combos == 1
                else None,
            )
        elif optimizer == OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING:
            fixed_keys = domain.inputs.get_fixed().get_keys()
            return _OptimizeAcqfMixedAlternatingInput(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                options={"maxiter_continuous": self.maxiter},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=self.get_fixed_features(domain=domain),
                discrete_dims={
                    features2idx[feat.key][0]: feat.values  # type: ignore
                    for feat in domain.inputs.get(DiscreteInput)
                },
                cat_dims={
                    features2idx[feat.key][0]: feat.to_ordinal_encoding(  # type: ignore
                        pd.Series(feat.get_allowed_categories())  # type: ignore
                    ).tolist()
                    for feat in domain.inputs.get(CategoricalInput)
                    if feat.key not in fixed_keys
                },
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def get_categorical_combinations(
        self,
        domain: Domain,
    ) -> list[dict[int, float]]:
        """Provides all possible combinations of fixed values

        Returns:
            list_of_fixed_features: Each dict contains a combination of fixed values
        """
        fixed_basis = self.get_fixed_features(domain=domain)

        combos = domain.inputs.get_categorical_combinations()
        # now build up the fixed feature list
        if len(combos) == 1:
            return [fixed_basis]
        features2idx = self._features2idx(domain)
        list_of_fixed_features = []

        for combo in combos:
            fixed_features = copy.deepcopy(fixed_basis)

            for pair in combo:
                feat, val = pair
                feature = domain.inputs.get_by_key(feat)
                if isinstance(feature, (ContinuousInput, DiscreteInput)):
                    fixed_features[features2idx[feat][0]] = val  # type: ignore
                if isinstance(feature, CategoricalInput):
                    assert feature.categories is not None
                    fixed_features[features2idx[feat][0]] = feature.categories.index(
                        val  # type: ignore
                    )  # this transforms to ordinal encoding

            list_of_fixed_features.append(fixed_features)
        return list_of_fixed_features


class GeneticAlgorithmOptimizer(AcquisitionOptimizer):
    """
    Genetic Algorithm for acquisition function optimization, using the Pymoo mixed-type algorithm.

    This optimizer uses a population-based approach to optimize acquisition functions. Currently, only
    single-objective optimization is supported. The algorithm evolves a population of
    candidate solutions over multiple generations using genetic operators such as mutation, crossover,
    and selection.

    - `CategoricalInput` variables, which are treated as one-hot-encoded columns by the model and the acquisition functions, are turned into categorical variables for the GA optimization. In the objective function, these categorical variables are transformed to one-hot-encoded tensors. The object `BofireDomainMixedVars` handles this conversion.
    - `CategoricalDescriptorInput` is also transformed in to a categorical pymoo variable, but transformed into the descriptor space
    - `DiscreteInput` will be converted to an pymoo Integer.

    All transformations are handled in the helper class `BofireDomainMixedVars`

    **Constraints**
    The GA cannot handle equality constraints well. Constraints are therefor handled differently:

    - Constraints of the type `LinearEqualityConstraint`, `LinearInequalityConstraint`, and `NChooseKConstraint` are handled in a "repair-function". This repair function is used by the GA to map all individuals from the population $x$ to the feasible space $x'$. In this case, I implemented a repair-function for an arbitrary mixture of linear equality and inequality constraints with a quadratic programming approach:

    $$
    \\min_{x'} \\left( ||x-x' ||_2^2 \right)
    $$

    s.t.

    $$
    A \\cdot x' = b
    $$

    $$
    G \\cdot x' <= h
    $$

    $$
    lb <= x' <= ub
    $$

    The `NChooseKConstraint` is also handled in the reapir function: For each experiment in the population, the smallest factors are set to 0, if the *max_features* constraint is violated, and the upper bound of the largest feactors is set to an offset (defaults to $1e-3$), if the *min_features* constraint is violated.

    The repair functions are handled in the class `LinearProjection`.

    - Other supported constraints are: `ProductInequalityConstraint` and `NonlinearInequalityConstraint`. `ProductInequalityConstraint` are evaluated by the torch-callable, provided by the `get_nonlinear_constraints` function. `NonlinearInequalityConstraint` are evaluated from the experiments data-frame, by the constraints `__call__` method.


    These are handled by the optimizer.

    `NonlinearEqualityConstraints` are not supported.



    """

    def __init__(self, data_model: GeneticAlgorithmDataModel):
        super().__init__(data_model)
        self.data_model = data_model

    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,
        experiments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Main function for optimizing the acquisition function using the genetic algorithm.

        Args:
            candidate_count (int): Number of candidates to generate.
            acqfs (List[AcquisitionFunction]): List of acquisition functions to optimize.
            domain (Domain): The domain of the optimization problem.
            input_preprocessing_specs (InputTransformSpecs): Preprocessing specifications for the inputs.
            experiments (Optional[pd.DataFrame]): Existing experiments, if any.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Optimized candidates and their corresponding objective values.
        """

        # Note: If sequential mode is needed, could be added here, and use the single_shot_optimization function in a loop
        input_preprocessing_specs = AcquisitionOptimizer._input_preprocessing_specs(
            domain
        )
        candidates, _ = self._single_shot_optimization(
            domain, input_preprocessing_specs, acqfs, candidate_count
        )

        return self._candidates_tensor_to_dataframe(
            candidates,
            domain,
        )

    def _single_shot_optimization(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        acqfs: List[AcquisitionFunction],
        q: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Single optimizer call. Either for sequential, or simultaneous optimization of q-experiment proposals

        Args:
            domain (Domain)
            input_preprocessing_specs (InputTransformSpecs): transformation specs, as they are needed for the models in
                the acquisition functions

        Returns
            Tensor: x_opt as (d,) Tensor
            Tensor: f_opt as (n_y,) Tensor
        """
        x_opt, f_opt = utils.run_ga(
            self.data_model,
            domain,
            acqfs,
            q,
            callable_format="torch",
            input_preprocessing_specs=input_preprocessing_specs,
            verbose=self.data_model.verbose,
            optimization_direction="max",
        )

        return x_opt, f_opt  # type: ignore


OPTIMIZER_MAP: Dict[Type[AcquisitionOptimizerDataModel], Type[AcquisitionOptimizer]] = {
    BotorchOptimizerDataModel: BotorchOptimizer,
    GeneticAlgorithmDataModel: GeneticAlgorithmOptimizer,
}


def get_optimizer(data_model: AcquisitionOptimizerDataModel) -> AcquisitionOptimizer:
    return OPTIMIZER_MAP[type(data_model)](data_model)
