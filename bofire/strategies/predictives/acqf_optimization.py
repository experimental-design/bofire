import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import CandidateGenerationError
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
    Constraint,
    InterpointConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    ProductConstraint,
    ProductInequalityConstraint,
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
from bofire.data_models.strategies.api import (
    ShortestPathStrategy as ShortestPathStrategyDataModel,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies import utils
from bofire.strategies.predictives._nchoosek_pruning import (
    is_nchoosek_pruning_applicable,
    is_pruning_applicable,
    prune_nchoosek,
    semicontinuous_specs_from_domain,
)
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.utils.torch_tools import (
    get_interpoint_constraints,
    get_linear_constraints,
    get_nonlinear_constraints,
    get_torch_bounds_from_domain,
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
                    experiments=experiments,
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
        return dict.fromkeys(
            domain.inputs.get_keys(CategoricalInput), CategoricalEncodingEnum.ORDINAL
        )

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
                    transform_type=input_preprocessing_specs.get(feat.key),
                )
                for j, idx in enumerate(features2idx[feat.key]):
                    fixed_features[idx] = fixed_values[j]

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
            [
                {e[0]: e[1] for e in combi}
                for combi in domain.inputs.get_categorical_combinations()
            ],
        )
        # adding categorical features that are fixed
        for feat in domain.inputs.get_fixed():
            fixed_val = feat.fixed_value()
            assert fixed_val is not None
            choices[feat.key] = fixed_val[0]
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
    generator: Any = None


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

        self.per_step_local_reopt = data_model.per_step_local_reopt
        self.final_local_reopt = data_model.final_local_reopt

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
        pruning_applicable = is_pruning_applicable(domain)

        input_preprocessing_specs = self._input_preprocessing_specs(domain)
        bounds = get_torch_bounds_from_domain(
            domain,
            input_preprocessing_specs,
            relax_allow_zero=pruning_applicable,
        )

        # setup local bounds
        assert experiments is not None
        local_bounds = get_torch_bounds_from_domain(
            domain,
            input_preprocessing_specs,
            reference_experiment=experiments.iloc[-1],
        )

        # do the global opt
        candidates, global_acqf_val = self._optimize_acqf_continuous(
            domain=domain,
            candidate_count=candidate_count,
            acqfs=acqfs,
            bounds=bounds,
        )
        # print(candidates)

        if pruning_applicable:
            candidates = self._prune(
                candidates=candidates,
                acqfs=acqfs,
                domain=domain,
                bounds=bounds,
            )

        # print(candidates)

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

    def _project_onto_nonlinear_constraints(
        self,
        X: Tensor,
        domain: Domain,
        bounds: Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
        tol: float = 1e-4,
    ) -> Tensor:
        """Project candidate tensor onto nonlinear constraint manifold (for fallback)."""
        nonlinear_constraints, _, _ = self._get_nonlinear_constraint_setup(domain)
        if nonlinear_constraints is None or len(nonlinear_constraints) == 0:
            return X
        shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1]).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([X_flat], lr=lr)
        for _ in range(max_iter):
            optimizer.zero_grad()
            total_violation = torch.tensor(0.0, dtype=X.dtype, device=X.device)
            for constraint_fn, _ in nonlinear_constraints:
                vals = constraint_fn(X_flat)
                violation = torch.clamp(-vals, min=0.0)
                total_violation = total_violation + (violation**2).sum()
            if total_violation.item() < tol:
                break
            total_violation.backward()
            optimizer.step()
            with torch.no_grad():
                X_flat.clamp_(bounds[0], bounds[1])
        return X_flat.detach().reshape(shape)

    def _prune(
        self,
        candidates: Tensor,
        acqfs: List[AcquisitionFunction],
        domain: Domain,
        bounds: Tensor,
    ) -> Tensor:
        """Apply BONSAI greedy pruning to the AF-winning candidate tensor.

        Caller must have already verified ``is_pruning_applicable(domain)``.
        Reuses the bounds (relaxed for semi-continuous features) the AF
        maximiser used, and the same linear-constraint and
        fixed-feature accessors.

        Forwards ``acqfs[0]`` to the greedy AF evaluation; multi-AF
        per-candidate pruning is not yet implemented.
        """
        # TODO: per-AF pruning for multi-objective.
        features2idx = self._features2idx(domain)
        inequality_constraints = get_linear_constraints(
            domain, constraint=LinearInequalityConstraint
        )
        equality_constraints = get_linear_constraints(
            domain, constraint=LinearEqualityConstraint
        )
        semicontinuous_specs = semicontinuous_specs_from_domain(domain, features2idx)
        nchoosek_constraints = list(domain.constraints.get(NChooseKConstraint))

        # Pinning policy: freeze every column at its per-row value
        # *except* those pruning genuinely needs to move. Pruning only
        # needs to move continuous, un-fixed features that are either
        # NChooseK / semi-continuous themselves, or participate in a
        # linear constraint that touches an NChooseK / semi-continuous
        # feature (the QP projection may need to redistribute mass
        # across those features after a zero/active/activate commit).
        # Everything else stays frozen: categorical / discrete /
        # molecular encodings (which can't be reasoned about by SLSQP
        # / optimize_acqf), fixed-value features, features in
        # Interpoint / Nonlinear / Product constraints (which the QP
        # cannot enforce), and continuous features that pruning has
        # no business touching at all.
        nchoosek_feat_keys: Set[str] = set()
        for c in nchoosek_constraints:
            nchoosek_feat_keys.update(c.features)
        semi_feat_keys: Set[str] = {
            feat.key
            for feat in domain.inputs.get(ContinuousInput)
            if isinstance(feat, ContinuousInput) and feat.is_semicontinuous
        }
        pruning_core_feat_keys: Set[str] = nchoosek_feat_keys | semi_feat_keys
        # Features dragged in via linear constraints that touch a
        # core feature: the QP may need to move them to redistribute
        # mass when a core feature is zeroed / activated.
        linear_drag_feat_keys: Set[str] = set()
        for c in domain.constraints.get(
            includes=[LinearEqualityConstraint, LinearInequalityConstraint]
        ):
            feat_set = set(c.features)
            if feat_set & pruning_core_feat_keys:
                linear_drag_feat_keys.update(feat_set)
        movable_feat_keys: Set[str] = pruning_core_feat_keys | linear_drag_feat_keys
        # Features in pruning-unhandled constraint types (Interpoint /
        # Nonlinear / Product) must be pinned even when they would
        # otherwise be movable -- those constraints are invisible to
        # the QP and freezing the feature at the per-row value
        # preserves them by inertia.
        unhandled_constraint_feat_keys: Set[str] = set()
        for c in domain.constraints.get(
            includes=[InterpointConstraint, NonlinearConstraint, ProductConstraint]
        ):
            unhandled_constraint_feat_keys.update(c.features)

        pinned_columns: Set[int] = set()
        for feat in domain.inputs:
            cols = features2idx[feat.key]
            is_movable_candidate = (
                isinstance(feat, ContinuousInput)
                and feat.fixed_value() is None
                and feat.key in movable_feat_keys
            )
            if is_movable_candidate and feat.key not in unhandled_constraint_feat_keys:
                continue
            pinned_columns.update(cols)

        return prune_nchoosek(
            X=candidates,
            acqf=acqfs[0],
            nchoosek_constraints=nchoosek_constraints,
            features2idx=features2idx,
            bounds=bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            semicontinuous_specs=semicontinuous_specs,
            pinned_columns=pinned_columns,
            per_step_local_reopt=self.per_step_local_reopt,
            final_local_reopt=self.final_local_reopt,
        )

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
        # Prefer principled retries over post-hoc projection/repair:
        # If we get infeasible candidates back from BoTorch, re-run the optimizer a few
        # times and *escalate* the search (more raw_samples / restarts) for tight
        # feasible regions.
        max_infeasible_retries = 4
        nonlinear_constraints, _, _ = self._get_nonlinear_constraint_setup(domain)

        for attempt in range(max_infeasible_retries + 1):
            # Escalate search budget on retries (no change for attempt==0).
            if attempt == 0:
                optimizer_input_i = optimizer_input
            else:
                factor = 2**attempt
                optimizer_input_i = optimizer_input.model_copy(
                    update={
                        "raw_samples": int(optimizer_input.raw_samples * factor),
                        "num_restarts": int(optimizer_input.num_restarts * factor),
                    }
                )
            try:
                candidates, acqf_vals = optimizer_mapping[optimizer](
                    **optimizer_input_i.model_dump(exclude_none=True)
                )  # type: ignore
            except (CandidateGenerationError, ValueError):
                # IC generation / constraint validation failures can surface as ValueError.
                # Retry with larger budgets rather than disabling constraints.
                if attempt == max_infeasible_retries:
                    raise
                continue

            if nonlinear_constraints is None or len(nonlinear_constraints) == 0:
                return candidates, acqf_vals

            # Check feasibility in tensor space: BoTorch-style constraints are feasible when >= 0.
            X_flat = candidates.reshape(-1, candidates.shape[-1])
            with torch.no_grad():
                feasible = True
                for constraint_fn, _ in nonlinear_constraints:
                    if torch.any(constraint_fn(X_flat) < 0.0):
                        feasible = False
                        break

            if feasible:
                return candidates, acqf_vals

            # Otherwise retry (unless this was the last attempt).
            if attempt == max_infeasible_retries:
                return candidates, acqf_vals
        return candidates, acqf_vals

    def _get_optimizer_options(self, domain: Domain) -> Dict[str, int]:
        """Returns a dictionary of settings passed to `optimize_acqf` controlling
        the behavior of the optimizer.

        Returns:
            Dict[str, int]: The dictionary with the settings.

        """
        assert self.batch_limit is not None
        pruning_applicable = is_nchoosek_pruning_applicable(domain)
        constraint_types = [
            ProductConstraint,
            NonlinearInequalityConstraint,
            NonlinearEqualityConstraint,
        ]
        if not pruning_applicable:
            constraint_types.append(NChooseKConstraint)
        return {
            "batch_limit": (
                self.batch_limit
                if len(domain.constraints.get(constraint_types)) == 0
                else 1
            ),
            "maxiter": self.maxiter,
        }

    def _determine_optimizer(self, domain: Domain, n_acqfs) -> OptimizerEnum:
        if n_acqfs > 1:
            return OptimizerEnum.OPTIMIZE_ACQF_LIST
        # When pruning is applicable, semi-continuous features
        # (`allow_zero=True` with `lb > 0`) are handled by the post-AF
        # pruning step rather than by enumerating their on/off states
        # at AF-optimisation time. Excluding them from the combination
        # count routes a pure-continuous semi-continuous domain to
        # `optimize_acqf` rather than `optimize_acqf_mixed`.
        n_categorical_combinations = (
            domain.inputs.get_number_of_categorical_combinations(
                include_semicontinuous=not is_pruning_applicable(domain),
            )
        )
        if n_categorical_combinations == 1:
            return OptimizerEnum.OPTIMIZE_ACQF
        # NChooseK is handled by post-AF pruning when applicable, so
        # exclude it from the AF-time nonlinear constraint set.
        nonlinear_types: List[Type[Constraint]] = [ProductInequalityConstraint]
        if not is_nchoosek_pruning_applicable(domain):
            nonlinear_types.append(NChooseKConstraint)
        if (
            n_categorical_combinations <= ALTERNATING_OPTIMIZER_THRESHOLD
            or len(get_nonlinear_constraints(domain, includes=nonlinear_types)) > 0
        ):
            return OptimizerEnum.OPTIMIZE_ACQF_MIXED
        return OptimizerEnum.OPTIMIZE_ACQF_MIXED_ALTERNATING

    def _get_nonlinear_constraint_setup(  # noqa: C901
        self,
        domain: Domain,
    ) -> tuple[
        Optional[list[tuple[Callable, bool]]],
        Optional[Callable],
        dict,
    ]:
        """Prepare nonlinear constraint callables and optional IC generator.

        Returns:
            nonlinear_constraints: List of (callable, is_equality) or None
            ic_generator: Optional initial condition generator callable
            ic_gen_kwargs: Extra kwargs for BoTorch optimize API (currently unused)
        """
        import torch

        nonlinear_types: List[Type[Constraint]] = [
            ProductInequalityConstraint,
            NonlinearInequalityConstraint,
            NonlinearEqualityConstraint,
        ]
        if not is_nchoosek_pruning_applicable(domain):
            nonlinear_types.append(NChooseKConstraint)

        nonlinear_constraints = get_nonlinear_constraints(
            domain,
            includes=nonlinear_types,
            equality_tolerance=1e-3,
        )
        # Special-case: if all NonlinearInequalityConstraints use callable expressions
        # (rather than string expressions), skip passing them through to BoTorch as
        # nonlinear_inequality_constraints. In this mode, BoTorch has no robust way
        # to construct feasible initial conditions and will otherwise raise when
        # validating batch_initial_conditions. We instead rely on BoFire's own
        # domain-level validation of candidates.
        from bofire.data_models.constraints.api import (
            NonlinearInequalityConstraint as _NIConstr,
        )

        ni_constraints = domain.constraints.get(_NIConstr)
        has_callable_nonlinear_ineq = any(
            callable(c.expression) for c in ni_constraints
        )

        if len(nonlinear_constraints) == 0 or has_callable_nonlinear_ineq:
            # Do not enforce nonlinear constraints inside BoTorch optimize_acqf;
            # also disable the custom initial-condition generator.
            ic_generator = None
            ic_gen_kwargs = {}
            return None, ic_generator, ic_gen_kwargs

        # NChooseK/Product only: use BoTorch's default IC generator with Sobol samples.
        has_nchoosek_or_product = (
            len(domain.constraints.get([NChooseKConstraint, ProductConstraint])) > 0
        )
        has_nonlinear_inequality = (
            len(domain.constraints.get(NonlinearInequalityConstraint)) > 0
        )
        has_nonlinear_equality = (
            len(domain.constraints.get(NonlinearEqualityConstraint)) > 0
        )
        if (
            has_nchoosek_or_product
            and not has_nonlinear_equality
            and not has_nonlinear_inequality
        ):
            return (
                nonlinear_constraints,
                gen_batch_initial_conditions,
                {"generator": "sobol"},
            )

        _captured_constraints = nonlinear_constraints

        def feasible_ic_generator(
            acq_function,
            bounds,
            num_restarts,
            raw_samples,
            q=1,
            fixed_features=None,
            options=None,
            inequality_constraints=None,
            equality_constraints=None,
            **kwargs,
        ):
            """Generate feasible initial conditions via rejection sampling.

            NonlinearEqualityConstraints are already converted to two inequality
            callables (±tolerance slab) by get_nonlinear_constraints, so a single
            rejection-sampling path handles both equalities and inequalities.
            The per-attempt sample budget is set generously (n_needed * 50) to
            handle the thin slabs produced by tight equality tolerances.
            """
            nonlinear_constraints_local = _captured_constraints

            if len(nonlinear_constraints_local) == 0:
                from botorch.optim.initializers import gen_batch_initial_conditions

                return gen_batch_initial_conditions(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options or {},
                )

            device = bounds.device
            dtype = bounds.dtype
            n_dims = bounds.shape[-1]
            n_needed = num_restarts * q

            lower = bounds[0].to(device=device, dtype=dtype)
            upper = bounds[1].to(device=device, dtype=dtype)

            # Rejection sampling: draw uniform random points and keep those that
            # satisfy all constraints with BoTorch's strict feasibility tolerance
            # (>= -1e-6). The generous per-attempt budget handles both wide
            # inequality regions and the thin slabs from equality constraints.
            max_attempts = 20
            raw_samples_per_attempt = max(512, n_needed * 50)

            all_feasible: list[Tensor] = []
            for _ in range(max_attempts):
                X_raw = lower + (upper - lower) * torch.rand(
                    raw_samples_per_attempt,
                    n_dims,
                    device=device,
                    dtype=dtype,
                )

                feasible_mask = torch.ones(len(X_raw), dtype=torch.bool, device=device)
                for constraint_fn, _ in nonlinear_constraints_local:
                    # BoTorch validates batch_initial_conditions with a strict
                    # >= 0 check; match that threshold exactly so every IC we
                    # return passes BoTorch's own feasibility gate.
                    feasible_mask &= constraint_fn(X_raw) >= 0

                X_feasible = X_raw[feasible_mask]
                if len(X_feasible) > 0:
                    all_feasible.append(X_feasible)

                if sum(len(x) for x in all_feasible) >= n_needed:
                    break

            if len(all_feasible) == 0:
                raise ValueError(
                    "Could not generate any feasible initial conditions from nonlinear constraints."
                )

            X_all = torch.cat(all_feasible)
            if len(X_all) < n_needed:
                # Tile the available feasible points to fill the required number.
                # This can happen for very tight equality-constraint slabs where
                # rejection sampling finds some but not all needed ICs. Duplicate
                # restarts are acceptable; BoTorch optimizes each independently.
                repeats = (n_needed + len(X_all) - 1) // len(X_all)
                X_all = X_all.repeat(repeats, 1)

            return X_all[:n_needed].reshape(num_restarts, q, n_dims)

        ic_generator = feasible_ic_generator
        ic_gen_kwargs = {}
        return nonlinear_constraints, ic_generator, ic_gen_kwargs

    def _get_arguments_for_optimizer(
        self,
        optimizer: OptimizerEnum,
        acqfs: List[AcquisitionFunction],
        bounds: Tensor,
        candidate_count: int,
        domain: Domain,
        skip_nonlinear: bool = False,
    ) -> (
        _OptimizeAcqfInput
        | _OptimizeAcqfMixedInput
        | _OptimizeAcqfListInput
        | _OptimizeAcqfMixedAlternatingInput
    ):
        features2idx = self._features2idx(domain)
        inequality_constraints = get_linear_constraints(
            domain, constraint=LinearInequalityConstraint
        )
        equality_constraints = get_linear_constraints(
            domain, constraint=LinearEqualityConstraint
        )

        nonlinear_constraints, ic_generator, ic_gen_kwargs = (
            self._get_nonlinear_constraint_setup(domain)
        )

        # Some BoTorch versions expect a callable `generator(n, q, seed)` in
        # `gen_batch_initial_conditions`. We use Sobol samples and close over `bounds`
        # so it works consistently across versions.
        if ic_gen_kwargs.get("generator") == "sobol":
            from botorch.utils.sampling import draw_sobol_samples

            def _sobol_generator(n: int, q: int, seed: int | None = None):
                X = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed).to(
                    device=bounds.device, dtype=bounds.dtype
                )
                # If the domain has NChooseK constraints, make the generated initial
                # conditions feasible by zeroing out the smallest components.
                # This is used with BoTorch's `gen_batch_initial_conditions`, which
                # requires `batch_initial_conditions` to satisfy nonlinear constraints.
                nchooseks = domain.constraints.get(NChooseKConstraint)
                if len(nchooseks) == 0:
                    return X

                X_flat = X.reshape(-1, X.shape[-1])
                cont_keys = domain.inputs.get_keys(ContinuousInput)
                for c in nchooseks:
                    # Only enforce the max_count relaxation here (common use case).
                    if c.max_count >= len(c.features):
                        continue
                    feat_indices = torch.tensor(
                        [cont_keys.index(k) for k in c.features],
                        device=X_flat.device,
                        dtype=torch.long,
                    )
                    sub = X_flat.index_select(-1, feat_indices)
                    n_zero = len(c.features) - c.max_count
                    if n_zero <= 0:
                        continue
                    # Set the smallest `n_zero` entries (per row) to zero.
                    _, small_idx = torch.topk(sub, k=n_zero, largest=False, dim=-1)
                    rows = torch.arange(sub.shape[0], device=X_flat.device).unsqueeze(
                        -1
                    )
                    sub = sub.clone()
                    sub[rows, small_idx] = 0.0
                    X_flat[:, feat_indices] = sub

                return X_flat.reshape_as(X)

            ic_gen_kwargs = {**ic_gen_kwargs, "generator": _sobol_generator}

        if skip_nonlinear:
            nonlinear_constraints = None
            ic_generator = None
        else:
            nonlinear_constraints = (
                nonlinear_constraints
                if (
                    nonlinear_constraints is not None and len(nonlinear_constraints) > 0
                )
                else None
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
                options=self._get_optimizer_options(domain),
                sequential=self.sequential,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints + interpoints,
                nonlinear_inequality_constraints=nonlinear_constraints,
                ic_generator=ic_generator,
                generator=ic_gen_kwargs.get("generator") if ic_gen_kwargs else None,
                fixed_features=self.get_fixed_features(domain=domain),
            )
        elif optimizer == OptimizerEnum.OPTIMIZE_ACQF_MIXED:
            return _OptimizeAcqfMixedInput(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                options=self._get_optimizer_options(domain),
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
                options=self._get_optimizer_options(domain),
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
                    features2idx[feat.key][0]: feat.values
                    for feat in domain.inputs.get(DiscreteInput)
                },
                cat_dims={
                    features2idx[feat.key][0]: feat.to_ordinal_encoding(
                        pd.Series(feat.get_allowed_categories())
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
                    fixed_features[features2idx[feat][0]] = (
                        val  # ty: ignore[invalid-assignment]
                    )
                if isinstance(feature, CategoricalInput):
                    assert feature.categories is not None
                    fixed_features[features2idx[feat][0]] = feature.categories.index(
                        val
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

        return x_opt, f_opt  # ty: ignore[invalid-return-type]


OPTIMIZER_MAP: Dict[Type[AcquisitionOptimizerDataModel], Type[AcquisitionOptimizer]] = {
    BotorchOptimizerDataModel: BotorchOptimizer,
    GeneticAlgorithmDataModel: GeneticAlgorithmOptimizer,
}


def get_optimizer(data_model: AcquisitionOptimizerDataModel) -> AcquisitionOptimizer:
    return OPTIMIZER_MAP[type(data_model)](data_model)
