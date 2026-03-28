import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

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
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
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
from bofire.data_models.strategies.api import (
    ShortestPathStrategy as ShortestPathStrategyDataModel,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies import utils
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.utils.torch_tools import (
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
                    **optimizer_input_i.model_dump()
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
        return {
            "batch_limit": (
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

    def _get_nonlinear_constraint_setup(
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

        nonlinear_constraints = get_nonlinear_constraints(
            domain,
            equality_tolerance=1e-3,
        )
        # Track if there are any true nonlinear equality constraints on the domain.
        has_nonlinear_equality = (
            len(
                domain.constraints.get(NonlinearEqualityConstraint),
            )
            > 0
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

        # NChooseK/Product only: use BoTorch's default IC generator and a generator for reproducibility.
        has_nchoosek_or_product = (
            len(domain.constraints.get([NChooseKConstraint, ProductConstraint])) > 0
        )
        has_nonlinear_inequality = (
            len(domain.constraints.get(NonlinearInequalityConstraint)) > 0
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
        _has_nonlinear_equality = has_nonlinear_equality

        def project_onto_constraints(
            X,
            constraints,
            bounds,
            max_iter: int = 100,
            lr: float = 0.01,
            tol: float = 1e-4,
        ):
            """Project candidates onto constraint manifold using gradient descent."""
            X_proj = X.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([X_proj], lr=lr)

            for _ in range(max_iter):
                optimizer.zero_grad()

                # Compute constraint violations
                total_violation = torch.tensor(0.0, dtype=X.dtype, device=X.device)

                for constraint_fn, _ in constraints:
                    vals = constraint_fn(X_proj)
                    violation = torch.clamp(-vals, min=0.0)
                    total_violation = total_violation + (violation**2).sum()

                if total_violation.item() < tol:
                    break

                total_violation.backward()
                optimizer.step()

                # Project back to box defined by `bounds`
                with torch.no_grad():
                    X_proj.clamp_(bounds[0], bounds[1])

            return X_proj.detach()

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
            """Generate initial conditions respecting nonlinear constraints where possible."""
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

            # Equality-style handling if there are explicit nonlinear equalities
            if _has_nonlinear_equality:
                n_candidates = n_needed * 50
                X_raw_unit = torch.rand(
                    n_candidates,
                    n_dims,
                    device=device,
                    dtype=dtype,
                )
                X_raw = lower + (upper - lower) * X_raw_unit

                X_projected = project_onto_constraints(
                    X_raw,
                    nonlinear_constraints_local,
                    bounds=bounds,
                    max_iter=200,
                    lr=0.05,
                    tol=1e-5,
                )

                feasible_mask = torch.ones(
                    len(X_projected), dtype=torch.bool, device=device
                )
                for constraint_fn, _ in nonlinear_constraints_local:
                    constraint_vals = constraint_fn(X_projected)
                    # BoTorch validates ICs with a (fairly strict) atol; use a
                    # tight feasibility tolerance to avoid passing slightly
                    # infeasible ICs, especially for very small feasible regions.
                    feasible_mask &= constraint_vals >= -1e-6

                X_feasible = X_projected[feasible_mask]

                if len(X_feasible) < n_needed:
                    raise ValueError(
                        f"Projection failed: only {len(X_feasible)} / {n_needed} candidates "
                        f"are feasible after projection. The equality constraint may be "
                        f"incompatible with variable bounds."
                    )

                violations = []
                for constraint_fn, _ in nonlinear_constraints_local:
                    vals = constraint_fn(X_feasible)
                    violations.append(torch.clamp(-vals, min=0.0).abs())

                total_violations = sum(violations)
                _, best_indices = torch.topk(
                    -total_violations,
                    k=min(n_needed, len(X_feasible)),
                    largest=True,
                )
                X_selected = X_feasible[best_indices]
            else:
                # Inequality-only: try to sample feasible ICs, but fall back gracefully
                max_attempts = 20
                raw_samples_per_attempt = max(512, n_needed * 10)

                all_feasible: list[Tensor] = []
                for _ in range(max_attempts):
                    X_raw_unit = torch.rand(
                        raw_samples_per_attempt,
                        n_dims,
                        device=device,
                        dtype=dtype,
                    )
                    X_raw = lower + (upper - lower) * X_raw_unit

                    feasible_mask = torch.ones(
                        len(X_raw), dtype=torch.bool, device=device
                    )
                    for constraint_fn, _ in nonlinear_constraints_local:
                        constraint_vals = constraint_fn(X_raw)
                        # BoTorch's feasible check is strict; allow only tiny
                        # numerical slack so `batch_initial_conditions` passes
                        # validation even for tight constraints.
                        feasible_mask &= constraint_vals >= -1e-6

                    X_feasible = X_raw[feasible_mask]
                    if len(X_feasible) > 0:
                        all_feasible.append(X_feasible)

                    total_feasible = sum(len(x) for x in all_feasible)
                    if total_feasible >= n_needed:
                        break

                if len(all_feasible) == 0:
                    # For tight feasible regions, returning unconstrained ICs will
                    # cause BoTorch to error (`batch_initial_conditions` infeasible).
                    # Signal failure so the outer optimizer can retry with a larger budget.
                    raise ValueError(
                        "Could not generate any feasible initial conditions from nonlinear constraints."
                    )

                X_all = torch.cat(all_feasible)
                if len(X_all) < n_needed:
                    raise ValueError(
                        f"Could not generate enough feasible initial conditions "
                        f"({len(X_all)} found, need {n_needed}) from nonlinear constraints."
                    )

                X_selected = X_all[:n_needed]

            return X_selected.reshape(num_restarts, q, n_dims)

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

        # if len(nonlinear_constraints) == 0:
        #     ic_generator = None
        #     ic_gen_kwargs = {}
        # else:
        #     def feasible_ic_generator(
        #         acq_function,
        #         bounds,
        #         num_restarts,      # ✅ FIXED: Match BoTorch's parameter name
        #         raw_samples,
        #         q=1,
        #         fixed_features=None,
        #         options=None,
        #         inequality_constraints=None,
        #         equality_constraints=None,
        #         **kwargs
        #     ):
        #         """
        #         Generate initial conditions validated in BoTorch tensor space.

        #         This ensures candidates that pass validation here will also pass
        #         BoTorch's constraint check (avoiding round-trip conversion errors).
        #         """
        #         device = bounds.device
        #         dtype = bounds.dtype
        #         n_dims = bounds.shape[-1]

        #         # Oversample to ensure enough feasible candidates
        #         max_attempts = 20
        #         raw_samples_per_attempt = max(512, num_restarts * q * 10)

        #         all_feasible = []

        #         for attempt in range(max_attempts):
        #             # Generate random candidates in [0, 1]^d
        #             X_raw = torch.rand(
        #                 raw_samples_per_attempt,
        #                 n_dims,
        #                 device=device,
        #                 dtype=dtype
        #             )

        #             # Validate against ALL nonlinear constraints
        #             feasible_mask = torch.ones(len(X_raw), dtype=torch.bool, device=device)

        #             for constraint_callable, is_equality in nonlinear_constraints:
        #                 try:
        #                     # Evaluate constraint (BoTorch expects c(x) >= 0 for feasible)
        #                     constraint_vals = constraint_callable(X_raw)

        #                     # Use slightly relaxed tolerance to account for numerical noise
        #                     # during subsequent optimization
        #                     feasible_mask &= (constraint_vals >= -1e-4)

        #                 except Exception as e:
        #                     # If constraint evaluation fails, mark all as infeasible
        #                     print(f"Warning: Constraint evaluation failed: {e}")
        #                     feasible_mask[:] = False
        #                     break

        #             # Collect feasible candidates
        #             X_feasible = X_raw[feasible_mask]

        #             if len(X_feasible) > 0:
        #                 all_feasible.append(X_feasible)

        #             # Check if we have enough
        #             total_feasible = sum(len(x) for x in all_feasible)
        #             if total_feasible >= num_restarts * q:
        #                 break

        #         # Combine all feasible candidates
        #         if len(all_feasible) == 0:
        #             raise ValueError(
        #                 f"Could not generate any feasible initial conditions after "
        #                 f"{max_attempts} attempts with {max_attempts * raw_samples_per_attempt} "
        #                 f"total samples. The feasible region may be very small or empty. "
        #                 f"Consider:\n"
        #                 f"  1. Relaxing constraint tolerances\n"
        #                 f"  2. Expanding variable bounds\n"
        #                 f"  3. Checking if feasible region is too small"
        #             )

        #         X_all = torch.cat(all_feasible)

        #         if len(X_all) < num_restarts * q:
        #             raise ValueError(
        #                 f"Could not generate enough feasible initial conditions. "
        #                 f"Found {len(X_all)} feasible candidates but need {num_restarts * q}. "
        #                 f"Consider:\n"
        #                 f"  1. Reducing num_restarts (currently {num_restarts})\n"
        #                 f"  2. Relaxing constraint tolerances\n"
        #                 f"  3. Checking if feasible region is too small"
        #             )

        #         # Return exactly num_restarts * q candidates, reshaped to [num_restarts, q, d]
        #         X_selected = X_all[:num_restarts * q]
        #         return X_selected.reshape(num_restarts, q, n_dims)

        #     ic_generator = feasible_ic_generator
        #     ic_gen_kwargs = {}  # No additional kwargs needed

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
