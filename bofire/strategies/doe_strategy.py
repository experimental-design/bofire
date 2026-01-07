from typing import Dict, List, Optional, cast

import pandas as pd
import torch
from pydantic.types import PositiveInt
from typing_extensions import Self

import bofire.data_models.strategies.api as data_models
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput, DiscreteInput, Input
from bofire.data_models.strategies.doe import (
    AnyDoEOptimalityCriterion,
    AnyOptimalityCriterion,
    DoEOptimalityCriterion,
)
from bofire.strategies.doe.design import find_local_max_ipopt, get_n_experiments
from bofire.strategies.doe.objective import ModelBasedObjective, get_objective_function
from bofire.strategies.doe.utils import get_formula_from_string, n_zero_eigvals
from bofire.strategies.doe.utils_categorical_discrete import (
    create_continuous_domain,
    encode_candidates_to_relaxed_domain,
    filter_out_categorical_and_categorical_auxilliary_vars,
    filter_out_discrete_auxilliary_vars,
    project_candidates_into_domain,
)
from bofire.strategies.strategy import Strategy, make_strategy


class DoEStrategy(Strategy):
    """Strategy for design of experiments. This strategy is used to generate a set of
    experiments for a given domain.
    The experiments are generated via minimization of a user defined optimality criterion.

    """

    def __init__(
        self,
        data_model: data_models.DoEStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self._data_model = data_model
        self._partially_fixed_candidates = None
        self._fixed_candidates = None
        self._sampling = (
            pd.DataFrame(self._data_model.sampling)
            if self._data_model.sampling is not None
            else None
        )
        self._return_fixed_candidates = (
            data_model.return_fixed_candidates
        )  # this defaults to False in the data model
        # DoE optimization has larger numerical errors (~1e-4) due to SCIP solver precision
        self._validation_tol = 1e-4

    def set_candidates(self, candidates: pd.DataFrame):
        original_columns = self.domain.inputs.get_keys(includes=Input)
        too_many_columns = []
        for col in candidates.columns:
            if col not in original_columns:
                too_many_columns.append(col)
        if len(too_many_columns) > 0:
            raise AttributeError(
                f"provided candidates have columns: {(*too_many_columns,)},  which do not exist in original domain",
            )

        too_few_columns = []
        for col in original_columns:
            if col not in candidates.columns:
                too_few_columns.append(col)
        if len(too_few_columns) > 0:
            raise AttributeError(
                f"provided candidates are missing columns: {(*too_few_columns,)} which exist in original domain",
            )

        self._candidates = candidates

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:  # type: ignore # due to inheriting from Strategy, we then later call this using self.candidates
        (
            relaxed_domain,
            mappings_categorical_var_key_to_aux_var_key_state_pairs,
            mapping_discrete_input_to_discrete_aux,
            aux_vars_for_discrete,
            mapped_aux_categorical_inputs,
            mapped_continous_inputs,
        ) = create_continuous_domain(domain=self.domain)

        # if you have fixed experiments, so-called _candidates, you need to relaxe them and add them to the total number of experiments
        if self.candidates is not None:  # aka if self._candidates is not None
            # transform candidates to new domain
            relaxed_candidates = self._transform_candidates_to_new_domain(
                relaxed_domain,
                self.candidates,
            )
            fixed_experiments_count = self.candidates.notnull().all(axis=1).sum()
        else:
            relaxed_candidates = None
            fixed_experiments_count = 0

        # total number of experiments that will go into the design
        _total_count = candidate_count + fixed_experiments_count

        objective_function = get_objective_function(
            self._data_model.criterion,
            domain=relaxed_domain,
            n_experiments=_total_count,
            inputs_for_formula=self.domain.inputs,
        )
        assert objective_function is not None, "Criterion type is not supported!"

        design = find_local_max_ipopt(
            relaxed_domain,
            fixed_experiments=None,  # effectively deprecated, but others use it so we have not removed it yet
            partially_fixed_experiments=relaxed_candidates,  # technically fixed experiments are also partially_fixed, so we only use this
            ipopt_options=self._data_model.ipopt_options,
            objective_function=objective_function,
        )

        # if cats or discrete var present, need to filture out all the aux vars and project back into original domain
        if len(self.domain.inputs.get([DiscreteInput, CategoricalInput])) > 0:
            # deal with tthe categoricals first
            design_no_categoricals, design_categoricals = (
                filter_out_categorical_and_categorical_auxilliary_vars(
                    design,
                    mappings_categorical_var_key_to_aux_var_key_state_pairs=mappings_categorical_var_key_to_aux_var_key_state_pairs,
                    mapped_aux_categorical_inputs=mapped_aux_categorical_inputs,
                )
            )
            # if no discrete in domain
            if len(self.domain.inputs.get([DiscreteInput])) == 0:
                return (
                    design_categoricals.join(design_no_categoricals)
                    .iloc[fixed_experiments_count:, :]
                    .reset_index(
                        drop=True,
                    )
                )
            else:
                design_projected = project_candidates_into_domain(
                    domain=self.domain,
                    candidates=design_no_categoricals,
                    mapping_discrete_input_to_discrete_aux=mapping_discrete_input_to_discrete_aux,
                    keys_continuous_inputs=[
                        continuous_input.key
                        for continuous_input in mapped_continous_inputs
                    ],
                    scip_params=self._data_model.scip_params,
                )
                design = filter_out_discrete_auxilliary_vars(
                    design_projected,
                    aux_vars_for_discrete=aux_vars_for_discrete,
                )
                design = pd.concat([design, design_categoricals], axis=1)
        if self._return_fixed_candidates:  # this is asking if the fixed candidates should be returned together with the new ones, or just the new ones. Default just the new ones.
            fixed_experiments_count = 0
        return design.iloc[fixed_experiments_count:, :].reset_index(
            drop=True,
        )

    def get_required_number_of_experiments(self) -> Optional[int]:
        if isinstance(self._data_model.criterion, DoEOptimalityCriterion):
            if self.domain.inputs.get([DiscreteInput, CategoricalInput]):
                _domain, *_ = create_continuous_domain(domain=self.domain)
            else:
                _domain = self.domain
            formula = get_formula_from_string(
                self._data_model.criterion.formula, inputs=self.domain.inputs
            )
            return get_n_experiments(formula) - n_zero_eigvals(
                domain=_domain, model_type=formula
            )
        else:
            ValueError(
                f"Only {AnyDoEOptimalityCriterion} type have required number of experiments."
            )

    def get_candidate_rank(self) -> int:
        """Get the rank of the model matrix with the current candidates."""
        if self.candidates is None:
            return 0

        # Only works for DoEOptimalityCriterion (model-based criteria)
        if not isinstance(self._data_model.criterion, DoEOptimalityCriterion):
            raise ValueError(
                "get_candidate_rank() only works with DoEOptimalityCriterion"
            )

        # Step 1: get_relaxed_domain(original_domain)
        (
            relaxed_domain,
            mappings_categorical_var_key_to_aux_var_key_state_pairs,
            mapping_discrete_input_to_discrete_aux,
            aux_vars_for_discrete,
            mapped_aux_categorical_inputs,
            mapped_continous_inputs,
        ) = create_continuous_domain(domain=self.domain)

        # Step 2: Properly encode candidates to relaxed domain
        relaxed_candidates = encode_candidates_to_relaxed_domain(
            candidates=self.candidates,
            mappings_categorical_var_key_to_aux_var_key_state_pairs=mappings_categorical_var_key_to_aux_var_key_state_pairs,
            mapping_discrete_input_to_discrete_aux=mapping_discrete_input_to_discrete_aux,
            domain=self.domain,
        )

        # Step 3: get_objective_function (combines model + objective)
        n_candidates = len(self.candidates)
        objective_function = get_objective_function(
            criterion=self._data_model.criterion,
            domain=relaxed_domain,
            n_experiments=n_candidates,
            inputs_for_formula=self.domain.inputs,
        )

        # Step 4 & 5: Combined tensor_to_model_matrix + rank calculation
        if isinstance(objective_function, ModelBasedObjective):
            # Ensure we only use columns that match the relaxed domain inputs
            expected_columns = relaxed_domain.inputs.get_keys()
            relaxed_candidates_clean = relaxed_candidates[expected_columns]

            # Convert to tensor
            candidates_tensor = torch.tensor(
                relaxed_candidates_clean.to_numpy(), dtype=torch.float64
            )

            # Get candidate model matrix using objective
            candidates_model_matrix = objective_function.tensor_to_model_matrix(
                candidates_tensor
            )

            model_matrix_rank = torch.linalg.matrix_rank(candidates_model_matrix).item()

            return model_matrix_rank

        else:
            raise ValueError(
                "Only ModelBasedObjective supports Fisher Information Matrix rank calculation"
            )

    def get_additional_experiments_needed(self) -> Optional[int]:
        """Calculate the additional number of experiments needed beyond current candidates.
        This method computes: get_required_number_of_experiments() - get_candidate_rank()

        Returns:
            Optional[int]: Number of additional experiments needed, or None if required number
                          cannot be calculated (e.g., for SpaceFillingCriterion).
        """
        required_experiments = self.get_required_number_of_experiments()
        if required_experiments is None:
            return None

        candidate_rank = self.get_candidate_rank()
        difference = required_experiments - candidate_rank
        return difference

    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise

        """
        return True

    def _transform_candidates_to_new_domain(
        self, domain: Domain, candidates: pd.DataFrame
    ) -> pd.DataFrame:
        new_candidates = candidates.copy()
        new_candidates[
            [key for key in domain.inputs.get_keys() if key not in candidates.columns]
        ] = None
        return new_candidates

    @classmethod
    def make(
        cls,
        domain: Domain,
        seed: int | None = None,
        criterion: AnyOptimalityCriterion | None = None,
        verbose: bool | None = None,
        ipopt_options: Dict | None = None,
        scip_params: Dict | None = None,
        use_hessian: bool | None = None,
        use_cyipopt: bool | None = None,
        sampling: List[List[float]] | None = None,
        return_fixed_candidates: bool | None = None,
    ) -> Self:
        """
        Create a new design of experimence strategy instance.
        Args:
            domain: The domain for the strategy.
            seed: Random seed for reproducibility.
            criterion: Optimality criterion for the strategy. Default is d-optimality.
            verbose: Verbosity level.
            ipopt_options: Options for IPOPT solver. IPOPT is used to minize the optimality criterion.
            scip_params: Parameters for SCIP solver. SCIP is used to for backprojection of
                         discrete and categorical variables.
            use_hessian: Whether to use Hessian information.
            use_cyipopt: Whether to use cyipopt.
            sampling: Initial points for the strategy.
            return_fixed_candidates: Whether to return fixed candidates.
        Returns:
            DoEStrategy: A new instance of the DoEStrategy class.
        """
        return cast(Self, make_strategy(cls, data_models.DoEStrategy, locals()))
