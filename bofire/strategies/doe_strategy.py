from typing import Dict, List, Optional, cast

import pandas as pd
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
from bofire.strategies.doe.objective import get_objective_function
from bofire.strategies.doe.utils import get_formula_from_string, n_zero_eigvals
from bofire.strategies.doe.utils_categorical_discrete import (
    create_continuous_domain,
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
        self._return_fixed_candidates = data_model.return_fixed_candidates

    def set_candidates(self, candidates: pd.DataFrame):
        original_columns = self.domain.inputs.get_keys(includes=Input)
        to_many_columns = []
        for col in candidates.columns:
            if col not in original_columns:
                to_many_columns.append(col)
        if len(to_many_columns) > 0:
            raise AttributeError(
                f"provided candidates have columns: {(*to_many_columns,)},  which do not exist in original domain",
            )

        to_few_columns = []
        for col in original_columns:
            if col not in candidates.columns:
                to_few_columns.append(col)
        if len(to_few_columns) > 0:
            raise AttributeError(
                f"provided candidates are missing columns: {(*to_few_columns,)} which exist in original domain",
            )

        self._candidates = candidates

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:  # type: ignore
        (
            relaxed_domain,
            mappings_categorical_var_key_to_aux_var_key_state_pairs,
            mapping_discrete_input_to_discrete_aux,
            aux_vars_for_discrete,
            mapped_aux_categorical_inputs,
            mapped_continous_inputs,
        ) = create_continuous_domain(domain=self.domain)
        fixed_experiments_count = 0
        _candidate_count = candidate_count
        if self.candidates is not None:
            adapted_partially_fixed_candidates = (
                self._transform_candidates_to_new_domain(
                    relaxed_domain,
                    self.candidates,
                )
            )
        else:
            adapted_partially_fixed_candidates = None
        if self.candidates is not None:
            fixed_experiments_count = self.candidates.notnull().all(axis=1).sum()
            _candidate_count = candidate_count + fixed_experiments_count
        objective_function = get_objective_function(
            self._data_model.criterion,
            domain=relaxed_domain,
            n_experiments=_candidate_count,
            inputs_for_formula=self.domain.inputs,
        )
        assert objective_function is not None, "Criterion type is not supported!"
        design = find_local_max_ipopt(
            relaxed_domain,
            fixed_experiments=None,
            partially_fixed_experiments=adapted_partially_fixed_candidates,
            ipopt_options=self._data_model.ipopt_options,
            objective_function=objective_function,
        )
        if len(self.domain.inputs.get([DiscreteInput, CategoricalInput])) > 0:
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
        if self._return_fixed_candidates:
            fixed_experiments_count = 0
        return design.iloc[fixed_experiments_count:, :].reset_index(
            drop=True,
        )

    def get_required_number_of_experiments(self) -> Optional[int]:
        if isinstance(self._data_model.criterion, DoEOptimalityCriterion):
            if self.domain.inputs.get([DiscreteInput, CategoricalInput]):
                _domain, _, _, _, _, _ = create_continuous_domain(domain=self.domain)
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
