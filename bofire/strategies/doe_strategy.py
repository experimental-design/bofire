from typing import Optional

import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.doe import (
    AnyDoEOptimalityCriterion,
    DoEOptimalityCriterion,
)
from bofire.strategies.doe.branch_and_bound import (
    find_local_max_ipopt_BaB,
    find_local_max_ipopt_exhaustive,
)
from bofire.strategies.doe.design import find_local_max_ipopt, get_n_experiments
from bofire.strategies.doe.utils import get_formula_from_string, n_zero_eigvals
from bofire.strategies.doe.utils_categorical_discrete import (
    design_from_new_to_original_domain,
    discrete_to_relaxable_domain_mapper,
    nchoosek_to_relaxable_domain_mapper,
)
from bofire.strategies.strategy import Strategy


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
        self.data_model = data_model
        self._allow_partially_filled_candidates = True

    @property
    def formula(self):
        if isinstance(self.data_model.criterion, DoEOptimalityCriterion):
            return get_formula_from_string(
                self.data_model.criterion.formula, self.data_model.domain
            )
        return None

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:  # type: ignore
        all_new_categories = []

        # map categorical/ discrete Domain to a relaxable Domain
        new_domain, new_categories, new_discretes = discrete_to_relaxable_domain_mapper(
            self.domain,
        )
        all_new_categories.extend(new_categories)

        # check for NchooseK constraint and solve the problem differently depending on the strategy
        if self.data_model.optimization_strategy != "partially-random":
            (
                new_domain,
                new_categories,
            ) = nchoosek_to_relaxable_domain_mapper(new_domain)
            all_new_categories.extend(new_categories)

        # here we adapt the (partially) fixed experiments to the new domain
        _candidate_count = candidate_count
        adapted_partially_fixed_candidates = self._transform_candidates_to_new_domain(
            new_domain,
            self.candidates,
        )
        # not yet working,
        # target is to also condition on self.experiments
        if self.candidates is not None:
            fixed_experiments_count = self.candidates.notnull().all(axis=1).sum()
            _candidate_count = candidate_count + fixed_experiments_count
            adapted_partially_fixed_candidates = (
                self._transform_candidates_to_new_domain(
                    new_domain,
                    self.candidates,
                )
            )

        # we have to also adapt the experiments, commented now to convince ruff for now
        # if self.experiments is not None:
        #     adapted_fixed_experiments = self._transform_candidates_to_new_domain(
        #         new_domain,
        #         self.experiments,
        #     )

        num_binary_vars = len([var for group in new_categories for var in group])
        num_discrete_vars = len(new_discretes)
        if (
            self.data_model.optimization_strategy == "relaxed"
            or (num_binary_vars == 0 and num_discrete_vars == 0)
            or (
                self.data_model.optimization_strategy == "partially-random"
                and num_binary_vars == 0
                and num_discrete_vars == 0
            )
        ):
            design = find_local_max_ipopt(
                new_domain,
                n_experiments=_candidate_count,
                fixed_experiments=None,
                partially_fixed_experiments=adapted_partially_fixed_candidates,
                ipopt_options=self.data_model.ipopt_options,
                criterion=self.data_model.criterion,
            )
        # TODO adapt to when exhaustive search accepts discrete variables
        elif (
            self.data_model.optimization_strategy == "exhaustive"
            and num_discrete_vars == 0
        ):
            design = find_local_max_ipopt_exhaustive(
                domain=new_domain,
                n_experiments=_candidate_count,
                fixed_experiments=None,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=adapted_partially_fixed_candidates,
                categorical_groups=all_new_categories,
                discrete_variables=new_discretes,
                ipopt_options=self.data_model.ipopt_options,
                criterion=self.data_model.criterion,
            )
        elif self.data_model.optimization_strategy in [
            "branch-and-bound",
            "default",
            "partially-random",
        ]:
            design = find_local_max_ipopt_BaB(
                domain=new_domain,
                n_experiments=_candidate_count,
                fixed_experiments=None,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=adapted_partially_fixed_candidates,
                categorical_groups=all_new_categories,
                discrete_variables=new_discretes,
                ipopt_options=self.data_model.ipopt_options,
                criterion=self.data_model.criterion,
            )
        elif self.data_model.optimization_strategy == "iterative":
            # a dynamic programming approach to shrink the optimization space by optimizing one experiment at a time
            assert (
                _candidate_count is not None
            ), "strategy iterative requires number of experiments to be set!"

            num_adapted_partially_fixed_candidates = 0
            if adapted_partially_fixed_candidates is not None:
                num_adapted_partially_fixed_candidates = len(
                    adapted_partially_fixed_candidates,
                )
            design = None
            for i in range(_candidate_count):
                design = find_local_max_ipopt_BaB(
                    domain=new_domain,
                    n_experiments=num_adapted_partially_fixed_candidates + i + 1,
                    fixed_experiments=None,
                    verbose=self.data_model.verbose,
                    partially_fixed_experiments=adapted_partially_fixed_candidates,
                    categorical_groups=all_new_categories,
                    discrete_variables=new_discretes,
                    ipopt_options=self.data_model.ipopt_options,
                    criterion=self.data_model.criterion,
                )
                adapted_partially_fixed_candidates = pd.concat(
                    [
                        adapted_partially_fixed_candidates,
                        design.round(6).tail(1),
                    ],
                    axis=0,
                    ignore_index=True,
                )
                print(
                    f"Status: {i + 1} of {_candidate_count} experiments determined \n"
                    f"Current experimental plan:\n {design_from_new_to_original_domain(self.domain, design)}",
                )

        else:
            raise RuntimeError("Could not find suitable optimization strategy")

        # mapping the solution to the variables from the original domain
        transformed_design = design_from_new_to_original_domain(self.domain, design)  # type: ignore

        return transformed_design.iloc[fixed_experiments_count:, :].reset_index(
            drop=True,
        )

    def get_required_number_of_experiments(self) -> Optional[int]:
        if self.formula:
            return get_n_experiments(self.formula) - n_zero_eigvals(
                domain=self.data_model.domain, model_type=self.formula
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

    def _transform_candidates_to_new_domain(self, new_domain, candidates):
        if candidates is not None:
            intermediate_candidates = candidates.copy()
            missing_columns = [
                key
                for key in new_domain.inputs.get_keys()
                if key not in candidates.columns
            ]

            for col in missing_columns:
                intermediate_candidates.insert(0, col, None)

            # this is doing the one-hot encoding in a well tested way
            intermediate_candidates = self.domain.inputs.transform(
                intermediate_candidates,
                {
                    key: CategoricalEncodingEnum.ONE_HOT
                    for key in self.domain.inputs.get_keys(CategoricalInput)
                },
            )

            # cat_columns = self.domain.inputs.get(includes=CategoricalInput)
            # for cat in cat_columns:
            #     for row_index, c in enumerate(intermediate_candidates[cat.key].values):
            #         if pd.isnull(c):
            #             continue
            #         if c not in cat.categories:  # type: ignore
            #             raise AttributeError(
            #                 f"provided value {c} for categorical variable {cat.key} "
            #                 f"does not exist in the corresponding categories {cat.categories}",  # type: ignore
            #             )
            #         intermediate_candidates.loc[row_index, cat.categories] = 0  # type: ignore
            #         intermediate_candidates.loc[row_index, c] = 1

            # intermediate_candidates = intermediate_candidates.drop(
            #     [cat.key for cat in cat_columns],
            #     axis=1,
            # )

            # What is this doing?
            adapted_partially_fixed_candidates = pd.concat(
                [
                    intermediate_candidates[candidates.notnull().all(axis=1)],
                    intermediate_candidates[candidates.isnull().any(axis=1)],
                ],
            )
            return adapted_partially_fixed_candidates
        return None
