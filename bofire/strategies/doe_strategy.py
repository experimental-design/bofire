import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.features.api import CategoricalInput, Input
from bofire.strategies.doe.design import (
    find_local_max_ipopt,
    find_local_max_ipopt_BaB,
    find_local_max_ipopt_exhaustive,
)
from bofire.strategies.doe.utils_categorical_discrete import (
    design_from_new_to_original_domain,
    discrete_to_relaxable_domain_mapper,
    nchoosek_to_relaxable_domain_mapper,
)
from bofire.strategies.strategy import Strategy


class DoEStrategy(Strategy):
    """Strategy for design of experiments. This strategy is used to generate a set of
    experiments for a given domain.
    The experiments are generated via minimization of the D-optimality criterion.

    """

    def __init__(
        self,
        data_model: data_models.DoEStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.formula = data_model.formula
        self.data_model = data_model
        self._partially_fixed_candidates = None
        self._fixed_candidates = None

    def set_candidates(self, candidates: pd.DataFrame):
        original_columns = self.domain.get_feature_keys(includes=Input)
        to_many_columns = []
        for col in candidates.columns:
            if col not in original_columns:
                to_many_columns.append(col)
        if len(to_many_columns) > 0:
            raise AttributeError(
                f"provided candidates have columns: {*to_many_columns,},  which do not exist in original domain"
            )

        to_few_columns = []
        for col in original_columns:
            if col not in candidates.columns:
                to_few_columns.append(col)
        if len(to_few_columns) > 0:
            raise AttributeError(
                f"provided candidates are missing columns: {*to_few_columns,} which exist in original domain"
            )

        self._candidates = candidates

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        all_new_categories = []

        # map categorical/ discrete Domain to a relaxable Domain
        new_domain, new_categories, new_discretes = discrete_to_relaxable_domain_mapper(
            self.domain
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
        fixed_experiments_count = 0
        _candidate_count = candidate_count
        adapted_partially_fixed_candidates = self._transform_candidates_to_new_domain(
            new_domain, self.candidates
        )

        if self.candidates is not None:
            fixed_experiments_count = self.candidates.notnull().all(axis=1).sum()
            _candidate_count = candidate_count + fixed_experiments_count

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
                self.formula,
                n_experiments=_candidate_count,
                fixed_experiments=None,
                partially_fixed_experiments=adapted_partially_fixed_candidates,
            )
        # todo adapt to when exhaustive search accepts discrete variables
        elif (
            self.data_model.optimization_strategy == "exhaustive"
            and num_discrete_vars == 0
        ):
            design = find_local_max_ipopt_exhaustive(
                domain=new_domain,
                model_type=self.formula,
                n_experiments=_candidate_count,
                fixed_experiments=None,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=adapted_partially_fixed_candidates,
                categorical_groups=all_new_categories,
                discrete_variables=new_discretes,
            )
        elif self.data_model.optimization_strategy in [
            "branch-and-bound",
            "default",
            "partially-random",
        ]:
            design = find_local_max_ipopt_BaB(
                domain=new_domain,
                model_type=self.formula,
                n_experiments=_candidate_count,
                fixed_experiments=None,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=adapted_partially_fixed_candidates,
                categorical_groups=all_new_categories,
                discrete_variables=new_discretes,
            )
        elif self.data_model.optimization_strategy == "iterative":
            # a dynamic programming approach to shrink the optimization space by optimizing one experiment at a time
            assert (
                _candidate_count is not None
            ), "strategy iterative requires number of experiments to be set!"

            num_adapted_partially_fixed_candidates = 0
            if adapted_partially_fixed_candidates is not None:
                num_adapted_partially_fixed_candidates = len(
                    adapted_partially_fixed_candidates
                )
            design = None
            for i in range(_candidate_count):
                design = find_local_max_ipopt_BaB(
                    domain=new_domain,
                    model_type=self.formula,
                    n_experiments=num_adapted_partially_fixed_candidates + i + 1,
                    fixed_experiments=None,
                    verbose=self.data_model.verbose,
                    partially_fixed_experiments=adapted_partially_fixed_candidates,
                    categorical_groups=all_new_categories,
                    discrete_variables=new_discretes,
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
                    f"Status: {i+1} of {_candidate_count} experiments determined \n"
                    f"Current experimental plan:\n {design_from_new_to_original_domain(self.domain, design)}"
                )

        else:
            raise RuntimeError("Could not find suitable optimization strategy")

        # mapping the solution to the variables from the original domain
        transformed_design = design_from_new_to_original_domain(self.domain, design)

        return transformed_design.iloc[fixed_experiments_count:, :].reset_index(drop=True)  # type: ignore

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

            cat_columns = self.domain.get_features(includes=CategoricalInput)
            for cat in cat_columns:
                for row_index, c in enumerate(intermediate_candidates[cat.key].values):
                    if pd.isnull(c):
                        continue
                    if c not in cat.categories:  # type: ignore
                        raise AttributeError(
                            f"provided value {c} for categorical variable {cat.key} does not exist in the corresponding categories {cat.categories}"  # type: ignore
                        )
                    intermediate_candidates.loc[row_index, cat.categories] = 0  # type: ignore
                    intermediate_candidates.loc[row_index, c] = 1

            intermediate_candidates = intermediate_candidates.drop(
                [cat.key for cat in cat_columns], axis=1
            )

            adapted_partially_fixed_candidates = pd.concat(
                [
                    intermediate_candidates[candidates.notnull().all(axis=1)],
                    intermediate_candidates[candidates.isnull().any(axis=1)],
                ]
            )
            return adapted_partially_fixed_candidates
        return None
