from typing import List

import numpy as np
import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    DiscreteInput,
    Input,
    Output,
)
from bofire.strategies.doe.design import (
    find_local_max_ipopt,
    find_local_max_ipopt_BaB,
    find_local_max_ipopt_exhaustive,
)
from bofire.strategies.doe.utils import (
    NChooseKGroup,
    discrete_to_relaxable_domain_mapper,
)
from bofire.strategies.doe.utils_features import (
    RelaxableBinaryInput,
    RelaxableDiscreteInput,
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
        self._partially_fixed_experiments_for_next_design = None

    def tell(
        self,
        experiments: pd.DataFrame,
        replace: bool = False,
    ) -> None:
        """This function passes new experimental data to the optimizer

        Args:
            experiments (pd.DataFrame): DataFrame with experimental data
            replace (bool, optional): Boolean to decide if the experimental data should replace the former DataFrame or if the new experiments should be attached. Defaults to False.
        """

        self._partially_fixed_experiments_for_next_design = experiments[
            experiments.isnull().any(axis=1)
        ][self.domain.get_feature_keys(includes=Input)]
        experiments = experiments[experiments.notnull().all(axis=1)]

        if len(experiments) == 0:
            return
        if replace:
            self.set_experiments(experiments=experiments)
        else:
            self.add_experiments(experiments=experiments)
        # we check here that the experiments do not have completely fixed columns
        cleaned_experiments = (
            self.domain.outputs.preprocess_experiments_all_valid_outputs(
                experiments=experiments
            )
        )
        for feature in self.domain.inputs.get_fixed():
            if (cleaned_experiments[feature.key] == feature.fixed_value()[0]).all():  # type: ignore
                raise ValueError(
                    f"No variance in experiments for fixed feature {feature.key}"
                )
        self._tell()

    def _tell(self) -> None:
        self.set_candidates(
            self.experiments[self.domain.get_feature_keys(includes=Input)]
        )

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        # map categorical/ discrete Domain to a relaxable Domain
        new_domain, categorical_groups = discrete_to_relaxable_domain_mapper(
            self.domain
        )
        # check for NchooseK constraint and solve the problem differently depending on the strategy
        new_vars = []
        new_constr = []
        var_occuring_in_nchoosek = []
        if self.data_model.optimization_strategy != "partially-random":
            n_choose_k_constraints = new_domain.constraints.get(
                includes=NChooseKConstraint
            )
            for constr in n_choose_k_constraints:
                current_features = [self.domain.get_feature(k) for k in constr.features]
                var_occuring_in_nchoosek.extend(constr.features)
                pick_at_least = constr.min_count
                pick_at_most = constr.max_count
                none_also_valid = constr.none_also_valid
                new_relaxable_categorical_vars, new_constraints = NChooseKGroup(
                    current_features, pick_at_least, pick_at_most, none_also_valid
                )
                new_vars.append(new_relaxable_categorical_vars)
                new_constr.extend(new_constraints)

                # allow vars to be set to 0
                for var in var_occuring_in_nchoosek:
                    current_var = new_domain.inputs.get_by_key(var)
                    if current_var.lower_bound > 0:
                        current_var.bounds = (0, current_var.upper_bound)
                    elif current_var.upper_bound < 0:
                        current_var.bounds = (current_var.lower_bound, 0)

        add_inputs = [var for group in new_vars for var in group]
        if len(add_inputs) > 0:
            new_domain.inputs = new_domain.inputs + [
                var for group in new_vars for var in group
            ]

        if len(new_constr) > 0:
            new_domain.constraints = (
                new_domain.constraints.get(excludes=NChooseKConstraint) + new_constr
            )

        categorical_groups.extend(new_vars)
        # here we adapt the (partially) fixed experiments to the new domain
        # todo

        # check categorical_groups
        validate_categorical_groups(categorical_groups, new_domain)
        if self.candidates is not None:
            _fixed_experiments_count = len(self.candidates)
            _candidate_count = candidate_count + len(self.candidates)
        else:
            _fixed_experiments_count = 0
            _candidate_count = candidate_count

        num_binary_vars = len(new_domain.get_features(includes=[RelaxableBinaryInput]))
        num_discrete_vars = len(
            new_domain.get_features(includes=[RelaxableDiscreteInput])
        )

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
                fixed_experiments=self.candidates,
                partially_fixed_experiments=self._partially_fixed_experiments_for_next_design,
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
                fixed_experiments=self.candidates,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=self._partially_fixed_experiments_for_next_design,
                categorical_groups=categorical_groups,
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
                fixed_experiments=self.candidates,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=self._partially_fixed_experiments_for_next_design,
                categorical_groups=categorical_groups,
            )
        else:
            raise RuntimeError("Could not find suitable optimization strategy")
        # mapping the solution to the variables from the original domain
        transformed_design = design[
            self.domain.get_feature_keys(excludes=[CategoricalInput, Output])
        ]
        for group in self.domain.get_features(includes=CategoricalInput):
            categorical_columns = design[group.categories]
            mask = ~np.isclose(categorical_columns.to_numpy(), 0)

            for i, row in enumerate(mask):
                index_to_keep = np.random.choice(np.argwhere(row).flatten())
                mask[i] = np.zeros_like(row, dtype=bool)
                mask[i][index_to_keep] = True

            categorical_columns = categorical_columns.where(
                np.invert(mask),
                pd.DataFrame(
                    np.full(
                        (len(categorical_columns), len(group.categories)),
                        group.categories,
                    ),
                    columns=categorical_columns.columns,
                    index=categorical_columns.index,
                ),
            )
            categorical_columns = categorical_columns.where(
                mask,
                pd.DataFrame(
                    np.full((len(categorical_columns), len(group.categories)), ""),
                    columns=categorical_columns.columns,
                    index=categorical_columns.index,
                ),
            )
            transformed_design[group.key] = categorical_columns.apply("".join, axis=1)

        for var in self.domain.get_features(includes=DiscreteInput):
            closest_solution = var.from_continuous(transformed_design)
            transformed_design[var.key] = closest_solution

        # restart the partially fixed experiments
        self._partially_fixed_experiments_for_next_design = None
        return transformed_design.iloc[_fixed_experiments_count:, :]  # type: ignore

    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise
        """
        return True


def validate_categorical_groups(
    categorical_group: List[List[RelaxableBinaryInput]], domain: Domain
):
    """Validate if features given as the categorical groups are also features in the domain and if each feature
    is in exactly one group

    Args: categorical_group (List[List[RelaxableBinaryInput]]) : groups of the different categories
    domain (Domain): Domain to test against

    Raises
        ValueError: Feature key not registered in any group or registered too often.

    Returns:
        List[List[RelaxableBinaryInput]]: groups of the different categories
    """

    bin_vars = domain.inputs.get_keys(includes=RelaxableBinaryInput)

    if len(bin_vars) == 0:
        return categorical_group

    simplified_groups = [[f.key for f in group] for group in categorical_group]
    groups_flattened = [var.key for group in categorical_group for var in group]
    for k in bin_vars:
        if groups_flattened.count(k) < 1:
            raise ValueError(
                f"feature {k} is not registered in any of the categorical groups {simplified_groups}."
            )
        elif groups_flattened.count(k) > 1:
            raise ValueError(
                f"feature {k} is registered to often in the categorical groups {simplified_groups}."
            )
    return categorical_group
