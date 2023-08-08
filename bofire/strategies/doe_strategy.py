import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.features.api import (
    Input,
)
from bofire.strategies.doe.design import (
    find_local_max_ipopt,
    find_local_max_ipopt_BaB,
    find_local_max_ipopt_exhaustive,
)
from bofire.strategies.doe.utils_categorical_discrete import (
    design_from_new_to_original_domain,
    discrete_to_relaxable_domain_mapper,
    nchoosek_to_relaxable_domain_mapper,
    validate_categorical_groups,
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
        all_new_categories = []

        # map categorical/ discrete Domain to a relaxable Domain
        new_domain, new_categories = discrete_to_relaxable_domain_mapper(self.domain)
        all_new_categories.extend(new_categories)

        # check for NchooseK constraint and solve the problem differently depending on the strategy
        if self.data_model.optimization_strategy != "partially-random":
            (
                new_domain,
                new_categories,
                new_variables,
            ) = nchoosek_to_relaxable_domain_mapper(new_domain)
            all_new_categories.extend(new_categories)

        # check categorical_groups
        validate_categorical_groups(all_new_categories, new_domain)

        # here we adapt the (partially) fixed experiments to the new domain
        # todo

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
                categorical_groups=all_new_categories,
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
                categorical_groups=all_new_categories,
            )
        else:
            raise RuntimeError("Could not find suitable optimization strategy")

        # mapping the solution to the variables from the original domain
        transformed_design = design_from_new_to_original_domain(self.domain, design)

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
