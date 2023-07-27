import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.features.api import (
    ContinuousBinaryInput,
    ContinuousDiscreteInput,
    Input,
)
from bofire.strategies.doe.design import (
    find_local_max_ipopt,
    find_local_max_ipopt_BaB,
    find_local_max_ipopt_binary_naive,
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
        if self.candidates is not None:
            _fixed_experiments_count = len(self.candidates)
            _candidate_count = candidate_count + len(self.candidates)
        else:
            _fixed_experiments_count = 0
            _candidate_count = candidate_count

        num_binary_vars = len(
            self.domain.get_features(includes=[ContinuousBinaryInput])
        )
        num_discrete_vars = len(
            self.domain.get_features(includes=[ContinuousDiscreteInput])
        )

        if self.data_model.optimization_strategy == "relaxed" or (
            num_binary_vars == 0 and num_discrete_vars == 0
        ):
            design = find_local_max_ipopt(
                self.domain,
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
            design = find_local_max_ipopt_binary_naive(
                domain=self.domain,
                model_type=self.formula,
                n_experiments=_candidate_count,
                fixed_experiments=self.candidates,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=self._partially_fixed_experiments_for_next_design,
            )
        elif self.data_model.optimization_strategy == "branch-and-bound":
            design = find_local_max_ipopt_BaB(
                domain=self.domain,
                model_type=self.formula,
                n_experiments=_candidate_count,
                fixed_experiments=self.candidates,
                verbose=self.data_model.verbose,
                partially_fixed_experiments=self._partially_fixed_experiments_for_next_design,
            )
        else:
            raise RuntimeError("Could not find suitable optimization strategy")

        # restart the partially fixed experiments
        self._partially_fixed_experiments_for_next_design = None
        return design.iloc[_fixed_experiments_count:, :]  # type: ignore

    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise
        """
        return True
