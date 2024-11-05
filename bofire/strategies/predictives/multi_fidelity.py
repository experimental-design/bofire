import pandas as pd

from bofire.data_models.features.api import TaskInput
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityStrategy as DataModel,
)
from bofire.strategies.predictives.sobo import SoboStrategy


class MultiFidelityStrategy(SoboStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.task_feature_key = self.domain.inputs.get_keys(TaskInput)[0]

        ft = data_model.fidelity_thresholds
        M = len(self.domain.inputs.get_by_key(self.task_feature_key).fidelities)  # type: ignore
        self.fidelity_thresholds = ft if isinstance(ft, list) else [ft] * M

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        """Generate new candidates (x, m).

        This is a greedy optimization of the acquisition function. We first
        optimize the acqf for the target fidelity to generate a candidate x,
        then select the lowest fidelity that has a variance exceeding a
        threshold.

        Args:
            candidate_count (int): number of candidates to be generated

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        task_feature: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        task_feature.allowed = [True] + [False] * (len(task_feature.categories) - 1)
        x = super()._ask(candidate_count)
        task_feature.allowed = [True] * len(task_feature.categories)
        m = self._select_fidelity(x)
        candidates = x.assign(**{self.task_feature_key: m})
        # TODO: assign the true pred and sd (not the one from the target fn)
        return candidates

    def _select_fidelity(self, X: pd.DataFrame) -> int:
        """Select the fidelity for a given input.

        Uses the variance based approach (see [Kandasamy et al. 2016,
        Folch et al. 2023]) to select the lowest fidelity that has a variance
        exceeding a threshold.

        Args:
            X (pd.DataFrame): optimum input of target fidelity

        Returns:
            int: selected fidelity
        """
        fidelity_input: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        assert self.model is not None

        fidelities = list(zip(fidelity_input.fidelities, fidelity_input.categories))
        for m, fidelity in reversed(fidelities[1:]):
            X_fid = X.assign(**{self.task_feature_key: fidelity})
            transformed = self.domain.inputs.transform(
                experiments=X_fid, specs=self.input_preprocessing_specs
            )
            _, std = self._predict(transformed)

            if std > self.fidelity_thresholds[m]:
                return fidelity

        # if no low fidelity is selected, return the target fidelity
        return fidelity_input.categories[0]
