from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import PositiveInt

from bofire.data_models.strategies.api import Strategy as DataModel
from bofire.strategies.data_models.candidate import Candidate
from bofire.strategies.data_models.values import InputValue


class Strategy(ABC):
    """Base class for all strategies

    Attributes:
    """

    def __init__(
        self,
        data_model: DataModel,
    ):
        self.domain = data_model.domain
        self.seed = data_model.seed or np.random.default_rng().integers(1000)
        self.rng = np.random.default_rng(self.seed)  # type: ignore
        self._experiments = None
        self._candidates = None

    @classmethod
    def from_spec(cls, data_model: DataModel) -> "Strategy":
        """Used by the mapper to map from data model to functional strategy."""
        return cls(data_model=data_model)

    @property
    def experiments(self) -> Optional[pd.DataFrame]:
        """Returns the experiments of the strategy.

        Returns:
            pd.DataFrame: Current experiments.
        """
        return self._experiments

    @property
    def candidates(self) -> Optional[pd.DataFrame]:
        """Returns the (pending) candidates of the strategy.

        Returns:
            pd.DataFrame: Pending experiments.
        """
        return self._candidates

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
        """Method to allow for customized tell functions in addition to self.tell()"""
        pass

    def ask(
        self,
        candidate_count: Optional[PositiveInt] = None,
        add_pending: bool = False,
        raise_validation_error: bool = True,
    ) -> pd.DataFrame:
        """Function to generate new candidates.

        Args:
            candidate_count (PositiveInt, optional): Number of candidates to be generated. If not provided, the number
                of candidates is determined automatically. Defaults to None.
            add_pending (bool, optional): If true the proposed candidates are added to the set of pending experiments. Defaults to False.
            raise_validation_error (bool, optional): If true an error will be raised if candidates violate constraints,
                otherwise only a warning will be displayed. Defaults to True.


        Raises:
            ValueError: if candidate count is smaller than 1
            ValueError: if not enough experiments are available to execute the strategy
            ValueError: if the number of generated candidates does not match the requested number

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        if candidate_count is not None and candidate_count < 1:
            raise ValueError(
                f"Candidate_count has to be at least 1 but got {candidate_count}."
            )
        if not self.has_sufficient_experiments():
            raise ValueError(
                "Not enough experiments available to execute the strategy."
            )

        candidates = self._ask(candidate_count=candidate_count)

        self.domain.validate_candidates(
            candidates=candidates,
            only_inputs=True,
            raise_validation_error=raise_validation_error,
        )

        if candidate_count is not None:
            if len(candidates) != candidate_count:
                raise ValueError(
                    f"expected {candidate_count} candidates, got {len(candidates)}"
                )

        if add_pending:
            self.add_candidates(candidates)

        return candidates

    @abstractmethod
    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise
        """
        pass

    @abstractmethod
    def _ask(
        self,
        candidate_count: Optional[PositiveInt] = None,
    ) -> pd.DataFrame:
        """Abstract method to implement how a strategy generates candidates.

        Args:
            candidate_count (PositiveInt, optional): Number of candidates to be generated. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments).
        """
        pass

    def to_candidates(self, candidates: pd.DataFrame) -> List[Candidate]:
        """Transform candiadtes dataframe to a list of `Candidate` objects.
        Args:
            candidates (pd.DataFrame): candidates formatted as dataframe
        Returns:
            List[Candidate]: candidates formatted as list of `Candidate` objects.
        """
        return [
            Candidate(
                inputValues={
                    key: InputValue(value=row[key])
                    for key in self.domain.inputs.get_keys()
                },
            )
            for _, row in candidates.iterrows()
        ]

    def set_candidates(self, candidates: pd.DataFrame):
        """Set pending candidates of the strategy. Overwrites existing ones.

        Args:
            experiments (pd.DataFrame): Dataframe with candidates.
        """
        candidates = self.domain.inputs.validate_experiments(
            candidates[self.domain.inputs.get_keys()], strict=False
        )
        self._candidates = candidates[self.domain.inputs.get_keys()]

    def add_candidates(self, candidates: pd.DataFrame):
        """Add pending candidates to the strategy. Appends to existing ones.

        Args:
            experiments (pd.DataFrame): Dataframe with candidates.
        """
        candidates = self.domain.inputs.validate_experiments(
            candidates[self.domain.inputs.get_keys()], strict=False
        )
        if self.candidates is None:
            self._candidates = candidates[self.domain.inputs.get_keys()]
        else:
            self._candidates = pd.concat(
                (self.candidates, candidates[self.domain.inputs.get_keys()]),
                ignore_index=True,
            )  # type: ignore

    def reset_candidates(self):
        """Resets the pending candidates of the strategy."""
        self._candidates = None

    @property
    def num_candidates(self) -> int:
        """Returns number of (pending) candidates"""
        if self.candidates is None:
            return 0
        return len(self.candidates)

    def set_experiments(self, experiments: pd.DataFrame):
        """Set experiments of the strategy. Overwrites existing ones.

        Args:
            experiments (pd.DataFrame): Dataframe with experiments.
        """
        experiments = self.domain.validate_experiments(experiments)
        self._experiments = experiments

    def add_experiments(self, experiments: pd.DataFrame):
        """Add experiments to the strategy. Appends to existing ones.

        Args:
            experiments (pd.DataFrame): Dataframe with experiments.
        """
        experiments = self.domain.validate_experiments(experiments)
        if self.experiments is None:
            self._experiments = experiments
        else:
            self._experiments = pd.concat(
                (self.experiments, experiments), ignore_index=True
            )

    @property
    def num_experiments(self) -> int:
        """Returns number of experiments"""
        if self.experiments is None:
            return 0
        return len(self.experiments)
