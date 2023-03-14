from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import PositiveInt

from bofire.data_models.features.api import TInputTransformSpecs
from bofire.data_models.objectives.api import Objective
from bofire.data_models.strategies.api import Strategy as DataModel
from bofire.strategies.data_models.candidate import Candidate
from bofire.strategies.data_models.values import InputValue, OutputValue
from bofire.strategies.strategy import Strategy


class PredictiveStrategy(Strategy):
    """Base class for all model based strategies.

    Provides abstract scaffold for fit, predict, and calc_acquistion methods.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.is_fitted = False

    @property
    @abstractmethod
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
        pass

    def ask(
        self,
        candidate_count: Optional[PositiveInt] = None,
        add_pending: bool = False,
        candidate_pool: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Function to generate new candidates.

        Args:
            candidate_count (PositiveInt, optional): Number of candidates to be generated. If not provided, the number of candidates is determined automatically. Defaults to None.
            add_pending (bool, optional): If true the proposed candidates are added to the set of pending experiments. Defaults to False.
            candidate_pool (pd.DataFrame, optional): Pool of candidates from which a final set of candidates should be chosen. If not provided, pool independent candidates are provided. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        candidates = super().ask(
            candidate_count=candidate_count,
            add_pending=add_pending,
            candidate_pool=candidate_pool,
        )
        # we have to generate predictions for the candidate pool candidates
        if candidate_pool is not None:
            pred = self.predict(candidates)
            pred.index = candidates.index
            candidates = pd.concat([candidates, pred], axis=1)
        self.domain.validate_candidates(candidates=candidates)
        return candidates

    def tell(
        self, experiments: pd.DataFrame, replace: bool = False, retrain: bool = True
    ):
        """This function passes new experimental data to the optimizer.

        Args:
            experiments (pd.DataFrame): DataFrame with experimental data
            replace (bool, optional): Boolean to decide if the experimental data should replace the former dataFrame or if the new experiments should be attached. Defaults to False.
            retrain (bool, optional): If True, model(s) are retrained when new experimental data is passed to the optimizer. Defaults to True.
        """
        # maybe unite the preprocessor here with the one of the parent tell
        # TODO: add self.domain.validate_experiments(self.experiments, strict=True) here to ensure variance in each feature?
        if len(experiments) == 0:
            return
        if replace:
            self.set_experiments(experiments)
        else:
            self.add_experiments(experiments)
        if retrain and self.has_sufficient_experiments():
            self.fit()
            # we have a seperate _tell here for things that are relevant when setting up the strategy but unrelated
            # to fitting the models like initializing the ACQF.
            self._tell()

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for the provided experiments. Only input features have to be provided.

        Args:
            experiments (pd.DataFrame): Experimental data for which predictions should be performed.

        Returns:
            pd.DataFrame: Dataframe with the predicted values.
        """
        if self.is_fitted is not True:
            raise ValueError("Model not yet fitted.")
        # TODO: validate also here the experiments but only for the input_columns
        # transformed = self.transformer.transform(experiments)
        transformed = self.domain.inputs.transform(
            experiments=experiments, specs=self.input_preprocessing_specs
        )
        preds, stds = self._predict(transformed)
        if stds is not None:
            predictions = pd.DataFrame(
                data=np.hstack((preds, stds)),
                columns=[
                    "%s_pred" % feat.key
                    for feat in self.domain.outputs.get_by_objective(Objective)
                ]
                + [
                    "%s_sd" % feat.key
                    for feat in self.domain.outputs.get_by_objective(Objective)
                ],
            )
        else:
            predictions = pd.DataFrame(
                data=preds,
                columns=[
                    "%s_pred" % feat.key
                    for feat in self.domain.outputs.get_by_objective(Objective)
                ],
            )
        desis = self.domain.outputs(predictions, predictions=True)
        predictions = pd.concat((predictions, desis), axis=1)
        predictions.index = experiments.index
        return predictions

    @abstractmethod
    def _predict(self, experiments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Abstract method in which the actual prediction is happening. Has to be overwritten."""
        pass

    def fit(self):
        """Fit the model(s) to the experimental data."""
        assert (
            self.experiments is not None and len(self.experiments) > 0
        ), "No fitting data available"
        self.domain.validate_experiments(self.experiments, strict=True)
        # transformed = self.transformer.fit_transform(self.experiments)
        self._fit(self.experiments)
        self.is_fitted = True

    @abstractmethod
    def _fit(self, experiments: pd.DataFrame):
        """Abstract method where the acutal prediction are occuring."""
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
                outputValues={
                    key: OutputValue(
                        predictedValue=row[f"{key}_pred"],
                        standardDeviation=row[f"{key}_sd"],
                        objective=row[f"{key}_des"],
                    )
                    for key in self.domain.outputs.get_keys()
                },
            )
            for _, row in candidates.iterrows()
        ]

    @abstractmethod
    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[PositiveInt] = None,
    ) -> pd.DataFrame:
        """Abstract method to implement how a strategy chooses a set of candidates from a candidate pool.

        Args:
            candidate_pool (pd.DataFrame): The pool of candidates from which the candidates should be chosen.
            candidate_count (Optional[PositiveInt], optional): Number of candidates to choose. Defaults to None.

        Returns:
            pd.DataFrame: The chosen set of candidates.
        """
        pass

    @abstractmethod
    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise
        """
        pass