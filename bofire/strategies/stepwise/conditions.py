from abc import abstractmethod
from typing import Union

import pandas as pd

import bofire.data_models.strategies.stepwise.conditions as data_models
from bofire.data_models.domain.api import Domain


class Condition(object):
    @abstractmethod
    def evaluate(self, domain: Domain, experiments: pd.DataFrame) -> bool:
        pass


class NumberOfExperimentsCondition(Condition):
    def __init__(self, data_model: data_models.NumberOfExperimentsCondition):
        self.n_experiments = data_model.n_experiments

    def evaluate(self, domain: Domain, experiments: Union[pd.DataFrame, None]) -> bool:
        if experiments is None:
            n_experiments = 0
        else:
            n_experiments = len(
                domain.outputs.preprocess_experiments_all_valid_outputs(experiments)
            )
        return n_experiments <= self.n_experiments


class AlwaysTrueCondition(Condition):
    def __init__(self, data_model: data_models.AlwaysTrueCondition):
        pass

    def evaluate(self, domain: Domain, experiments: pd.DataFrame) -> bool:
        return True


class CombiCondition(Condition):
    def __init__(self, data_model: data_models.CombiCondition) -> None:
        self.conditions = [map(c) for c in data_model.conditions]  # type: ignore
        self.n_required_conditions = data_model.n_required_conditions

    def evaluate(self, domain: Domain, experiments: pd.DataFrame) -> bool:
        n_matched_conditions = 0
        for c in self.conditions:
            if c.evaluate(domain, experiments):
                n_matched_conditions += 1
        if n_matched_conditions >= self.n_required_conditions:
            return True
        return False


CONDITION_MAP = {
    data_models.CombiCondition: CombiCondition,
    data_models.NumberOfExperimentsCondition: NumberOfExperimentsCondition,
    data_models.AlwaysTrueCondition: AlwaysTrueCondition,
}


def map(
    data_model: Union[
        data_models.CombiCondition,
        data_models.NumberOfExperimentsCondition,
        data_models.AlwaysTrueCondition,
    ],
) -> Union[CombiCondition, NumberOfExperimentsCondition, AlwaysTrueCondition]:
    return CONDITION_MAP[data_model.__class__](data_model)
