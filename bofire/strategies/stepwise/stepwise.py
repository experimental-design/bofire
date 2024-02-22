from typing import Dict, Literal, Optional, Tuple, Type, TypeVar, Union

import pandas as pd
from pydantic import PositiveInt

import bofire.data_models.strategies.api as data_models
import bofire.data_models.transforms as transforms
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import StepwiseStrategy as data_model
from bofire.strategies.doe_strategy import DoEStrategy
from bofire.strategies.factorial import FactorialStrategy
from bofire.strategies.predictives.mobo import MoboStrategy
from bofire.strategies.predictives.qehvi import QehviStrategy
from bofire.strategies.predictives.qnehvi import QnehviStrategy
from bofire.strategies.predictives.qparego import QparegoStrategy
from bofire.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.strategies.random import RandomStrategy
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.strategies.space_filling import SpaceFillingStrategy
from bofire.strategies.strategy import Strategy

# we have to duplicate the map functionality due to prevent circular imports
STRATEGY_MAP: Dict[Type[data_models.Strategy], Type[Strategy]] = {
    data_models.RandomStrategy: RandomStrategy,
    data_models.SoboStrategy: SoboStrategy,
    data_models.AdditiveSoboStrategy: AdditiveSoboStrategy,
    data_models.MultiplicativeSoboStrategy: MultiplicativeSoboStrategy,
    data_models.CustomSoboStrategy: CustomSoboStrategy,
    data_models.QehviStrategy: QehviStrategy,
    data_models.QnehviStrategy: QnehviStrategy,
    data_models.QparegoStrategy: QparegoStrategy,
    data_models.SpaceFillingStrategy: SpaceFillingStrategy,
    data_models.DoEStrategy: DoEStrategy,
    data_models.FactorialStrategy: FactorialStrategy,
    data_models.MoboStrategy: MoboStrategy,
    data_models.ShortestPathStrategy: ShortestPathStrategy,
}


def _map(data_model: data_models.Strategy) -> Strategy:
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)


_T = TypeVar("_T", pd.DataFrame, Domain)


def _apply_tf(
    data: Optional[_T],
    transform: Optional[transforms.Transform],
    tf: Union[Literal["experiments"], Literal["candidates"], Literal["domain"]],
) -> Optional[_T]:
    if data is not None and transform is not None:
        return getattr(transform, f"transform_{tf}")(data)


class StepwiseStrategy(Strategy):
    def __init__(self, data_model: data_model, **kwargs):
        super().__init__(data_model, **kwargs)
        self.stratgies = [_map(s.strategy_data) for s in data_model.steps]
        self.conditions = [s.condition for s in data_model.steps]
        self.transforms = [s.transform for s in data_model.steps]

    def has_sufficient_experiments(self) -> bool:
        return True

    def _get_step(self) -> Tuple[Strategy, Optional[transforms.Transform]]:
        """Returns index of the current step, the step itself"""
        for i, condition in enumerate(self.conditions):
            if condition.evaluate(self.domain, experiments=self.experiments):
                return self.stratgies[i], self.transforms[i]
        raise ValueError("No condition could be satisfied.")

    def _ask(self, candidate_count: Optional[PositiveInt]) -> pd.DataFrame:
        strategy, transform = self._get_step()

        candidate_count = candidate_count or 1

        # handle a possible transform
        tf_domain = _apply_tf(self.domain, transform, "domain")
        transformed_domain = tf_domain or self.domain
        strategy.domain = transformed_domain
        tf_exp = _apply_tf(self.experiments, transform, "experiments")
        transformed_experiments = self.experiments if tf_exp is None else tf_exp
        tf_cand = _apply_tf(self.candidates, transform, "candidates")
        transformed_candidates = self.candidates if tf_cand is None else tf_cand
        # tell the experiments
        if transformed_experiments is not None and self.num_experiments > 0:
            strategy.tell(experiments=transformed_experiments, replace=True)
        # tell pending
        if transformed_candidates is not None and len(transformed_candidates) > 0:
            strategy.set_candidates(transformed_candidates)
        # ask and return
        return strategy.ask(candidate_count=candidate_count)
