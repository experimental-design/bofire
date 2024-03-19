from typing import Literal, Optional, Tuple, TypeVar, Union

import pandas as pd
from pydantic import PositiveInt

import bofire.transforms.api as transforms
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import StepwiseStrategy as data_model
from bofire.strategies.mapper_actual import map as map_actual
from bofire.strategies.strategy import Strategy
from bofire.transforms.transform import Transform

T = TypeVar("T", pd.DataFrame, Domain)


TfData = Union[Literal["experiments"], Literal["candidates"], Literal["domain"]]


def _apply_tf(
    data: Optional[T],
    transform: Optional[Transform],
    tf: TfData,
) -> Optional[T]:
    if data is not None and transform is not None:
        return getattr(transform, f"transform_{tf}")(data)


class StepwiseStrategy(Strategy):
    def __init__(self, data_model: data_model, **kwargs):
        super().__init__(data_model, **kwargs)
        self.strategies = [map_actual(s.strategy_data) for s in data_model.steps]
        self.conditions = [s.condition for s in data_model.steps]
        self.transforms = [
            s.transform and transforms.map(s.transform) for s in data_model.steps
        ]

    def has_sufficient_experiments(self) -> bool:
        return True

    def get_step(self) -> Tuple[Strategy, Optional[Transform]]:
        """Returns the strategy at the current step and the corresponding transform if given."""
        for i, condition in enumerate(self.conditions):
            if condition.evaluate(self.domain, experiments=self.experiments):
                return self.strategies[i], self.transforms[i]
        raise ValueError("No condition could be satisfied.")

    def _ask(self, candidate_count: Optional[PositiveInt]) -> pd.DataFrame:
        strategy, transform = self.get_step()

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
        if transformed_experiments is not None and len(transformed_experiments) > 0:
            strategy.tell(experiments=transformed_experiments, replace=True)
        # tell pending
        if transformed_candidates is not None and len(transformed_candidates) > 0:
            strategy.set_candidates(transformed_candidates)
        # ask and return
        candidates = strategy.ask(candidate_count=candidate_count)
        if transform is not None:
            return transform.untransform_candidates(candidates)
        else:
            return candidates
