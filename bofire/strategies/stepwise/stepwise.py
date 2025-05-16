from typing import List, Literal, Optional, Tuple, TypeVar

import pandas as pd
from pydantic import PositiveInt

import bofire.transforms.api as transforms
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import StepwiseStrategy as data_model
from bofire.data_models.strategies.stepwise.stepwise import Step
from bofire.data_models.surrogates.api import BotorchSurrogates as BotorchSurrogateSpecs
from bofire.strategies.data_models.candidate import Candidate
from bofire.strategies.mapper_actual import map as map_actual
from bofire.strategies.strategy import Strategy, make_strategy
from bofire.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.transforms.transform import Transform


T = TypeVar("T", pd.DataFrame, Domain)


TfData = Literal["experiments", "candidates", "domain"]


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

    def _ask(self, candidate_count: Optional[PositiveInt]) -> pd.DataFrame:  # type: ignore
        strategy, transform = self.get_step()

        candidate_count = candidate_count or 1

        # handle a possible transform, no need to apply transforms to domains, as domains
        # do not have to be exactly the same for each step, they only have to be compatible
        # to the master domain of the stepwise strategy
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
        return candidates

    def to_candidates(self, candidates: pd.DataFrame) -> List[Candidate]:
        strategy, _ = self.get_step()

        candiate_list = strategy.to_candidates(candidates=candidates)

        return candiate_list

    @property
    def surrogates_specs(self) -> BotorchSurrogateSpecs:
        strategy, _ = self.get_step()
        try:
            specs = strategy.surrogate_specs  # type: ignore
        except AttributeError:
            raise ValueError("Current Step do not possess any surrogates.")
        return specs

    @property
    def surrogates(self) -> BotorchSurrogates:
        strategy, _ = self.get_step()
        try:
            surrogates = strategy.surrogates  # type: ignore
        except AttributeError:
            raise ValueError("Current Step do not possess any surrogates.")
        return surrogates

    @classmethod
    def make(
        cls, domain: Domain, steps: List[Step] | None = None, seed: int | None = None
    ):
        """
        Create a StepwiseStrategy from a list of steps. Each step is a strategy the runs until a
        condition is satisfied.  One example of a stepwise strategy is
        is to start with a few random samples to gather initial data for subsequent Bayesian optimization.
        An example steps-list for two random experiments followed by a SoboStrategy is:

        ```python
        from bofire.data_models.strategies.api import (
            Step,
            RandomStrategy,
            SoboStrategy,
            NumberOfExperimentsCondition,
            AlwaysTrueCondition
        )
        from bofire.stratgies.api import StepwiseStrategy
        steps = [
            Step(
                strategy_data=RandomStrategy(domain=domain),
                condition=NumberOfExperimentsCondition(n_experiments=2),
            ),
            Step(
                strategy_data=SoboStrategy(domain=domain), condition=AlwaysTrueCondition()
            )
        ]
        stepwise_strategy = StepwiseStrategy.make(domain, steps)
        ```

        All passed domains need to compatible, i.e.,

        - they have the same number of features,
        - the same feature keys and
        - the features with the same key have the same type and categories.
        - The bounds and allowed categories of the features can vary.

        Further, the data and domain are passed to the next step. They can also
        be transformed before being passed to the next step.
        Args:
            steps: List of steps to be used in the strategy.
            seed: Seed for random number generation.
        """
        return make_strategy(cls, data_model, locals())
