from typing import Optional

import numpy as np
import pandas as pd

from bofire.data_models.strategies.api import FractionalFactorialStrategy as DataModel
from bofire.strategies.strategy import Strategy
from bofire.utils.doe import fracfact, get_generator


class FractionalFactorialStrategy(Strategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.n_repetitions = data_model.n_repetitions
        self.n_center = data_model.n_center
        self.n_generators = data_model.n_generators
        self.generator = data_model.generator

    def _ask(self, candidate_count: Optional[int] = None) -> pd.DataFrame:
        if candidate_count is not None:
            raise ValueError(
                "FractionalFactorialStrategy will ignore the specified value of candidate_count. "
                "The strategy automatically determines how many candidates to "
                "propose.",
            )
        gen = self.generator or get_generator(
            n_factors=len(self.domain.inputs),
            n_generators=self.n_generators,
        )
        design = pd.DataFrame(fracfact(gen=gen), columns=self.domain.inputs.get_keys())
        # setup the repetitions
        if self.n_repetitions > 1:
            design = pd.concat([design] * (self.n_repetitions), ignore_index=True)
        # setup the center points
        centers = pd.DataFrame(
            {key: [0] * self.n_center for key in self.domain.inputs.get_keys()},
        )
        design = pd.concat([design, centers], ignore_index=True)
        # scale the design to 0 and 1
        design = (design + 1.0) / 2.0
        # scale to correct bounds
        lower, upper = self.domain.inputs.get_bounds(specs={})
        lower, upper = np.array(lower), np.array(upper)
        design = design * (upper - lower).reshape(1, -1) + lower.reshape(1, -1)
        return design

    def has_sufficient_experiments(self) -> bool:
        return True
