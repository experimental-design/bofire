import warnings
from typing import Optional

import numpy as np
import pandas as pd

from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
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
        self.randomize_runoder = data_model.randomize_runorder

    def _get_continuous_design(self) -> pd.DataFrame:
        continuous_inputs = self.domain.inputs.get(ContinuousInput)
        gen = self.generator or get_generator(
            n_factors=len(continuous_inputs),
            n_generators=self.n_generators,
        )
        design = pd.DataFrame(fracfact(gen=gen), columns=continuous_inputs.get_keys())
        # setup the repetitions
        if self.n_repetitions > 1:
            design = pd.concat([design] * (self.n_repetitions), ignore_index=True)
        # setup the center points
        centers = pd.DataFrame(
            {key: [0] * self.n_center for key in continuous_inputs.get_keys()},
        )
        design = pd.concat([design, centers], ignore_index=True)
        # scale the design to 0 and 1
        design = (design + 1.0) / 2.0
        # scale to correct bounds
        lower, upper = continuous_inputs.get_bounds(specs={})
        lower, upper = np.array(lower), np.array(upper)
        design = design * (upper - lower).reshape(1, -1) + lower.reshape(1, -1)
        return design

    def _get_categorical_design(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {e[0]: e[1] for e in combi}
                for combi in self.domain.inputs.get_categorical_combinations()
            ],
        )

    def _ask(self, candidate_count: Optional[int] = None) -> pd.DataFrame:
        if candidate_count is not None:
            warnings.warn(
                "FractionalFactorialStrategy will ignore the specified value of candidate_count. "
                "The strategy automatically determines how many candidates to "
                "propose.",
                UserWarning,
            )
        design = None
        if len(self.domain.inputs.get(ContinuousInput)) > 0:
            design = self._get_continuous_design()
            if len(self.domain.inputs.get(ContinuousInput)) == len(self.domain.inputs):
                return self.randomize_design(design)

        categorical_design = self._get_categorical_design()
        if len(self.domain.inputs.get([CategoricalInput, DiscreteInput])) == len(
            self.domain.inputs
        ):
            return self.randomize_design(categorical_design)

        assert isinstance(design, pd.DataFrame)
        # combine the two designs
        design = pd.concat(
            [
                pd.concat([design] * len(categorical_design), ignore_index=True),
                pd.concat([categorical_design] * len(design), ignore_index=True),  # type: ignore
            ],
            axis=1,
        ).sort_values(by=self.domain.inputs.get_keys([CategoricalInput, DiscreteInput]))
        return self.randomize_design(design)

    def randomize_design(self, design: pd.DataFrame) -> pd.DataFrame:
        """Randomize the run order of the design if `self.randomize_runorder` is True."""
        return (
            design.sample(frac=1, random_state=self._get_seed()).reset_index(drop=True)
            if self.randomize_runoder
            else design
        )

    def has_sufficient_experiments(self) -> bool:
        return True
