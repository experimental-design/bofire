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
from bofire.utils.doe import (
    apply_block_generator,
    fracfact,
    get_block_generator,
    get_generator,
)


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
        # blocking
        self.block_feature_key = data_model.block_feature_key
        if self.block_feature_key is not None:
            block_feature = self.domain.inputs.get_by_key(self.block_feature_key)
            self.n_blocks = (
                len(block_feature.get_allowed_categories())
                if isinstance(block_feature, CategoricalInput)
                else len(block_feature.values)  # type: ignore
            )
        else:
            self.n_blocks = 1

    def _get_continuous_design(self) -> pd.DataFrame:
        continuous_inputs = self.domain.inputs.get(ContinuousInput)
        gen = self.generator or get_generator(
            n_factors=len(continuous_inputs),
            n_generators=self.n_generators,
        )
        design = pd.DataFrame(fracfact(gen=gen), columns=continuous_inputs.get_keys())

        if self.n_blocks > 1 and self.n_repetitions % self.n_blocks != 0:
            block_generator = get_block_generator(
                n_factors=len(continuous_inputs),
                n_blocks=self.n_blocks,
                n_repetitions=self.n_repetitions,
                n_generators=self.n_generators,
            )
            design["block"] = apply_block_generator(
                design=design.to_numpy(), gen=block_generator
            )
        else:
            design["block"] = 0

        # setup the repetitions
        if self.n_repetitions > 1:
            if self.n_blocks > 1 and design["block"].max() + 1 != self.n_blocks:
                designs = []
                for i in range(self.n_repetitions):
                    d = design.copy()
                    d.block += i
                    designs.append(d)
            else:
                designs = [design] * (self.n_repetitions)
            design = pd.concat(designs, ignore_index=True)

        # setup the center points
        centers = pd.DataFrame(
            {key: [0] * self.n_center for key in continuous_inputs.get_keys()},
        )
        centers["block"] = 0

        all_centers = []  # including blocking

        for i in range(self.n_blocks):
            c = centers.copy()
            c.block += i
            all_centers.append(c)

        design = pd.concat([design, *all_centers], ignore_index=True)
        # scale the design to 0 and 1
        design[continuous_inputs.get_keys()] = (
            design[continuous_inputs.get_keys()] + 1.0
        ) / 2.0
        # scale to correct bounds
        lower, upper = continuous_inputs.get_bounds(specs={})
        lower, upper = np.array(lower), np.array(upper)
        design[continuous_inputs.get_keys()] = design[continuous_inputs.get_keys()] * (
            upper - lower
        ).reshape(1, -1) + lower.reshape(1, -1)
        if self.block_feature_key is not None:
            block_feature = self.domain.inputs.get_by_key(self.block_feature_key)
            block_vals = (
                block_feature.get_allowed_categories()
                if isinstance(block_feature, CategoricalInput)
                else block_feature.values  # type: ignore
            )
            design["block"] = design["block"].map(dict(enumerate(block_vals)))
            design = design.rename(columns={"block": self.block_feature_key})
            return design
        return design[continuous_inputs.get_keys()]

    def _get_categorical_design(self) -> pd.DataFrame:
        categorical_keys = [
            key
            for key in self.domain.inputs.get_keys([CategoricalInput, DiscreteInput])
            if key != self.block_feature_key
        ]
        categorical_inputs = self.domain.inputs.get_by_keys(categorical_keys)
        return pd.DataFrame(
            [
                {e[0]: e[1] for e in combi}
                for combi in categorical_inputs.get_categorical_combinations()
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
        ).sort_values(
            by=self.domain.inputs.get_keys([CategoricalInput, DiscreteInput]),
            ignore_index=True,
        )
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
