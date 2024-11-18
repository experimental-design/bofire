from typing import cast

import numpy as np
import pandas as pd

import bofire.strategies.api as strategies
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.acquisition_functions.acquisition_function import qNEI
from bofire.data_models.strategies.predictives.sobo import SoboStrategy
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.stepwise.conditions import (
    AlwaysTrueCondition,
    NumberOfExperimentsCondition,
)
from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy
from bofire.data_models.transforms.drop_data import DropDataTransform


def test_dropdata_transform():
    benchmark = Himmelblau()
    d = benchmark.domain
    params = d.inputs.get_keys() + d.outputs.get_keys()

    def test(to_be_removed_rows):
        data_model = StepwiseStrategy(
            domain=benchmark.domain,
            steps=[
                Step(
                    strategy_data=RandomStrategy(domain=benchmark.domain),
                    condition=NumberOfExperimentsCondition(n_experiments=2),
                ),
                Step(
                    strategy_data=SoboStrategy(
                        domain=benchmark.domain,
                        acquisition_function=qNEI(),
                    ),
                    condition=AlwaysTrueCondition(),
                    transform=DropDataTransform(
                        to_be_removed_experiments=to_be_removed_rows,
                    ),
                ),
            ],
        )
        strategy = cast(strategies.StepwiseStrategy, strategies.map(data_model))
        n_samples = 6
        experiments = pd.concat(
            [
                benchmark.domain.inputs.sample(n_samples),
                pd.DataFrame({"y": [1] * n_samples}),
            ],
            axis=1,
        )
        strategy.tell(experiments=experiments)
        strategy.ask()

        last_strategy, _ = strategy.get_step()
        assert last_strategy.experiments is not None and len(
            last_strategy.experiments,
        ) == n_samples - len(to_be_removed_rows)
        kept_rows = [i for i in range(n_samples) if i not in to_be_removed_rows]
        for i, row in enumerate(kept_rows):
            assert np.all(
                last_strategy.experiments[params].values[i]
                == experiments[params].values[row],
            )

    test([0])
    test([0, 1])
    test([1, 3])
