from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from bofire.benchmarks import api as benchmarks
from bofire.data_models import api as domain
from bofire.data_models.features.api import ContinuousInput, DiscreteInput
from bofire.data_models.strategies import api as data_models_strategies
from bofire.strategies import api as strategies
from bofire.strategies.predictives.acqf_optimization import get_optimizer


@pytest.fixture(
    params=[  # (benchmark, params, stategy, map_conti_inputs_to_discrete)
        ("Himmelblau", {}, "SoboStrategy", False),
        ("DTLZ2", {"dim": 2, "num_objectives": 2}, "AdditiveSoboStrategy", False),
        (
            "Ackley",
            {"num_categories": 3, "categorical": True, "dim": 4},
            "SoboStrategy",
            False,
        ),
        ("Detergent", {}, "SoboStrategy", False),
        (
            "Ackley",
            {"num_categories": 3, "categorical": True, "dim": 3},
            "SoboStrategy",
            True,
        ),  # this is for testing the "all-categoric" usecase
    ]
)
def benchmark(request) -> Tuple[benchmarks.Benchmark, strategies.PredictiveStrategy]:
    benchmark_name, params, strategy, map_conti_inputs_to_discrete = request.param
    bm = getattr(benchmarks, benchmark_name)(**params)

    if map_conti_inputs_to_discrete:
        # replace a continuous input with a discrete input of the same name, but only 5 possible values
        for idx, ft in enumerate(bm.domain.inputs.features):
            if isinstance(ft, ContinuousInput):
                bm.domain.inputs.features[idx] = DiscreteInput(
                    key=ft.key, values=np.linspace(ft.bounds[0], ft.bounds[1], 5)
                )

    strategy = getattr(data_models_strategies, strategy)(domain=bm.domain)
    return bm, strategy


@pytest.fixture()
def optimization_scope(benchmark) -> Tuple[domain.Domain, dict, pd.DataFrame, list]:
    """ """
    benchmark, strategy_data = benchmark
    domain = benchmark.domain

    strategy = strategies.map(strategy_data)

    experiments = benchmark.f(domain.inputs.sample(10), return_complete=True)
    strategy.tell(experiments=experiments)
    input_preprocessing_specs = strategy.input_preprocessing_specs
    acqfs = strategy._get_acqfs(2)

    return domain, input_preprocessing_specs, experiments, acqfs


def test_optimizer(optimization_scope):
    domain, input_preprocessing_specs, experiments, acqfs = optimization_scope

    optimizer_data_model = data_models_strategies.BotorchOptimizer()
    optimizer = get_optimizer(optimizer_data_model)

    candidates, acqf_vals = optimizer.optimize(
        candidate_count=2,
        acqfs=acqfs,
        domain=domain,
        input_preprocessing_specs=input_preprocessing_specs,
        experiments=experiments,
    )
