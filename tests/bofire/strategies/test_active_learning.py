import pandas as pd

import bofire.strategies.api as strategies
from bofire.benchmarks.multi import DTLZ2
from bofire.data_models.acquisition_functions.api import qNegIntPosVar
from bofire.data_models.domain.api import Outputs
from bofire.data_models.strategies.api import ActiveLearningStrategy
from bofire.data_models.surrogates.api import BotorchSurrogates, SingleTaskGPSurrogate


def test_active_learning():
    """Tests the initialization of the ActiveLearningStrategy.
    This is done for the most complicated case meaning a multi-objective scenario with
    a unique weight for each output feature.
    """
    benchmark = DTLZ2(dim=3)
    output_keys = benchmark.domain.outputs.get_keys()
    weights = {
        output_keys[0]: 0.3,
        output_keys[1]: 0.7,
    }
    aqcf_data_model = qNegIntPosVar(weights=weights)
    data_model = ActiveLearningStrategy(
        domain=benchmark.domain,
        surrogate_specs=BotorchSurrogates(
            surrogates=[
                SingleTaskGPSurrogate(
                    inputs=benchmark.domain.inputs,
                    outputs=Outputs(features=[benchmark.domain.outputs[0]]),
                ),
                SingleTaskGPSurrogate(
                    inputs=benchmark.domain.inputs,
                    outputs=Outputs(features=[benchmark.domain.outputs[1]]),
                ),
            ],
        ),
        acquisition_function=aqcf_data_model,
    )
    initial_points = benchmark.domain.inputs.sample(10)
    initial_experiments = pd.concat(
        [initial_points, benchmark.f(initial_points)],
        axis=1,
    )
    recommender = strategies.map(data_model=data_model)
    recommender.tell(initial_experiments)  # Check whether the model can be trained.
    acqf = recommender._get_acqfs(1)[
        0
    ]  # Check if an instance of the acqf can be created.
    weight_list = []
    [
        weight_list.append(aqcf_data_model.weights.get(key))
        for key in benchmark.domain.outputs.get_keys()
    ]
    assert (
        weight_list == acqf.posterior_transform.weights.tolist()
    )  # Check whether the weights in the posterior_transfrom are set up correctly.
    _ = recommender.ask(2)  # Check whether the optimization of the acqf works.
