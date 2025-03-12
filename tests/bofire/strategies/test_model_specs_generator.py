import pytest

import bofire.data_models.strategies.api as data_models
import bofire.data_models.surrogates.api as surrogate_data_models
from bofire.benchmarks.api import DTLZ2
from bofire.data_models.domain.api import Outputs
from bofire.data_models.features.api import ContinuousOutput


def test_generate_surrogate_specs():
    bench = DTLZ2(dim=6)
    domain = bench.domain
    # first with no specified surrogate
    stategy_data = data_models.MoboStrategy(domain=domain)
    assert len(stategy_data.surrogate_specs.surrogates) == 2
    # then with a specified surrogate
    stategy_data = data_models.MoboStrategy(
        domain=domain,
        surrogate_specs=surrogate_data_models.BotorchSurrogates(
            surrogates=[
                surrogate_data_models.SingleTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=domain.outputs.get_by_keys(["f_1"]),
                )
            ]
        ),
    )
    assert len(stategy_data.surrogate_specs.surrogates) == 2


def test_generate_surrogate_specs_invalid():
    bench = DTLZ2(dim=6)
    domain = bench.domain
    with pytest.raises(ValueError, match="Output features do not match."):
        (
            data_models.MoboStrategy(
                domain=domain,
                surrogate_specs=surrogate_data_models.BotorchSurrogates(
                    surrogates=[
                        surrogate_data_models.SingleTaskGPSurrogate(
                            inputs=domain.inputs,
                            outputs=Outputs(features=[ContinuousOutput(key="f_11")]),
                        )
                    ]
                ),
            ),
        )
