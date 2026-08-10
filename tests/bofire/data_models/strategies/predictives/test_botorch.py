import pytest

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.strategies.api import BotorchOptimizer, SoboStrategy
from bofire.data_models.surrogates.api import (
    MixedSingleTaskGPSurrogate,
    SingleTaskGPSurrogate,
)


def test_botorch_strategy():
    domain = Domain(
        inputs=[ContinuousInput(key="x", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    sobo = SoboStrategy(
        domain=domain,
        acquisition_optimizer=BotorchOptimizer(),
    )
    assert isinstance(sobo.acquisition_optimizer, BotorchOptimizer)


@pytest.mark.parametrize(
    "categorical, expected",
    [
        # a plain categorical is encoded as levels and needs the categorical kernel
        (
            CategoricalInput(key="c", categories=["a", "b"]),
            MixedSingleTaskGPSurrogate,
        ),
        # descriptor data makes it continuous columns, so a plain GP suffices. This is
        # the canonical form of the deprecated `CategoricalDescriptorInput`; selecting
        # on the type alone used to miss it.
        (
            CategoricalInput(
                key="c", categories=["a", "b"], descriptors={"d": [1.0, 2.0]}
            ),
            SingleTaskGPSurrogate,
        ),
        (
            CategoricalInput(
                key="c", categories=["CCO", "CC"], structure=["CCO", "CC"]
            ),
            SingleTaskGPSurrogate,
        ),
    ],
    ids=["plain", "descriptors", "structure"],
)
def test_default_surrogate_selection_is_data_driven(categorical, expected):
    domain = Domain(
        inputs=[ContinuousInput(key="x", bounds=(0, 1)), categorical],
        outputs=[ContinuousOutput(key="y")],
    )
    spec = SoboStrategy(domain=domain).surrogate_specs.surrogates[0]
    assert isinstance(spec, expected)
