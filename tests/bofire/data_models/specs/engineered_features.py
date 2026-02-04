from bofire.data_models.domain.api import EngineeredFeatures
from bofire.data_models.features.api import MeanFeature, SumFeature
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    EngineeredFeatures,
    lambda: {
        "features": [
            SumFeature(key="sum1", features=["a", "b"]).model_dump(),
            MeanFeature(key="mean1", features=["a", "b"]).model_dump(),
        ],
    },
)
