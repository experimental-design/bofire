import bofire.data_models.outlier_detection.api as models
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from tests.bofire.data_models.specs.specs import Specs
from tests.bofire.data_models.specs.surrogates import specs as surrogates


specs = Specs([])

specs.add_valid(
    models.IterativeTrimming,
    lambda: {
        "base_gp": surrogates.valid(SingleTaskGPSurrogate).obj().model_dump(),
        "alpha1": 0.5,
        "alpha2": 0.975,
        "nsh": 2,
        "ncc": 2,
        "nrw": 1,
    },
)
