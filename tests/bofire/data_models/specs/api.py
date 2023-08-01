from tests.bofire.data_models.specs.acquisition_functions import (
    specs as acquisition_functions,
)
from tests.bofire.data_models.specs.constraints import specs as constraints
from tests.bofire.data_models.specs.domain import specs as domain
from tests.bofire.data_models.specs.features import specs as features
from tests.bofire.data_models.specs.objectives import specs as objectives
from tests.bofire.data_models.specs.specs import Spec, Specs

try:
    # in case of the minimal installation these import are not available
    from tests.bofire.data_models.specs.conditions import specs as conditions
    from tests.bofire.data_models.specs.kernels import specs as kernels
    from tests.bofire.data_models.specs.molfeatures import (
        specs as molfeatures,
    )
    from tests.bofire.data_models.specs.outlier_detection import (
        specs as outlier_detection,
    )
    from tests.bofire.data_models.specs.priors import specs as priors
    from tests.bofire.data_models.specs.strategies import specs as strategies
    from tests.bofire.data_models.specs.surrogates import specs as surrogates
except ImportError:
    pass
