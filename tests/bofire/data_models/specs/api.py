from tests.bofire.data_models.specs.acquisition_functions import (
    specs as acquisition_functions,
)  # noqa: F401
from tests.bofire.data_models.specs.constraints import (
    specs as constraints,
)  # noqa: F401
from tests.bofire.data_models.specs.domain import specs as domain  # noqa: F401
from tests.bofire.data_models.specs.features import specs as features  # noqa: F401
from tests.bofire.data_models.specs.objectives import specs as objectives  # noqa: F401
from tests.bofire.data_models.specs.specs import Spec, Specs  # noqa: F401

try:
    # in case of the minimal installation these import are not available
    from tests.bofire.data_models.specs.kernels import specs as kernels  # noqa: F401
    from tests.bofire.data_models.specs.molfeatures import (
        specs as molfeatures,
    )  # noqa: F401
    from tests.bofire.data_models.specs.priors import specs as priors  # noqa: F401
    from tests.bofire.data_models.specs.strategies import (
        specs as strategies,
    )  # noqa: F401
    from tests.bofire.data_models.specs.surrogates import (
        specs as surrogates,
    )  # noqa: F401
except ImportError:
    pass
