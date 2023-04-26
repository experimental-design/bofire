from tests.bofire.data_models.specs.acquisition_functions import (  # noqa: F401
    specs as acquisition_functions,
)
from tests.bofire.data_models.specs.constraints import (  # noqa: F401
    specs as constraints,
)
from tests.bofire.data_models.specs.domain import specs as domain  # noqa: F401
from tests.bofire.data_models.specs.features import specs as features  # noqa: F401
from tests.bofire.data_models.specs.objectives import specs as objectives  # noqa: F401
from tests.bofire.data_models.specs.specs import Spec, Specs  # noqa: F401

try:
    # in case of the minimal installation these import are not available
    from tests.bofire.data_models.specs.kernels import specs as kernels  # noqa: F401
    from tests.bofire.data_models.specs.priors import specs as priors  # noqa: F401
    from tests.bofire.data_models.specs.strategies import (  # noqa: F401
        specs as strategies,
    )
    from tests.bofire.data_models.specs.surrogates import (  # noqa: F401
        specs as surrogates,
    )
except ImportError:
    pass
