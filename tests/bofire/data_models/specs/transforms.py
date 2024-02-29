from pydantic import ValidationError

from bofire.data_models.strategies.api import DropDataTransform
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    DropDataTransform,
    lambda: {
        "to_be_removed_experiments": [1, 2, 3],
        "to_be_removed_candidates": [4, 5, 6],
    },
)

specs.add_valid(
    DropDataTransform,
    lambda: {},
)
specs.add_valid(
    DropDataTransform,
    lambda: {"to_be_removed_candidates": [4, 5, 6]},
)
specs.add_valid(
    DropDataTransform,
    lambda: {"to_be_removed_experiments": None, "to_be_removed_candidates": [4, 5, 6]},
)
specs.add_valid(
    DropDataTransform,
    lambda: {"to_be_removed_experiments": [1, 2, 3], "to_be_removed_candidates": None},
)
specs.add_invalid(
    DropDataTransform,
    lambda: {"to_be_removed_exp": None, "to_be_removed_cand": None},
    error=ValidationError,
)
