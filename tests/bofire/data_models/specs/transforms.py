from pydantic import ValidationError

from bofire.data_models.transforms.api import DropDataTransform, ManipulateDataTransform
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
    dict,
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

specs.add_valid(
    ManipulateDataTransform,
    lambda: {"experiment_transforms": ["a=b+c"]},
)

specs.add_invalid(
    ManipulateDataTransform,
    lambda: {
        "experiment_transforms": None,
        "candidate_transforms": None,
        "candidate_untransforms": None,
    },
    error=ValueError,
    message="At least one of experiment_transforms, candidate_transforms, or candidate_untransforms must be provided.",
)
