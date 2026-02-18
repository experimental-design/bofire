import random
import uuid

import bofire.data_models.features.api as features
from bofire.data_models.molfeatures.api import MordredDescriptors
from bofire.data_models.objectives.api import (
    ConstrainedCategoricalObjective,
    MaximizeObjective,
)
from tests.bofire.data_models.specs.objectives import specs as objectives
from tests.bofire.data_models.specs.specs import Specs


# RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

specs = Specs([])


specs.add_valid(
    features.SumFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["a", "b", "c"],
        "keep_features": True,
    },
)


specs.add_valid(
    features.ProductFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["a", "b", "c"],
        "keep_features": True,
    },
)

specs.add_valid(
    features.ProductFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["a", "a"],
        "keep_features": True,
    },
)

specs.add_valid(
    features.MeanFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["a", "b", "c"],
        "keep_features": False,
    },
)

specs.add_valid(
    features.WeightedSumFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["a", "b", "c"],
        "descriptors": ["alpha", "beta"],
        "keep_features": True,
    },
)

specs.add_valid(
    features.MolecularWeightedSumFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["a", "b", "c"],
        "molfeatures": MordredDescriptors(
            descriptors=["NssCH2", "ATSC2d"],
            ignore_3D=True,
        ).model_dump(),
        "keep_features": True,
    },
)

specs.add_valid(
    features.InterpolateFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["x1", "x2", "y1", "y2"],
        "x_keys": ["x1", "x2"],
        "y_keys": ["y1", "y2"],
        "n_interpolation_points": 20,
        "interpolation_range": [0.0, 60.0],
        "keep_features": True,
        "prepend_x": [],
        "append_x": [],
        "prepend_y": [],
        "append_y": [],
        "normalize_y": 1.0,
        "normalize_x": False,
    },
)

specs.add_valid(
    features.InterpolateFeature,
    lambda: {
        "key": str(uuid.uuid4()),
        "features": ["x1", "x2", "y1", "y2"],
        "x_keys": ["x1", "x2"],
        "y_keys": ["y1", "y2"],
        "n_interpolation_points": 20,
        "interpolation_range": [0.0, 1.0],
        "keep_features": True,
        "prepend_x": [0.0],
        "append_x": [],
        "prepend_y": [0.0],
        "append_y": [],
        "normalize_y": 2.0,
        "normalize_x": True,
    },
)

specs.add_invalid(
    features.InterpolateFeature,
    lambda: {
        "key": "interp1",
        "features": ["x1", "y1"],
        "interpolation_range": [0.0, 60.0],
        "x_keys": ["x1"],
        "y_keys": ["x1"],
        "n_interpolation_points": 20,
    },
    error=ValueError,
    message=r"x_keys and y_keys must not overlap\.",
)

specs.add_invalid(
    features.InterpolateFeature,
    lambda: {
        "key": "interp1",
        "features": ["x1", "x2", "y1"],
        "interpolation_range": [0.0, 60.0],
        "x_keys": ["x1"],
        "y_keys": ["y1"],
        "n_interpolation_points": 20,
    },
    error=ValueError,
    message=r"features must match x_keys \+ y_keys\.",
)

specs.add_invalid(
    features.InterpolateFeature,
    lambda: {
        "key": "interp1",
        "features": ["x1", "x2", "y1", "y2"],
        "x_keys": ["x1", "x2"],
        "y_keys": ["y1", "y2"],
        "n_interpolation_points": 20,
        "interpolation_range": [0.0, 60.0],
        "prepend_x": [0.0],
    },
    error=ValueError,
    message=r"Total number of x and y values must be equal\.",
)

specs.add_invalid(
    features.InterpolateFeature,
    lambda: {
        "key": "interp1",
        "features": ["x1", "x2", "y1", "y2"],
        "x_keys": ["x1", "x2"],
        "y_keys": ["y1", "y2"],
        "n_interpolation_points": 20,
        "interpolation_range": [0.0, 60.0],
        "normalize_x": True,
    },
    error=ValueError,
    message=r"When normalize_x is True, interpolation_range must be \(0, 1\)",
)

specs.add_valid(
    features.CloneFeature,
    lambda: {"key": str(uuid.uuid4()), "features": ["a", "b"], "keep_features": True},
)

specs.add_valid(
    features.CloneFeature,
    lambda: {"key": str(uuid.uuid4()), "features": ["a"], "keep_features": True},
)

specs.add_valid(
    features.DiscreteInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "values": [random.random(), random.random() + 3],
        "unit": random.choice(["°C", "mg", "mmol/l", None]),
        "rtol": 1e-7,
    },
)

specs.add_invalid(
    features.DiscreteInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "values": [1.0],
        "unit": random.choice(["°C", "mg", "mmol/l", None]),
        "rtol": 1e-7,
    },
    error=ValueError,
    message="Fixed discrete inputs are not supported. Please use a fixed continuous input.",
)


specs.add_valid(
    features.ContinuousInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "bounds": [3, 5.3],
        "unit": random.choice(["°C", "mg", "mmol/l", None]),
        "local_relative_bounds": None,
        "stepsize": None,
        "allow_zero": False,
    },
)

specs.add_invalid(
    features.ContinuousInput,
    lambda: {"key": "a", "bounds": [5, 3]},
    error=ValueError,
    message="Sequence is not monotonically increasing.",
)

specs.add_invalid(
    features.ContinuousInput,
    lambda: {"key": "a", "bounds": [-1, 5], "allow_zero": True},
    error=ValueError,
    message="If `allow_zero==True`, then zero must not lie within the bounds.",
)

specs.add_valid(
    features.ContinuousDescriptorInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "bounds": [3, 5.3],
        "descriptors": ["d1", "d2"],
        "values": [1.0, 2.0],
        "unit": random.choice(["°C", "mg", "mmol/l", None]),
        "local_relative_bounds": None,
        "stepsize": None,
        "allow_zero": False,
    },
)
specs.add_valid(
    features.CategoricalInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": ["c1", "c2", "c3"],
        "allowed": [True, True, False],
    },
)

specs.add_invalid(
    features.CategoricalInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": ["c1", "c2", "c2"],
        "allowed": [True, True, False],
    },
    error=ValueError,
    message="Categories must be unique",
)

specs.add_invalid(
    features.CategoricalInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": ["c1", "c2", "c3"],
        "allowed": [True, True],
    },
    error=ValueError,
    message="allowed must have same length as categories",
)

specs.add_invalid(
    features.CategoricalInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": ["c1", "c2", "c3"],
        "allowed": [False, False, False],
    },
    error=ValueError,
    message="no category is allowed",
)


specs.add_valid(
    features.CategoricalDescriptorInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": ["c1", "c2", "c3"],
        "allowed": [True, True, False],
        "descriptors": ["d1", "d2"],
        "values": [
            [1.0, 2.0],
            [3.0, 7.0],
            [5.0, 1.0],
        ],
    },
)
specs.add_valid(
    features.ContinuousOutput,
    lambda: {
        "key": str(uuid.uuid4()),
        "objective": objectives.valid(MaximizeObjective).typed_spec(),
        "unit": random.choice(["%", "area %", None]),
    },
)

specs.add_valid(
    features.CategoricalOutput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": ["a", "b", "c"],
        "objective": ConstrainedCategoricalObjective(
            categories=["a", "b", "c"],
            desirability=[True, True, False],
        ).model_dump(),
    },
)


specs.add_valid(
    features.CategoricalMolecularInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": [
            "CC(=O)Oc1ccccc1C(=O)O",
            "c1ccccc1",
            "[CH3][CH2][OH]",
            "N[C@](C)(F)C(=O)O",
        ],
        "allowed": [True, True, True, True],
    },
)


specs.add_valid(
    features.ContinuousMolecularInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "molecule": "CC",
        "bounds": [0.0, 1.0],
        "allow_zero": False,
        "unit": random.choice(["°C", "mg", "mmol/l", None]),
        "local_relative_bounds": None,
        "stepsize": None,
    },
)


specs.add_valid(
    features.TaskInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": [
            "process_1",
            "process_2",
            "process_3",
        ],
        "allowed": [True, True, True],
        "fidelities": [0, 1, 2],
    },
)

specs.add_invalid(
    features.TaskInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": [
            "process_1",
            "process_2",
            "process_3",
        ],
        "allowed": [True, True, True],
        "fidelities": [0, 1],
    },
    error=ValueError,
    message="Length of fidelity lists must be equal to the number of tasks",
)

specs.add_invalid(
    features.TaskInput,
    lambda: {
        "key": str(uuid.uuid4()),
        "categories": [
            "process_1",
            "process_2",
            "process_3",
        ],
        "allowed": [True, True, True],
        "fidelities": [0, 1, 3],
    },
    error=ValueError,
    message="Fidelities must be a list containing integers, starting from 0 and increasing by 1",
)
