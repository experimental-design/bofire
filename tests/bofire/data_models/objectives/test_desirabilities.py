from typing import List, Tuple

import pytest
import pydantic

from bofire.data_models.objectives.desirabilities import (
    DesirabilityObjective,
    IncreasingDesirabilityObjective,
    DecreasingDesirabilityObjective,
    PeakDesirabilityObjective,
    )

valid_specs: List[Tuple[callable, dict]] = []
invalid_specs: List[Tuple[callable, dict]] = []

valid_specs.append((
    DesirabilityObjective,
    {},
))
valid_specs.append((
    DesirabilityObjective,
    {"bounds": (0, 10.)},
))

for obj in [IncreasingDesirabilityObjective, DecreasingDesirabilityObjective, PeakDesirabilityObjective]:
    valid_specs.append((
        obj,
        {},
    ))
    valid_specs.append((
        obj,
        {"bounds": (0, 10.), "log_shape_factor": 1.},
    ))
    valid_specs.append((
        obj,
        {"bounds": (0, 10.), "log_shape_factor": -1., "clip": True},
    ))
    invalid_specs.append((
        obj,
        {"bounds": (0, 10.), "log_shape_factor": -1., "clip": False},
    ))


valid_specs.append((
    PeakDesirabilityObjective,
    {"bounds": (0, 10.), "peak_position": 4.},
))
valid_specs.append((
    PeakDesirabilityObjective,
    {"bounds": (0, 10.), "peak_position": 4., "log_shape_factor": 1.},
))
invalid_specs.append((
    PeakDesirabilityObjective,
    {"bounds": (0, 10.), "peak_position": 15.},
))
invalid_specs.append((
    PeakDesirabilityObjective,
    {"bounds": (0, 10.), "peak_position": -1.},
))

def test_desirabilities():

    for spec in valid_specs:
        obj = spec[0](**spec[1])

    with pytest.raises(pydantic.ValidationError):
        for spec in invalid_specs:
            obj = spec[0](**spec[1])
