"""Deprecated alias for :mod:`bofire.data_models.descriptor_generators.api`.

``MolFeatures`` was renamed to ``DescriptorGenerator`` (it generates descriptors from a
structure column; it is not a feature and not intrinsically molecular). This shim keeps
the old import path and names working for one release.
"""

import warnings

from bofire.data_models.descriptor_generators.api import (
    AnyDescriptorGenerator as AnyMolFeatures,
)
from bofire.data_models.descriptor_generators.api import (
    DescriptorGenerator as MolFeatures,
)
from bofire.data_models.descriptor_generators.api import (
    Fingerprints,
    Fragments,
    MordredDescriptors,
)


warnings.warn(
    "`bofire.data_models.molfeatures` is deprecated; use "
    "`bofire.data_models.descriptor_generators` (`MolFeatures` -> "
    "`DescriptorGenerator`, `AnyMolFeatures` -> `AnyDescriptorGenerator`).",
    DeprecationWarning,
    stacklevel=2,
)
