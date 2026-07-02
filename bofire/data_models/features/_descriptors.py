"""Shared per-feature descriptor table (categorical / continuous / discrete)."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.encodings.reserved import get_reserved_descriptor, is_reserved


class DescriptorsMixin(BaseModel):
    """Mixin giving a feature a per-level ``descriptors`` table.

    A "level" is a category (categorical), a discrete value (discrete), or the
    single component (continuous). Each column of ``descriptors`` has one entry per
    level. Reserved keys (e.g. ``smiles``) are validated + role-shielded by the
    reserved-descriptor registry; all other columns are numeric descriptors.

    Subclasses must implement :meth:`descriptor_levels`.
    """

    if TYPE_CHECKING:
        # always mixed into a ``Feature`` (which supplies ``key``); declared here
        # only so the descriptor helpers and sources can reference ``self.key``.
        key: str

    descriptors: Dict[str, List[Any]] = Field(default_factory=dict)

    def descriptor_levels(self) -> List:
        """The row labels of the descriptor table (categories / values / [key])."""
        raise NotImplementedError

    @field_validator("descriptors")
    @classmethod
    def _coerce_descriptors(cls, descriptors):
        """Per-column dtype validation (reserved dtype / numeric). Length is checked
        against the levels in the model validator below."""
        validated: Dict[str, List[Any]] = {}
        for name, column in descriptors.items():
            if is_reserved(name):
                reserved = get_reserved_descriptor(name)
                coerced = [reserved.dtype(v) for v in column]
                reserved.validator(coerced)
                validated[name] = coerced
            else:
                try:
                    validated[name] = [float(v) for v in column]
                except (TypeError, ValueError):
                    raise ValueError(
                        f"descriptor column '{name}' must be numeric",
                    )
        return validated

    @model_validator(mode="after")
    def _validate_descriptor_lengths(self):
        n = len(self.descriptor_levels())
        for name, column in self.descriptors.items():
            if len(column) != n:
                raise ValueError(
                    f"descriptor column '{name}' must have {n} value(s) "
                    "(one per level)",
                )
        return self

    def descriptor_columns(self, role: Optional[str] = None) -> List[str]:
        """Descriptor column names, optionally filtered by role.

        Non-reserved columns have role ``"descriptor"``; reserved columns carry
        their registered role (e.g. ``smiles`` is ``"structure"``).
        """
        columns = list(self.descriptors.keys())
        if role is None:
            return columns
        return [
            c
            for c in columns
            if (get_reserved_descriptor(c).role if is_reserved(c) else "descriptor")
            == role
        ]

    def descriptor_table(self, columns: List[str]) -> pd.DataFrame:
        """Per-level table (rows = levels, columns = selected descriptors)."""
        return pd.DataFrame(
            {c: self.descriptors[c] for c in columns},
            index=self.descriptor_levels(),
        )
