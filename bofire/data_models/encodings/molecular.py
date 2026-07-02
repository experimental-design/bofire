"""Molecular encoding: generate descriptors from a feature's SMILES column."""

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import PrivateAttr

from bofire.data_models.encodings.encoding import CategoricalEncoding
from bofire.data_models.encodings.naming import get_encoded_name
from bofire.data_models.encodings.reserved import get_reserved_descriptor, is_reserved
from bofire.data_models.molfeatures.api import AnyMolFeatures


if TYPE_CHECKING:
    from bofire.data_models.features.categorical import CategoricalInput


class MolecularEncoding(CategoricalEncoding):
    """Encode categories by generating molecular descriptors from a SMILES column.

    The SMILES are *data* on the feature (the ``structure`` reserved column); the
    generator and its hyperparameters (``n_bits``, correlation cutoff, ...) are the
    per-surrogate modelling choice and live here.

    Attributes:
        structure: Name of the reserved structure (SMILES) column on the feature.
        generator: The molecular feature generator (Fingerprints/Fragments/Mordred/Composite).
    """

    type: Literal["MolecularEncoding"] = "MolecularEncoding"
    structure: str = "smiles"
    generator: AnyMolFeatures

    # correlation filtering result cached per SMILES-set, so reusing one encoding
    # across features never leaks feature A's filtered descriptors into feature B.
    _filtered_cache: Dict[tuple, Optional[List[str]]] = PrivateAttr(
        default_factory=dict
    )

    def _structure_column(self, feature: "CategoricalInput") -> List[str]:
        if self.structure not in feature.descriptors:
            raise ValueError(
                f"{feature.key}: structure column '{self.structure}' is not present "
                "in the feature's descriptors.",
            )
        if not (
            is_reserved(self.structure)
            and get_reserved_descriptor(self.structure).role == "structure"
        ):
            raise ValueError(
                f"{feature.key}: column '{self.structure}' is not a structure "
                "identifier and cannot be used for molecular encoding.",
            )
        return [str(s) for s in feature.descriptors[self.structure]]

    def _smiles_map(self, feature: "CategoricalInput") -> dict:
        return dict(zip(feature.categories, self._structure_column(feature)))

    def _prepare(self, feature: "CategoricalInput") -> None:
        """Ensure the generator reflects THIS feature's SMILES set.

        Correlation filtering is cached per SMILES-set: reusing one
        ``MolecularEncoding`` across features with different molecules no longer
        leaks the first feature's filtered descriptor set into the others.
        """
        smiles = tuple(self._structure_column(feature))
        if smiles not in self._filtered_cache:
            # reset any stale filter state, then compute for this SMILES set
            self.generator._descriptors = None
            self.generator.remove_correlated_descriptors(list(smiles))
            self._filtered_cache[smiles] = self.generator._descriptors
        # point the generator at this feature's descriptor set before any use
        self.generator._descriptors = self._filtered_cache[smiles]

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        self._prepare(feature)
        return [
            get_encoded_name(feature.key, d)
            for d in self.generator.get_descriptor_names()
        ]

    def encode(self, feature: "CategoricalInput", values: pd.Series) -> pd.DataFrame:
        self._prepare(feature)
        smiles = values.map(self._smiles_map(feature))
        descriptor_values = self.generator.get_descriptor_values(smiles)
        descriptor_values.columns = self.get_names(feature)
        descriptor_values.index = values.index
        return descriptor_values

    def get_bounds(
        self,
        feature: "CategoricalInput",
        values: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        data = self.encode(
            feature,
            pd.Series(feature.get_allowed_categories())
            if values is None
            else pd.Series(feature.categories),
        )
        return data.min(axis=0).values.tolist(), data.max(axis=0).values.tolist()

    def decode(self, feature: "CategoricalInput", values: pd.DataFrame) -> pd.Series:
        self._prepare(feature)
        cat_cols = self.get_names(feature)
        # we allow here explicitly that the dataframe can have more columns than
        # needed to make the back-transform easier.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{feature.key}: Column names don't match categorical levels: "
                f"{values.columns}, {cat_cols}.",
            )
        allowed = feature.get_allowed_categories()
        reference = self.encode(feature, pd.Series(allowed))
        s = pd.DataFrame(
            data=np.sqrt(
                np.sum(
                    (
                        values[cat_cols].to_numpy()[:, np.newaxis, :]
                        - reference.to_numpy()
                    )
                    ** 2,
                    axis=2,
                ),
            ),
            columns=allowed,
            index=values.index,
        ).idxmin(1)
        s.name = feature.key
        return s
