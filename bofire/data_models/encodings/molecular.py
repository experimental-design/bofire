"""Molecular encoding: generate descriptors from a feature's SMILES column."""

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from bofire.data_models.encodings.encoding import CategoricalEncoding
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
        """Run correlation filtering once (cached on the generator)."""
        if self.generator._descriptors is None:
            self.generator.remove_correlated_descriptors(
                self._structure_column(feature)
            )

    def get_names(self, feature: "CategoricalInput") -> List[str]:
        from bofire.data_models.features.feature import get_encoded_name

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
