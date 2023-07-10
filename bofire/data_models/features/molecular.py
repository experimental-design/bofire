from typing import ClassVar, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import Field

from bofire.data_models.features.categorical import _CAT_SEP
from bofire.data_models.features.feature import Input, TMolecularVals
from bofire.data_models.molfeatures.api import AnyMolFeatures
from bofire.utils.cheminformatics import smiles2mol


class MolecularInput(Input):
    type: Literal["MolecularInput"] = "MolecularInput"
    descriptor_values: TMolecularVals = Field(default_factory=list)
    order: ClassVar[int] = 6

    def validate_experimental(
        self, values: pd.Series, strict: bool = False
    ) -> pd.Series:
        for smi in values:
            smiles2mol(smi)

        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        for smi in values:
            smiles2mol(smi)
        return values

    def is_fixed(self) -> bool:
        return False

    def fixed_value(self, transform_type: Optional[AnyMolFeatures] = None) -> None:
        return None

    def sample(self, n: int) -> pd.Series:
        raise ValueError("Sampling not supported for `MolecularInput`")

    def get_bounds(
        self, transform_type: AnyMolFeatures, values: pd.Series
    ) -> Tuple[List[float], List[float]]:
        if len(self.descriptor_values) == 0:
            self.descriptor_values = transform_type.get_descriptor_values(
                values
            ).values.tolist()

        if values is None:
            raise NotImplementedError(
                "`values` is currently required for `MolecularInput`"
            )
        else:
            data = self.to_descriptor_encoding(transform_type, values)

        lower = data.min(axis=0).values.tolist()
        upper = data.max(axis=0).values.tolist()

        return lower, upper

    def to_descriptor_encoding(
        self, transform_type: AnyMolFeatures, values: pd.Series
    ) -> pd.DataFrame:
        """Converts values to descriptor encoding.

        Args:
            values (pd.Series): Values to transform.

        Returns:
            pd.DataFrame: Descriptor encoded dataframe.
        """
        if len(self.descriptor_values) == 0:
            self.descriptor_values = transform_type.get_descriptor_values(
                values
            ).values.tolist()

        return pd.DataFrame(
            data=self.descriptor_values,
            columns=[
                f"{self.key}{_CAT_SEP}{d}"
                for d in transform_type.get_descriptor_names()
            ],
            index=values.index,
        )
