from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import validator

from bofire.data_models.features.categorical import _CAT_SEP, TTransform

from bofire.data_models.features.feature import Input
from bofire.utils.cheminformatics import smiles2mol
from bofire.data_models.enum import MolecularEncodingEnum

from bofire.data_models.molfeatures.api import AnyMolFeatures
from bofire.data_models.features.feature import TSmiles, TMolecularVals


class MolecularInput(Input):
    type: Literal["MolecularInput"] = "MolecularInput"
    smiles: TSmiles
    molfeatures: AnyMolFeatures
    descriptor_values: Optional[TMolecularVals] = None # Optional[List[List[Union[float, int]]]] = None
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

    def fixed_value(self, transform_type: Optional[TTransform] = None) -> None:
        return None

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(
            name=self.key,
            data=np.random.choice(self.smiles, n),
        )

    def generate_descriptor_values(self):
        self.descriptor_values = self.molfeatures(pd.Series(self.smiles)).values.tolist()

    def get_bounds(
        self, transform_type: TTransform, values: pd.Series
    ) -> Tuple[List[float], List[float]]:
        if self.descriptor_values is None:
            self.generate_descriptor_values()

        if values is None:
            data = self.to_df().loc[self.smiles]
        else:
            data = self.to_descriptor_encoding(values)

        lower = data.min(axis=0).values.tolist()
        upper = data.max(axis=0).values.tolist()

        return lower, upper

    def to_df(self):
        """tabular overview of the feature as DataFrame

        Returns:
            pd.DataFrame: tabular overview of the feature as DataFrame
        """
        data = {smi: values for smi, values in zip(self.smiles, self.descriptor_values)}
        return pd.DataFrame.from_dict(
            data, orient="index", columns=self.molfeatures.descriptors
        )

    def to_descriptor_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to descriptor encoding.

        Args:
            values (pd.Series): Values to transform.

        Returns:
            pd.DataFrame: Descriptor encoded dataframe.
        """
        if self.descriptor_values is None:
            self.generate_descriptor_values()

        return pd.DataFrame(
            data=values.map(
                {smi: value for smi, value in zip(self.smiles, self.descriptor_values)}
            ).values.tolist(),
            columns=[f"{self.key}{_CAT_SEP}{d}" for d in self.molfeatures.descriptors],
            index=values.index,
        )

    def from_descriptor_encoding(self, values: pd.DataFrame) -> pd.Series:
        """Converts values back from descriptor encoding.

        Args:
            values (pd.DataFrame): Descriptor encoded dataframe.

        Raises:
            ValueError: If descriptor columns not found in the dataframe.

        Returns:
            pd.Series: Series with categorical values.
        """
        cat_cols = [f"{self.key}{_CAT_SEP}{d}" for d in self.molfeatures.descriptors]
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols}."
            )
        s = pd.DataFrame(
            data=np.sqrt(
                np.sum(
                    (
                        values[cat_cols].to_numpy()[:, np.newaxis, :]
                        - self.to_df().to_numpy()
                    )
                    ** 2,
                    axis=2,
                )
            ),
            columns=self.smiles,
            index=values.index,
        ).idxmin(1)
        s.name = self.key
        return s
