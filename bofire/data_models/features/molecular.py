from typing import ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bofire.data_models.features.categorical import _CAT_SEP, TTransform

from bofire.data_models.features.categorical import CategoricalInput
from bofire.utils.cheminformatics import smiles2mol
from bofire.data_models.enum import MolecularEncodingEnum

from bofire.data_models.molfeatures.api import AnyMolFeatures

class MolecularInput(CategoricalInput):
    smiles: List[str]
    molfeatures: AnyMolFeatures
    values: Optional[List[List[Union[float, int]]]] = None
    type: Literal["MolecularInput"] = "MolecularInput"
    order: ClassVar[int] = 6

    def validate_experimental(
        self, values: pd.Series, strict: bool = False
    ) -> pd.Series:
        for smi in self.name2smiles(values):
            smiles2mol(smi)
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        for smi in self.name2smiles(values):
            smiles2mol(smi)
        return values

    def fixed_value(
        self, transform_type: Optional[TTransform] = None
    ) -> Union[List[str], List[float], None]:
        """Returns the categories to which the feature is fixed, None if the feature is not fixed

        Returns:
            List[str]: List of categories or None
        """
        if transform_type != MolecularEncodingEnum.FINGERPRINTS or transform_type != MolecularEncodingEnum.FRAGMENTS or transform_type != MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS or transform_type != MolecularEncodingEnum.MOL_DESCRIPTOR:
            return super().fixed_value(transform_type)
        else:
            val = self.get_allowed_categories()[0]
            return self.to_descriptor_encoding(pd.Series([val])).values[0].tolist()

    def is_fixed(self) -> bool:
        return False

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(
            name=self.key,
            data=np.random.choice(self.categories, n),
        )

    def name2smiles(self, values:pd.Series) -> pd.Series:
        return values.replace({cat: smi for cat, smi in zip(self.categories, self.smiles)})

    def smiles2name(self, values:pd.Series) -> pd.Series:
        return values.replace({smi: cat for cat, smi in zip(self.categories, self.smiles)})

    def generate_descriptors(self):
        self.values = self.molfeatures(pd.Series(self.smiles)).values.tolist()

    def get_bounds(
        self, transform_type: TTransform, values: pd.Series
    ) -> Tuple[List[float], List[float]]:
        if self.values is None:
            self.generate_descriptors()

        if values is None:
            data = self.to_df().loc[self.get_allowed_categories()]
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
        data = {cat: values for cat, values in zip(self.categories, self.values)}
        return pd.DataFrame.from_dict(data, orient="index", columns=self.molfeatures.descriptors)

    def to_descriptor_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to descriptor encoding.

        Args:
            values (pd.Series): Values to transform.

        Returns:
            pd.DataFrame: Descriptor encoded dataframe.
        """
        if self.values is None:
            self.generate_descriptors()

        return pd.DataFrame(
            data=values.map(
                {cat: value for cat, value in zip(self.categories, self.values)}
            ).values.tolist(),  # type: ignore
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
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols}."
            )
        s = pd.DataFrame(
            data=np.sqrt(
                np.sum(
                    (
                        values[cat_cols].to_numpy()[:, np.newaxis, :]
                        - self.to_df().iloc[self.allowed].to_numpy()
                    )
                    ** 2,
                    axis=2,
                )
            ),
            columns=self.get_allowed_categories(),
            index=values.index,
        ).idxmin(1)
        s.name = self.key
        return s

