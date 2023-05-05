from typing import ClassVar, List, Literal, Optional, Tuple, Union

import pandas as pd

from bofire.data_models.features.categorical import _CAT_SEP, TTransform
from bofire.data_models.enum import CategoricalMolecularEncodingEnum

from bofire.data_models.features.feature import Input
from bofire.utils.cheminformatics import (
    smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2mol,
    smiles2mordred,
)

from bofire.data_models.features.feature import (
    TDescriptors,
    # DescriptorConstructors,
)


class MolecularInput(Input):
    type: Literal["MolecularInput"] = "MolecularInput"
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

    def fixed_value(self, transform_type: Optional[TTransform] = None) -> None:
        return None

    def is_fixed(self) -> bool:
        return False

    # TODO: model descriptors as pydantic class
    def to_fingerprints(
        self, values: pd.Series, bond_radius: int = 5, n_bits: int = 2048
    ) -> pd.DataFrame:
        # validate it
        data = smiles2fingerprints(
            values.to_list(), bond_radius=bond_radius, n_bits=n_bits
        )
        return pd.DataFrame(
            data=data,
            columns=[f"{self.key}{_CAT_SEP}{i}" for i in range(data.shape[1])],
        )

    def to_bag_of_characters(
        self, values: pd.Series, max_ngram: int = 5
    ) -> pd.DataFrame:
        # todo: add selfies later
        data = smiles2bag_of_characters(values.to_list(), max_ngram=max_ngram)
        return pd.DataFrame(
            data=data,
            columns=[f"{self.key}{_CAT_SEP}{i}" for i in range(data.shape[1])],
        )

    def to_fragments(self, values: pd.Series):
        data = smiles2fragments(values.to_list())
        return pd.DataFrame(
            data=data,
            columns=[f"{self.key}{_CAT_SEP}{i}" for i in range(data.shape[1])],
        )

    def to_fingerprints_fragments(self, values: pd.Series):
        fingerprints_df = self.to_fingerprints(values)
        fragments_df = self.to_fragments(values)

        df = pd.concat((fingerprints_df, fragments_df), axis=1)
        df.columns = [f"{self.key}{_CAT_SEP}{i}" for i in range(fingerprints_df.shape[1]+ fragments_df.shape[1])]

        return df

    def sample(self, n: int) -> pd.Series:
        raise ValueError("Sampling not supported for `MolecularInput`.")

    def get_bounds(
        self, transform_type: TTransform, values: pd.Series
    ) -> Tuple[List[float], List[float]]:
        # TODO: this is only needed for optimization for which we need also
        # MolecularCategorical, this will be added later.
        # raise NotImplementedError("`get_bounds` not yet implemented.")
        pass

class CategoricalMolecularDescriptorInput(MolecularInput):
    descriptors: TDescriptors

    type: Literal["CategoricalMolecularDescriptorInput"] = "CategoricalMolecularDescriptorInput"
    order: ClassVar[int] = 7
    descriptor_values: Optional[List[List[Union[float, int]]]] = None

    def to_mordred_descriptors(self, values: pd.Series):
        if self.descriptor_values is None:
            self.descriptor_values = smiles2mordred(values.to_list(), self.descriptors).values.tolist()

        if len(self.descriptor_values) != len(values):
            self.descriptor_values = smiles2mordred(values.to_list(), self.descriptors).values.tolist()

        # return self.descriptor_values
        return pd.DataFrame(
            data=self.descriptor_values,  # type: ignore
            columns=[f"{self.key}{_CAT_SEP}{d}" for d in self.descriptors],
            index=values.index,
        )

    def get_bounds(
        self, transform_type: TTransform, values: pd.Series
    ) -> Tuple[List[float], List[float]]:
        if self.descriptor_values is None:
            data = self.to_mordred_descriptors(values)

        if len(self.descriptor_values) != len(values):
            data = self.to_mordred_descriptors(values)

        else:
            data = pd.DataFrame(
                data=self.descriptor_values,  # type: ignore
                columns=[f"{self.key}{_CAT_SEP}{d}" for d in self.descriptors],
                index=values.index,
        )

        lower = data.min(axis=0).values.tolist()
        upper = data.max(axis=0).values.tolist()

        return lower, upper