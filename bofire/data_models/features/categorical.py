from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo

from bofire.data_models.encodings.reserved import get_reserved_descriptor, is_reserved
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.feature import (
    Input,
    Output,
    TTransform,
    get_encoded_name,
)
from bofire.data_models.objectives.api import AnyCategoricalObjective
from bofire.data_models.types import CategoryVals


# Max number of allowed categories still encoded as ``Literal[...]`` by
# ``to_pydantic_field``. Above this we emit ``str`` with the allowed values
# kept in the field description; the ``Domain.validate_candidates`` pass then
# catches invalid values at ask-time.
#
# Why: providers that offer constrained-decoding for structured output (OpenAI,
# Anthropic) compile each JSON Schema enum into a token-level mask. The
# compile cost scales with the total byte-length of all enum values, not just
# their count. For hundreds of long strings (e.g. SMILES categories) this
# blows the provider's compiled-schema budget and the request is rejected.
# Observed failure: Anthropic returns 400 "Schema is too complex for
# compilation." with ~390 SMILES. OpenAI documents a hard cap at 500 enum
# values and an additional 7500-char combined-length cap above 250 values.
# 32 is well below any documented limit and leaves headroom for very long
# category strings.
LLM_ENUM_SCHEMA_THRESHOLD = 32


class CategoricalInput(Input):
    """Base class for all categorical input features.

    Attributes:
        categories (List[str]): Names of the categories.
        allowed (List[bool]): List of bools indicating if a category is allowed within the optimization.

    """

    type: Literal["CategoricalInput"] = "CategoricalInput"
    # order_id: ClassVar[int] = 5
    order_id: ClassVar[int] = 7

    categories: CategoryVals
    allowed: Optional[Annotated[List[bool], Field(min_length=2)]] = Field(
        default=None,
        validate_default=True,
    )
    descriptors: Dict[str, List[Any]] = Field(default_factory=dict)

    @field_validator("allowed")
    @classmethod
    def generate_allowed(cls, allowed, info):
        """Generates the list of allowed categories if not provided."""
        if allowed is None and "categories" in info.data.keys():
            return [True for _ in range(len(info.data["categories"]))]
        return allowed

    @field_validator("descriptors")
    @classmethod
    def validate_descriptors(cls, descriptors, info):
        """Validates the per-category descriptor table.

        Each column must have one entry per category. Reserved keys (e.g.
        ``smiles``) are validated and typed by the reserved-descriptor registry;
        all other columns are free-form numeric descriptors coerced to ``float``.
        """
        categories = info.data.get("categories")
        if categories is None:
            return descriptors
        validated: Dict[str, List[Any]] = {}
        for name, column in descriptors.items():
            if len(column) != len(categories):
                raise ValueError(
                    f"descriptor column '{name}' must have same length as categories",
                )
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

    def descriptor_columns(self, role: Optional[str] = None) -> List[str]:
        """Names of descriptor columns, optionally filtered by role.

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
        """Per-category table (rows=categories, columns=selected descriptors)."""
        return pd.DataFrame(
            {c: self.descriptors[c] for c in columns},
            index=list(self.categories),
        )

    @model_validator(mode="after")
    def validate_categories_fitting_allowed(self):
        if len(self.allowed) != len(self.categories):
            raise ValueError("allowed must have same length as categories")
        if sum(self.allowed) == 0:
            raise ValueError("no category is allowed")
        return self

    def _description_prefix(self) -> str:
        """Leading description string identifying this feature kind."""
        return f"Categorical, allowed: {self.get_allowed_categories()}"

    def _extra_description_parts(self) -> List[str]:
        """Optional extras appended after the prefix, before context."""
        return []

    def to_pydantic_field(self) -> Tuple[type, FieldInfo]:
        """Return ``(Literal[...], Field(description=...))`` with allowed categories.

        When the number of allowed categories exceeds
        ``LLM_ENUM_SCHEMA_THRESHOLD`` the type falls back to ``str`` (the
        allowed values stay in the description). See the module-level comment
        on the constant for the reason.

        Subclasses customize the output by overriding ``_description_prefix``
        and/or ``_extra_description_parts``.

        Example::

            >>> feat = CategoricalInput(key="solvent", categories=["water", "ethanol", "toluene"])
            >>> field_type, _ = feat.to_pydantic_field()
            >>> # field_type = Literal['water', 'ethanol', 'toluene']
        """
        allowed = self.get_allowed_categories()
        desc_parts = [self._description_prefix(), *self._extra_description_parts()]
        if self.context:
            desc_parts.append(self.context)
        field_type: type = (
            str
            if len(allowed) > LLM_ENUM_SCHEMA_THRESHOLD
            else Literal[tuple(allowed)]  # ty: ignore[invalid-assignment]
        )
        return (
            field_type,
            Field(description=" — ".join(desc_parts)),
        )

    def valid_transform_types(self) -> List:
        """Valid encodings for this feature.

        The param-less enum encodings are always valid; the object encoders are
        valid only when the feature actually carries the data they consume
        (numeric descriptor columns for ``DescriptorEncoding``, a structure
        column for ``MolecularEncoding``).
        """
        from bofire.data_models.encodings.api import (
            DescriptorEncoding,
            MolecularEncoding,
        )

        types: List = [
            CategoricalEncodingEnum.ONE_HOT,
            CategoricalEncodingEnum.DUMMY,
            CategoricalEncodingEnum.ORDINAL,
        ]
        if self.descriptor_columns(role="descriptor"):
            types.append(DescriptorEncoding)
        if self.descriptor_columns(role="structure"):
            types.append(MolecularEncoding)
        return types

    def is_fixed(self) -> bool:
        """Returns True if there is only one allowed category.

        Returns:
            [bool]: True if there is only one allowed category

        """
        if self.allowed is None:
            return False
        return sum(self.allowed) == 1

    def fixed_value(
        self,
        transform_type: Optional[TTransform] = None,
    ) -> Union[List[str], List[float], None]:
        """Returns the categories to which the feature is fixed, None if the feature is not fixed

        Returns:
            List[str]: List of categories or None

        """
        if self.is_fixed():
            val = self.get_allowed_categories()[0]
            if transform_type is None:
                return [val]
            if transform_type == CategoricalEncodingEnum.ONE_HOT:
                return self.to_onehot_encoding(pd.Series([val])).values[0].tolist()
            if transform_type == CategoricalEncodingEnum.DUMMY:
                return self.to_dummy_encoding(pd.Series([val])).values[0].tolist()
            if transform_type == CategoricalEncodingEnum.ORDINAL:
                return self.to_ordinal_encoding(pd.Series([val])).tolist()
            raise ValueError(
                f"Unknown transform type {transform_type} for categorical input {self.key}",
            )
        return None

    def get_allowed_categories(self) -> list[str]:
        """Returns the allowed categories.

        Returns:
            list of str: The allowed categories

        """
        if self.allowed is None:
            return []
        return [c for c, a in zip(self.categories, self.allowed) if a]

    def validate_experimental(
        self,
        values: pd.Series,
        strict: bool = False,
    ) -> pd.Series:
        """Method to validate the experimental dataFrame

        Args:
            values (pd.Series): A dataFrame with experiments
            strict (bool, optional): Boolean to distinguish if the occurrence of fixed features in the dataset should be considered or not. Defaults to False.

        Raises:
            ValueError: when an entry is not in the list of allowed categories
            ValueError: when there is no variation in a feature provided by the experimental data

        Returns:
            pd.Series: A dataFrame with experiments

        """
        values = values.map(str)
        if sum(values.isin(self.categories)) != len(values):
            raise ValueError(
                f"invalid values for `{self.key}`, allowed are: `{self.categories}`",
            )
        if strict:
            possible_categories = self.get_possible_categories(values)
            if len(possible_categories) != len(self.categories):
                raise ValueError(
                    f"Categories {list(set(self.categories) - set(possible_categories))} of feature {self.key} not used. Remove them.",
                )
        return values

    def is_fulfilled(self, values: pd.Series) -> pd.Series:
        """Method to check if the values are all allowed categories.

        Args:
            values: A series with values for the input feature.

        Returns:
            A series with boolean values indicating if the input feature is fulfilled.
        """
        return values.isin(self.get_allowed_categories())

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Raises:
            ValueError: when not all values for a feature are one of the allowed categories

        Returns:
            pd.Series: The passed dataFrame with candidates

        """
        values = values.map(str)
        if sum(values.isin(self.get_allowed_categories())) != len(values):
            raise ValueError(
                f"not all values of input feature `{self.key}` are a valid allowed category from {self.get_allowed_categories()}",
            )
        return values

    def get_forbidden_categories(self):
        """Returns the non-allowed categories

        Returns:
            List[str]: List of the non-allowed categories

        """
        return list(set(self.categories) - set(self.get_allowed_categories()))

    def get_possible_categories(self, values: pd.Series) -> list:
        """Return the superset of categories that have been used in the experimental dataset and
        that can be used in the optimization

        Args:
            values (pd.Series): Series with the values for this feature

        Returns:
            list: list of possible categories

        """
        return sorted(set(list(set(values.tolist())) + self.get_allowed_categories()))

    def to_onehot_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to a one-hot encoding.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.DataFrame: One-hot transformed data frame.

        """
        return pd.DataFrame(
            {get_encoded_name(self.key, c): values == c for c in self.categories},
            dtype=float,
            index=values.index,
        )

    def from_onehot_encoding(self, values: pd.DataFrame) -> pd.Series:
        """Converts values back from one-hot encoding.

        Args:
            values (pd.DataFrame): One-hot encoded values.

        Raises:
            ValueError: If one-hot columns not present in `values`.

        Returns:
            pd.Series: Series with categorical values.

        """
        cat_cols = [get_encoded_name(self.key, c) for c in self.categories]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols}.",
            )
        s = values[cat_cols].idxmax(1).str[(len(self.key) + 1) :]
        s.name = self.key
        return s

    def to_dummy_encoding(self, values: pd.Series) -> pd.DataFrame:
        """Converts values to a dummy-hot encoding, dropping the first categorical level.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.DataFrame: Dummy-hot transformed data frame.

        """
        return pd.DataFrame(
            {get_encoded_name(self.key, c): values == c for c in self.categories[1:]},
            dtype=float,
            index=values.index,
        )

    def from_dummy_encoding(self, values: pd.DataFrame) -> pd.Series:
        """Convert points back from dummy encoding.

        Args:
            values (pd.DataFrame): Dummy-hot encoded values.

        Raises:
            ValueError: If one-hot columns not present in `values`.

        Returns:
            pd.Series: Series with categorical values.

        """
        cat_cols = [get_encoded_name(self.key, c) for c in self.categories]
        # we allow here explicitly that the dataframe can have more columns than needed to have it
        # easier in the backtransform.
        if np.any([c not in values.columns for c in cat_cols[1:]]):
            raise ValueError(
                f"{self.key}: Column names don't match categorical levels: {values.columns}, {cat_cols[1:]}.",
            )
        values = values.copy()
        values[cat_cols[0]] = 1 - values[cat_cols[1:]].sum(axis=1)
        s = values[cat_cols].idxmax(1).str[(len(self.key) + 1) :]
        s.name = self.key
        return s

    def to_ordinal_encoding(self, values: pd.Series) -> pd.Series:
        """Converts values to an ordinal integer based encoding.

        Args:
            values (pd.Series): Series to be transformed.

        Returns:
            pd.Series: Ordinal encoded values.

        """
        enc = pd.Series(range(len(self.categories)), index=list(self.categories))
        s = enc[values]
        s.index = values.index
        s.name = self.key
        return s

    def from_ordinal_encoding(self, values: pd.Series) -> pd.Series:
        """Convertes values back from ordinal encoding.

        Args:
            values (pd.Series): Ordinal encoded series.

        Returns:
            pd.Series: Series with categorical values.

        """
        enc = np.array(self.categories)
        return pd.Series(enc[values], index=values.index, name=self.key)

    def sample(self, n: int, seed: Optional[int] = None) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.
            seed (int, optional): random seed. Defaults to None.

        Returns:
            pd.Series: drawn samples.

        """
        return pd.Series(
            name=self.key,
            data=np.random.default_rng(seed=seed).choice(
                self.get_allowed_categories(),
                n,
            ),
        )

    def get_bounds(
        self,
        transform_type: TTransform,
        values: Optional[pd.Series] = None,
        reference_value: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[float], List[float]]:
        assert isinstance(transform_type, CategoricalEncodingEnum)
        if transform_type == CategoricalEncodingEnum.ORDINAL:
            return [0], [len(self.categories) - 1]
        if transform_type == CategoricalEncodingEnum.ONE_HOT:
            # in the case that values are None, we return the bounds
            # based on the optimization bounds, else we return the true
            # bounds as this is for model fitting.
            if values is None:
                lower = [0.0 for _ in self.categories]
                upper = [
                    1.0
                    if self.allowed[i] is True  # ty: ignore[not-subscriptable]
                    else 0.0
                    for i, _ in enumerate(self.categories)
                ]
            else:
                lower = [0.0 for _ in self.categories]
                upper = [1.0 for _ in self.categories]
            return lower, upper
        if transform_type == CategoricalEncodingEnum.DUMMY:
            lower = [0.0 for _ in range(len(self.categories) - 1)]
            upper = [1.0 for _ in range(len(self.categories) - 1)]
            return lower, upper
        raise ValueError(
            f"Invalid transform_type {transform_type} provided for categorical {self.key}.",
        )

    def __str__(self) -> str:
        """Returns the number of categories as str

        Returns:
            str: Number of categories

        """
        return f"{len(self.categories)} categories"


class CategoricalOutput(Output):
    type: Literal["CategoricalOutput"] = "CategoricalOutput"
    order_id: ClassVar[int] = 10

    categories: CategoryVals
    objective: AnyCategoricalObjective

    def to_description(self) -> str:
        raise NotImplementedError

    @model_validator(mode="after")
    def validate_objective_categories(self):
        """Validates that objective categories match the output categories

        Raises:
            ValueError: when categories do not match objective categories

        Returns:
            self

        """
        if self.objective.categories != self.categories:
            raise ValueError("categories must match to objective categories")
        return self

    def __call__(self, values: pd.Series, values_adapt: pd.Series) -> pd.Series:
        if self.objective is None:
            return pd.Series(
                data=[np.nan for _ in range(len(values))],
                index=values.index,
                name=values.name,
            )
        return self.objective(values, values_adapt)  # ty: ignore[invalid-return-type]

    def validate_experimental(self, values: pd.Series) -> pd.Series:
        values = values.map(str)
        if sum(values.isin(self.categories)) != len(values):
            raise ValueError(
                f"invalid values for `{self.key}`, allowed are: `{self.categories}`",
            )
        return values

    def __str__(self) -> str:
        return "CategoricalOutputFeature"
