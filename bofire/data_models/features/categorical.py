from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo

from bofire.data_models.encodings.reserved import get_reserved_descriptor, is_reserved
from bofire.data_models.features.feature import Input, Output, TTransform
from bofire.data_models.objectives.api import AnyCategoricalObjective
from bofire.data_models.types import CategoryVals


if TYPE_CHECKING:
    from bofire.data_models.encodings.encoding import CategoricalEncoding


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
        """Valid encoding classes for this feature.

        One-hot and ordinal are always valid; the data-backed encoders are valid
        only when the feature carries the data they consume (numeric descriptor
        columns for ``DescriptorEncoding``, a structure column for
        ``MolecularEncoding``).
        """
        from bofire.data_models.encodings.api import (
            DescriptorEncoding,
            MolecularEncoding,
            OneHotEncoding,
            OrdinalEncoding,
        )

        types: List = [OneHotEncoding, OrdinalEncoding]
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
            return (
                transform_type.encode(
                    self, pd.Series([val])
                )  # ty: ignore[unresolved-attribute]
                .values[0]
                .tolist()
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

    def to_encoding(
        self,
        encoding: "CategoricalEncoding",
        values: pd.Series,
    ) -> pd.DataFrame:
        """Encode a series of categories with the given encoding.

        Generic pandas entry point: the encoding object owns the actual transform.
        """
        return encoding.encode(self, values)

    def from_encoding(
        self,
        encoding: "CategoricalEncoding",
        values: pd.DataFrame,
    ) -> pd.Series:
        """Back-transform encoded columns to categories with the given encoding."""
        return encoding.decode(self, values)

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
        # bounds are encoding-specific and delegated to the encoder object.
        if transform_type is None:
            raise ValueError(
                f"An encoding must be provided to get bounds for categorical {self.key}.",
            )
        return transform_type.get_bounds(
            self, values
        )  # ty: ignore[unresolved-attribute]

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
