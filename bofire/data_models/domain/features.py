from __future__ import annotations

import itertools
import re
import warnings
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, validate_call
from scipy.stats.qmc import LatinHypercube, Sobol
from typing_extensions import Self

from bofire.data_models.base import BaseModel
from bofire.data_models.enum import CategoricalEncodingEnum, SamplingMethodEnum
from bofire.data_models.features.api import (
    AnyFeature,
    AnyInput,
    AnyOutput,
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    Input,
    MolecularInput,
    Output,
    TaskInput,
)
from bofire.data_models.features.feature import get_encoded_name
from bofire.data_models.filters import filter_by_attribute, filter_by_class
from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.objectives.api import (
    AbstractObjective,
    ConstrainedCategoricalObjective,
    Objective,
)
from bofire.data_models.types import InputTransformSpecs


F = TypeVar("F", bound=AnyFeature)
FeatureSequence = Sequence[F]


class _BaseFeatures(BaseModel, Generic[F]):
    """Container of features, both input and output features are allowed.

    Attributes:
        features: list of the features.

    """

    type: Literal["Features"] = "Features"
    features: FeatureSequence = Field(default_factory=list)

    @field_validator("features")
    @classmethod
    def validate_unique_feature_keys(
        cls: type[_BaseFeatures],
        features: FeatureSequence,
    ) -> FeatureSequence:
        keys = [feat.key for feat in features]
        if len(keys) != len(set(keys)):
            raise ValueError("Feature keys are not unique.")
        return features

    def __iter__(self) -> Iterator[F]:  # type: ignore
        return iter(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def __add__(self, other: Union[Sequence[AnyFeature], _BaseFeatures]):
        if isinstance(other, Features):
            other_feature_seq = other.features
        else:
            other_feature_seq = other
        new_feature_seq = list(itertools.chain(self.features, other_feature_seq))

        def is_feats_of_type(feats, ftype_collection, ftype_element):
            return isinstance(feats, ftype_collection) or (
                not isinstance(feats, Features)
                and (len(feats) > 0 and isinstance(feats[0], ftype_element))
            )

        def is_infeats(feats):
            return is_feats_of_type(feats, Inputs, Input)

        def is_outfeats(feats):
            return is_feats_of_type(feats, Outputs, Output)

        if is_infeats(self) and is_infeats(other):
            return Inputs(features=cast(Tuple[AnyInput, ...], new_feature_seq))
        if is_outfeats(self) and is_outfeats(other):
            return Outputs(features=cast(Tuple[AnyOutput, ...], new_feature_seq))
        return Features(features=new_feature_seq)

    def get_by_key(self, key: str, use_regex: bool = False) -> F:
        """Get a feature by its key.

        First, the method tries to find the feature by its key. If no feature is
        found, the method tries to find the feature by a regular expression match,
        if `use_regex` is set to `True`.

        If no feature is found, a KeyError is raised.

        Args:
            key: Feature key of the feature of interest
            use_regex: Boolean to distinguish if the key can be interpreted as a regular
                expression.

        Returns:
            Feature: Feature of interest

        """
        try:
            if use_regex:
                return next(
                    f for f in self.features if re.search(key, f.key) is not None
                )
            else:
                return next(f for f in self.features if f.key == key)
        except StopIteration:
            raise KeyError(f"Feature with key {key} not found.")

    def get_by_keys(self, keys: Sequence[str], include: bool = True) -> Self:
        """Get features of the domain specified by its keys.

        Args:
            keys: List of the keys of the features that should be returned.
            include: Boolean to distinguish if the features with the keys in the
                list should be included or excluded.

        Returns:
            Features: Features object with the requested features.

        """
        if include:
            features = [self.get_by_key(key) for key in keys]
        else:
            features = [f for f in self.features if f.key not in keys]
        return self.__class__(features=sorted(features))

    def get(
        self,
        includes: Union[Type, List[Type], None] = AnyFeature,  # type: ignore
        excludes: Union[Type, List[Type], None] = None,
        exact: bool = False,
    ) -> Self:
        """Get features of this container and filter via includes and excludes.

        Args:
            includes: All features in this container that are instances of an
                include are returned. If None, the include filter is not active.
            excludes: All features in this container that are not instances of
                an exclude are returned. If None, the exclude filter is not active.
            exact: Boolean to distinguish if only the exact class listed in
                includes and no subclasses inherenting from this class shall
                be returned.

        Returns:
            List of features in the domain fitting to the passed requirements.

        """
        return self.__class__(
            features=sorted(
                filter_by_class(
                    self.features,
                    includes=includes,
                    excludes=excludes,
                    exact=exact,
                ),
            ),
        )

    def get_keys(
        self,
        includes: Union[Type, List[Type], None] = AnyFeature,  # type: ignore
        excludes: Union[Type, List[Type], None] = None,
        exact: bool = False,
    ) -> List[str]:
        """Get feature-keys of this container and filter via includes and excludes.

        Args:
            includes: All features in this container that are instances of an
                include are returned. If None, the include filter is not active.
            excludes: All features in this container that are not instances of
                an exclude are returned. If None, the exclude filter is not active.
            exact: Boolean to distinguish if only the exact class listed in
                includes and no subclasses inherenting from this class shall
                be returned.

        Returns:
            List of feature keys fitting to the passed requirements.

        """
        return [
            f.key
            for f in self.get(
                includes=includes,
                excludes=excludes,
                exact=exact,
            )
        ]

    def get_reps_df(self) -> pd.DataFrame:
        """Returns a pandas dataframe describing the features contained in the optimization domain."""
        df = pd.DataFrame(
            index=self.get_keys(Feature),
            columns=["Type", "Description"],
            data={
                "Type": [feat.__class__.__name__ for feat in self.get(Feature)],
                "Description": [feat.__str__() for feat in self.get(Feature)],
            },
        )
        return df


class Features(_BaseFeatures[AnyFeature]):
    pass


class Inputs(_BaseFeatures[AnyInput]):
    """Container of input features, only input features are allowed.

    Attributes:
        features (List(Inputs)): list of the features.

    """

    type: Literal["Inputs"] = "Inputs"  # type: ignore

    @field_validator("features")
    @classmethod
    def validate_only_one_task_input(cls, features: Sequence[AnyInput]):
        filtered = filter_by_class(
            features,
            includes=TaskInput,
            excludes=None,
            exact=False,
        )
        if len(filtered) > 1:
            raise ValueError(f"Only one `TaskInput` is allowed, got {len(filtered)}.")
        return features

    def get_fixed(self) -> Inputs:
        """Gets all features in `self` that are fixed and returns them as new
        `Inputs` object.

        Returns:
            Inputs: Input features object containing only fixed features.

        """
        return Inputs(features=[feat for feat in self if feat.is_fixed()])

    def get_free(self) -> Inputs:
        """Gets all features in `self` that are not fixed and returns them as
        new `Inputs` object.

        Returns:
            Inputs: Input features object containing only non-fixed features.

        """
        return Inputs(features=[feat for feat in self if not feat.is_fixed()])

    @validate_call
    def sample(
        self,
        n: int = 1,
        method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Draw sobol samples

        Args:
            n (int, optional): Number of samples, has to be larger than 0.
                Defaults to 1.
            method (SamplingMethodEnum, optional): Method to use, implemented
                methods are `UNIFORM`, `SOBOL` and `LHS`. Defaults to `UNIFORM`.
            reference_value
            seed (int, optional): random seed. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe containing the samples.

        """
        if len(self) == 0:
            return pd.DataFrame()

        if method == SamplingMethodEnum.UNIFORM:
            # we cannot just propagate the provided seed to the sample methods
            # as they would then sample always the same value if the bounds
            # are the same for a feature.
            rng = np.random.default_rng(seed=seed)
            return self.validate_candidates(
                pd.concat(
                    [
                        feat.sample(n, seed=int(rng.integers(1, 1000000)))
                        for feat in self.get(Input)
                    ],
                    axis=1,
                ),
            )

        free_features = self.get_free()
        if method == SamplingMethodEnum.SOBOL:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = Sobol(len(free_features), seed=seed).random(n)
        else:
            X = LatinHypercube(len(free_features), seed=seed).random(n)

        res = []
        for i, feat in enumerate(free_features):
            if isinstance(feat, ContinuousInput):
                x = feat.from_unit_range(X[:, i])
            elif isinstance(feat, (DiscreteInput, CategoricalInput)):
                levels = (
                    feat.values
                    if isinstance(feat, DiscreteInput)
                    else feat.get_allowed_categories()
                )
                bins = np.linspace(0, 1, len(levels) + 1)
                idx = np.digitize(X[:, i], bins) - 1
                x = np.array(levels)[idx]
            else:
                raise ValueError(
                    f"Unknown input feature with key {feat.key} of type {feat.type}",
                )
            res.append(pd.Series(x, name=feat.key))

        samples = pd.concat(res, axis=1)

        for feat in self.get_fixed():
            samples[feat.key] = feat.fixed_value()[0]  # type: ignore

        return self.validate_candidates(samples)[self.get_keys(Input)]

    def validate_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Validate a pandas dataframe with input feature values.

        Args:
            candidates (pd.Dataframe): Inputs to validate.

        Raises:
            ValueError: Raises a Valueerror if a feature based validation raises an exception.

        Returns:
            pd.Dataframe: Validated dataframe

        """
        for feature in self:
            if feature.key not in candidates:
                raise ValueError(f"no col for input feature `{feature.key}`")
            candidates[feature.key] = feature.validate_candidental(
                candidates[feature.key],
            )
        if candidates[self.get_keys()].isnull().to_numpy().any():
            raise ValueError("there are null values")
        if candidates[self.get_keys()].isna().to_numpy().any():
            raise ValueError("there are na values")
        return candidates

    def validate_experiments(
        self,
        experiments: pd.DataFrame,
        strict=False,
    ) -> pd.DataFrame:
        for feature in self:
            if feature.key not in experiments:
                raise ValueError(f"no col for input feature `{feature.key}`")
            experiments[feature.key] = feature.validate_experimental(
                experiments[feature.key],
                strict=strict,
            )
        if experiments[self.get_keys()].isnull().to_numpy().any():
            raise ValueError("there are null values")
        if experiments[self.get_keys()].isna().to_numpy().any():
            raise ValueError("there are na values")
        return experiments

    def get_categorical_combinations(
        self,
        include: Union[Type, List[Type]] = Input,
        exclude: Union[Type, List[Type]] = None,  # type: ignore
    ):
        """Get a list of tuples pairing the feature keys with a list of valid categories

        Args:
            include (Feature, optional): Features to be included. Defaults to Input.
            exclude (Feature, optional): Features to be excluded, e.g. subclasses
                of the included features. Defaults to None.

        Returns:
            List[(str, List[str])]: Returns a list of tuples pairing the feature
                keys with a list of valid categories (str)

        """
        features = [
            f
            for f in self.get(includes=include, excludes=exclude)
            if (isinstance(f, CategoricalInput) and not f.is_fixed())
        ]
        list_of_lists = [
            [(f.key, cat) for cat in f.get_allowed_categories()] for f in features
        ]

        discretes = [
            f
            for f in self.get(includes=include, excludes=exclude)
            if (isinstance(f, DiscreteInput) and not f.is_fixed())
        ]

        list_of_lists_2 = [[(d.key, v) for v in d.values] for d in discretes]

        list_of_lists = list_of_lists + list_of_lists_2

        return list(itertools.product(*list_of_lists))

    # transformation related methods
    def _get_transform_info(
        self,
        specs: InputTransformSpecs,
    ) -> Tuple[Dict[str, Tuple[int]], Dict[str, Tuple[str]]]:
        """Generates two dictionaries. The first one specifies which key is mapped to
        which column indices when applying `transform`. The second one specifies
        which key is mapped to which transformed keys.

        Args:
            specs (InputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.

        Returns:
            Dict[str, Tuple[int]]: Dictionary mapping feature keys to column indices.
            Dict[str, Tuple[str]]: Dictionary mapping feature keys to transformed feature
                keys.

        """
        self._validate_transform_specs(specs)
        features2idx = {}
        features2names = {}
        counter = 0
        for _, feat in enumerate(self.get()):
            if feat.key not in specs.keys():
                features2idx[feat.key] = (counter,)
                features2names[feat.key] = (feat.key,)
                counter += 1
            elif specs[feat.key] == CategoricalEncodingEnum.ONE_HOT:
                assert isinstance(feat, CategoricalInput)
                features2idx[feat.key] = tuple(
                    (np.array(range(len(feat.categories))) + counter).tolist(),
                )
                features2names[feat.key] = tuple(
                    [get_encoded_name(feat.key, c) for c in feat.categories],
                )
                counter += len(feat.categories)
            elif specs[feat.key] == CategoricalEncodingEnum.ORDINAL:
                features2idx[feat.key] = (counter,)
                features2names[feat.key] = (feat.key,)
                counter += 1
            elif specs[feat.key] == CategoricalEncodingEnum.DUMMY:
                assert isinstance(feat, CategoricalInput)
                features2idx[feat.key] = tuple(
                    (np.array(range(len(feat.categories) - 1)) + counter).tolist(),
                )
                features2names[feat.key] = tuple(
                    [get_encoded_name(feat.key, c) for c in feat.categories[1:]],
                )
                counter += len(feat.categories) - 1
            elif specs[feat.key] == CategoricalEncodingEnum.DESCRIPTOR:
                assert isinstance(feat, CategoricalDescriptorInput)
                features2idx[feat.key] = tuple(
                    (np.array(range(len(feat.descriptors))) + counter).tolist(),
                )
                features2names[feat.key] = tuple(
                    [get_encoded_name(feat.key, d) for d in feat.descriptors],
                )
                counter += len(feat.descriptors)
            elif isinstance(specs[feat.key], MolFeatures):
                assert isinstance(feat, MolecularInput)
                descriptor_names = specs[feat.key].get_descriptor_names()  # type: ignore
                features2idx[feat.key] = tuple(
                    (np.array(range(len(descriptor_names))) + counter).tolist(),
                )
                features2names[feat.key] = tuple(
                    [get_encoded_name(feat.key, d) for d in descriptor_names],
                )
                counter += len(descriptor_names)
        return features2idx, features2names

    def transform(
        self,
        experiments: pd.DataFrame,
        specs: InputTransformSpecs,
    ) -> pd.DataFrame:
        """Transform a dataframe to the representation specified in `specs`.

        Currently only input categoricals are supported.

        Args:
            experiments (pd.DataFrame): Data dataframe to be transformed.
            specs (InputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.

        Returns:
            pd.DataFrame: Transformed dataframe. Only input features are included.

        """
        # TODO: clean this up and move it into the individual classes
        specs = self._validate_transform_specs(specs)
        transformed = []
        for feat in self.get():
            s = experiments[feat.key]
            if feat.key not in specs.keys():
                transformed.append(s)
            elif specs[feat.key] == CategoricalEncodingEnum.ONE_HOT:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.to_onehot_encoding(s))
            elif specs[feat.key] == CategoricalEncodingEnum.ORDINAL:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.to_ordinal_encoding(s))
            elif specs[feat.key] == CategoricalEncodingEnum.DUMMY:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.to_dummy_encoding(s))
            elif specs[feat.key] == CategoricalEncodingEnum.DESCRIPTOR:
                assert isinstance(feat, CategoricalDescriptorInput)
                transformed.append(feat.to_descriptor_encoding(s))
            elif isinstance(specs[feat.key], MolFeatures):
                assert isinstance(feat, MolecularInput)
                transformed.append(feat.to_descriptor_encoding(specs[feat.key], s))  # type: ignore
        return pd.concat(transformed, axis=1)

    def inverse_transform(
        self,
        experiments: pd.DataFrame,
        specs: InputTransformSpecs,
    ) -> pd.DataFrame:
        """Transform a dataframe back to the original representations.

        The original applied transformation has to be provided via the specs dictionary.
        Currently only input categoricals are supported.

        Args:
            experiments (pd.DataFrame): Transformed data dataframe.
            specs (InputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.

        Returns:
            pd.DataFrame: Back transformed dataframe. Only input features are included.

        """
        # TODO: clean this up and move it into the individual classes
        self._validate_transform_specs(specs=specs)
        transformed = []
        for feat in self.get():
            if isinstance(feat, DiscreteInput):
                transformed.append(feat.from_continuous(experiments))
            elif feat.key not in specs.keys():
                transformed.append(experiments[feat.key])
            elif specs[feat.key] == CategoricalEncodingEnum.ONE_HOT:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.from_onehot_encoding(experiments))
            elif specs[feat.key] == CategoricalEncodingEnum.ORDINAL:
                assert isinstance(feat, CategoricalInput)
                transformed.append(
                    feat.from_ordinal_encoding(experiments[feat.key].astype(int)),
                )
            elif specs[feat.key] == CategoricalEncodingEnum.DUMMY:
                assert isinstance(feat, CategoricalInput)
                transformed.append(feat.from_dummy_encoding(experiments))
            elif specs[feat.key] == CategoricalEncodingEnum.DESCRIPTOR:
                assert isinstance(feat, CategoricalDescriptorInput)
                transformed.append(feat.from_descriptor_encoding(experiments))
            elif isinstance(specs[feat.key], MolFeatures):
                assert isinstance(feat, CategoricalMolecularInput)
                transformed.append(
                    feat.from_descriptor_encoding(specs[feat.key], experiments),  # type: ignore
                )

        return pd.concat(transformed, axis=1)

    def _validate_transform_specs(
        self,
        specs: InputTransformSpecs,
    ) -> InputTransformSpecs:
        """Checks the validity of the transform specs .

        Args:
            specs (InputTransformSpecs): Transform specs to be validated.

        """
        # first check that the keys in the specs dict are correct also correct feature keys
        # next check that all values are of type CategoricalEncodingEnum or MolFeatures
        for key, value in specs.items():
            try:
                feat = self.get_by_key(key)
            except KeyError:
                raise ValueError(
                    f"Unknown feature with key {key} specified in transform specs.",
                )
            # TODO
            # this is ugly, on the long run we have to get rid of the transform enums
            # and replace them with classes, then the following lines collapse into just two
            assert isinstance(feat, Input)
            enums = [t for t in feat.valid_transform_types() if isinstance(t, Enum)]
            no_enums = [
                t for t in feat.valid_transform_types() if not isinstance(t, Enum)
            ]
            if isinstance(value, Enum):
                if value not in enums:
                    raise ValueError(
                        f"Forbidden transform type for feature with key {key}",
                    )
            else:
                if len(no_enums) == 0:
                    raise ValueError(
                        f"Forbidden transform type for feature with key {key}",
                    )
                if not isinstance(value, tuple(no_enums)):  # type: ignore
                    raise ValueError(
                        f"Forbidden transform type for feature with key {key}",
                    )

        return specs

    def get_bounds(
        self,
        specs: InputTransformSpecs,
        experiments: Optional[pd.DataFrame] = None,
        reference_experiment: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        """Returns the boundaries of the optimization problem based on the transformations
        defined in the  `specs` dictionary.

        Args:
            specs (InputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.
            experiments (Optional[pd.DataFrame], optional): Dataframe with input features.
                If provided the real feature bounds are returned based on both the opt.
                feature bounds and the extreme points in the dataframe. Defaults to None,
            reference_experiment (Optional[pd.Serues], optional): If a reference experiment provided,
            then the local bounds based on a local search region are provided as reference to the
                reference experiment. Currently only supported for continuous inputs.
                For more details, it is referred to https://www.merl.com/publications/docs/TR2023-057.pdf. Defaults to None.

        Raises:
            ValueError: If a feature type is not known.
            ValueError: If no transformation is provided for a categorical feature.

        Returns:
            Tuple[List[float], List[float]]: list with lower bounds, list with upper bounds.

        """
        if reference_experiment is not None and experiments is not None:
            raise ValueError(
                "Only one can be used, `reference_experiments` or `experiments`.",
            )

        self._validate_transform_specs(specs=specs)

        lower = []
        upper = []

        for feat in self.get():
            assert isinstance(feat, Input)
            lo, up = feat.get_bounds(
                transform_type=specs.get(feat.key),  # type: ignore
                values=experiments[feat.key] if experiments is not None else None,  # type: ignore
                reference_value=(
                    reference_experiment[feat.key]
                    if reference_experiment is not None
                    else None
                ),
            )
            lower += lo
            upper += up
        return lower, upper

    def get_feature_indices(
        self,
        specs: InputTransformSpecs,
        feature_keys: List[str],
    ) -> List[int]:
        """Returns a list of indices of the given feature key list.

        Args:
            specs (InputTransformSpecs): Dictionary specifying which
                input feature is transformed by which encoder.
            feature_keys (List[str]): List of feature keys.

        Returns:
            List[int]: The list of indices.

        """
        features2idx, _ = self._get_transform_info(specs)
        return sorted(
            itertools.chain.from_iterable(
                [features2idx[feat] for feat in feature_keys]
            ),
        )

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        """Check if the provided experiments fulfill all constraints defined on the
        input features itself like the bounds or the allowed categories.

        Args:
            experiments: Dataframe with input features.

        Returns:
            Series with boolean values indicating if the experiments fulfill the
                constraints on the input features.

        """
        return (
            pd.concat(
                [feat.is_fulfilled(experiments[feat.key]) for feat in self.get()],
                axis=1,
            )
            .fillna(True)
            .all(axis=1)
        )


class Outputs(_BaseFeatures[AnyOutput]):
    """Container of output features, only output features are allowed.

    Attributes:
        features (List(Outputs)): list of the features.

    """

    type: Literal["Outputs"] = "Outputs"  # type: ignore

    def get_by_objective(
        self,
        includes: Union[
            List[Type[AbstractObjective]],
            Type[AbstractObjective],
            Type[Objective],
        ] = Objective,
        excludes: Union[
            List[Type[AbstractObjective]],
            Type[AbstractObjective],
            None,
        ] = None,
        exact: bool = False,
    ) -> Outputs:
        """Get output features filtered by the type of the attached objective.

        Args:
            includes (Union[List[TObjective], TObjective], optional): Objective class or list of objective classes
                to be returned. Defaults to Objective.
            excludes (Union[List[TObjective], TObjective, None], optional): Objective class or list of specific objective classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact classes listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[AnyOutput]: List of output features fitting to the passed requirements.

        """
        if len(self.features) == 0:
            return Outputs(features=[])
        return Outputs(
            features=sorted(
                filter_by_attribute(
                    self.get([ContinuousOutput, CategoricalOutput]).features,
                    lambda of: of.objective,
                    includes,
                    excludes,
                    exact,
                ),
            ),
        )

    def get_keys_by_objective(
        self,
        includes: Union[
            List[Type[AbstractObjective]],
            Type[AbstractObjective],
            Type[Objective],
        ] = Objective,
        excludes: Union[
            List[Type[AbstractObjective]],
            Type[AbstractObjective],
            None,
        ] = None,
        exact: bool = False,
    ) -> List[str]:
        """Get keys of output features filtered by the type of the attached objective.

        Args:
            includes (Union[List[TObjective], TObjective], optional): Objective class or list of objective classes
                to be returned. Defaults to Objective.
            excludes (Union[List[TObjective], TObjective, None], optional): Objective class or list of specific objective classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact classes listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[str]: List of output feature keys fitting to the passed requirements.

        """
        return [f.key for f in self.get_by_objective(includes, excludes, exact)]

    def __call__(
        self,
        experiments: pd.DataFrame,
        experiments_adapt: Optional[pd.DataFrame] = None,
        predictions: bool = False,
    ) -> pd.DataFrame:
        """Evaluate the objective for every feature.

        Args:
            experiments (pd.DataFrame): Experiments for which the objectives
                should be evaluated.
            experiments_adapt (pd.DataFrame, optional): Experimental values
                which are used to update the objective parameters on the fly.
                This is for example needed when a `MovingMaximizeSigmoidObjective`
                is used as this depends on the best experimental value achieved
                so far. For this reason `experiments_adapt` has to be provided
                if `predictions=True` ie. that the objectives of candidates
                are evaluated. Defaults to None.
            predictions (bool, optional): If True use the prediction columns in
                the dataframe to calc the desirabilities `f"{feat.key}_pred`,
                furthermore `experiments_adapt` has to be provided.

        Returns:
            pd.DataFrame: Objective values for the experiments of interest.

        """
        if predictions and experiments_adapt is None:
            raise ValueError(
                "If predictions are used, `experiments_adapt` has to be provided.",
            )
        else:
            experiments_adapt = (
                experiments if experiments_adapt is None else experiments_adapt
            )

        desis = pd.concat(
            [
                feat(
                    experiments[f"{feat.key}_pred" if predictions else feat.key],
                    experiments_adapt[feat.key].dropna(),  # type: ignore
                )
                for feat in self.features
                if feat.objective is not None
                and not isinstance(feat, CategoricalOutput)
            ]
            + [
                (
                    pd.Series(  # type: ignore
                        data=feat(
                            experiments.filter(regex=f"{feat.key}(.*)_prob"),  # type: ignore
                            experiments.filter(regex=f"{feat.key}(.*)_prob"),  # type: ignore
                        ),
                        name=f"{feat.key}_pred",
                    )
                    if predictions
                    else experiments[feat.key]
                )
                for feat in self.features
                if feat.objective is not None and isinstance(feat, CategoricalOutput)
            ],
            axis=1,
        )
        return desis.rename(
            {
                f"{feat.key}_pred" if predictions else feat.key: f"{feat.key}_des"
                for feat in self.features
                if feat.objective is not None
            },
            axis=1,
        )

    def add_valid_columns(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Add the `valid_{feature.key}` columns to the experiments dataframe,
        in case that they are not present.

        Args:
            experiments (pd.DataFrame): Dataframe holding the experiments.

        Returns:
            pd.DataFrame: Dataframe holding the experiments.

        """
        valid_keys = [
            f"valid_{output_feature_key}" for output_feature_key in self.get_keys()
        ]
        for valid_key in valid_keys:
            if valid_key not in experiments:
                experiments[valid_key] = True
            else:
                try:
                    experiments[valid_key] = (
                        experiments[valid_key].astype(int).astype(bool)
                    )
                except ValueError:
                    raise ValueError(f"Column {valid_key} cannot casted to dtype bool.")
        return experiments

    def validate_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        for feat in self.get():
            if feat.key not in experiments:
                raise ValueError(f"no col for output feature `{feat.key}`")
            experiments[feat.key] = feat.validate_experimental(experiments[feat.key])
        experiments = self.add_valid_columns(experiments=experiments)
        return experiments

    def validate_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        # for each continuous output feature with an attached objective object
        continuous_cols = list(
            itertools.chain.from_iterable(
                [
                    [f"{feat.key}_pred", f"{feat.key}_sd", f"{feat.key}_des"]
                    for feat in self.get_by_objective(
                        includes=Objective,
                        excludes=ConstrainedCategoricalObjective,
                    )
                ]
                + [
                    [f"{key}_pred", f"{key}_sd"]
                    for key in self.get_keys_by_objective(
                        excludes=Objective,
                        includes=None,  # type: ignore
                    )
                ],
            ),
        )
        # check that pred, sd, and des cols are specified and numerical
        for col in continuous_cols:
            if col not in candidates:
                raise ValueError(f"missing column {col}")
            try:
                candidates[col] = pd.to_numeric(candidates[col], errors="raise").astype(
                    "float64",
                )
            except ValueError:
                raise ValueError(f"Not all values of column `{col}` are numerical.")
            if candidates[col].isnull().to_numpy().any():
                raise ValueError(f"Nan values are present in {col}.")
        # Looping over features allows to check categories objective wise
        for feat in self.get(CategoricalOutput):
            cols = [f"{feat.key}_pred", f"{feat.key}_des"]
            for col in cols:
                if col not in candidates:
                    raise ValueError(f"missing column {col}")
                if col == f"{feat.key}_pred":
                    feat.validate_experimental(candidates[col])
                # Check sd and desirability
                elif candidates[col].isnull().to_numpy().any():
                    raise ValueError(f"Nan values are present in {col}.")
        return candidates

    def preprocess_experiments_one_valid_output(
        self,
        output_feature_key: str,
        experiments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Method to get a dataframe where non-valid entries of the provided output feature are removed

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data
            output_feature_key (str): The feature based on which non-valid entries rows are removed

        Returns:
            pd.DataFrame: Dataframe with all experiments where only valid entries of the specific feature are included

        """
        clean_exp = experiments.loc[
            (experiments["valid_%s" % output_feature_key] == 1)
            & (experiments[output_feature_key].notna())
        ]

        return clean_exp

    def preprocess_experiments_all_valid_outputs(
        self,
        experiments: pd.DataFrame,
        output_feature_keys: Optional[List] = None,
    ) -> pd.DataFrame:
        """Method to get a dataframe where non-valid entries of all output feature are removed

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data
            output_feature_keys (Optional[List], optional): List of output feature keys which should be considered for removal of invalid values. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with all experiments where only valid entries of the selected features are included

        """
        if (output_feature_keys is None) or (len(output_feature_keys) == 0):
            output_feature_keys = self.get_keys(Output)

        clean_exp = experiments.query(
            " & ".join(["(`valid_%s` > 0)" % key for key in output_feature_keys]),
        )
        clean_exp = clean_exp.dropna(subset=output_feature_keys)

        return clean_exp

    def preprocess_experiments_any_valid_output(
        self,
        experiments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Method to get a dataframe where at least one output feature has a valid entry

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data

        Returns:
            pd.DataFrame: Dataframe with all experiments where at least one output feature has a valid entry

        """
        output_feature_keys = self.get_keys(Output)

        # clean_exp = experiments.query(" or ".join(["(valid_%s > 0)" % key for key in output_feature_keys]))
        # clean_exp = clean_exp.query(" or ".join(["%s.notna()" % key for key in output_feature_keys]))

        assert experiments is not None
        clean_exp = experiments.query(
            " or ".join(
                [
                    "((`valid_%s` >0) & `%s`.notna())" % (key, key)
                    for key in output_feature_keys
                ],
            ),
        )
        return clean_exp
