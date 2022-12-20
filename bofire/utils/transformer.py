import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from bofire.domain.domain import Domain, DomainError
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    InputFeature,
    OutputFeature,
    is_continuous,
)
from bofire.utils.categoricalDescriptorEncoder import CategoricalDescriptorEncoder
from bofire.utils.enum import (
    CategoricalEncodingEnum,
    DescriptorEncodingEnum,
    ScalerEnum,
)

# with warnings.catch_warnings():
#    warnings.simplefilter("ignore")

# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning, append=True)


class Transformer(BaseModel):
    domain: Domain
    descriptor_encoding: Optional[Union[DescriptorEncodingEnum, None]]
    categorical_encoding: Optional[Union[CategoricalEncodingEnum, None]]
    scale_inputs: Optional[ScalerEnum] = None
    scale_outputs: Optional[ScalerEnum] = None
    is_fitted: bool = False
    features2transformedFeatures: Dict = Field(default_factory=lambda: {})
    encoders: Dict = Field(default_factory=lambda: {})
    """Pre/post-processing of data for strategies

    Parameters:
        is_fitted: bool
        features2transformedFeatures: Dict
        encoders: Dict

    """

    def __init__(
        self,
        domain,
        descriptor_encoding=None,  # TODO: default tbd!
        categorical_encoding=None,  # TODO: default tbd!
        scale_inputs=None,
        scale_outputs=None,
    ) -> None:
        """Pre/post-processing of data for strategies

        Args:
            domain (Domain):
                    A domain for that is being used in the strategy
            descriptor_encoding (DescriptorEncodingEnum, optional):
                    Transform the descriptors into continuous features ("DESCRIPTOR")/
                    numerical representation of categoricals ("CATEGORICAL"). Defaults to None.
            categorical_encoding (CategoricalEncodingEnum, optional):
                Distinction between one hot and ordinal encoding for categoricals.
                Defaults to None.
            scale_inputs/ scale_outputs (ScalerEnum, optional):
                In-/Outputs can be standardized, normalized or not scaled. Defaults to None.
        """

        super().__init__(
            domain=domain,
            descriptor_encoding=descriptor_encoding,
            # (
            #     descriptor_encoding.name
            #     if isinstance(descriptor_encoding, Enum)
            #     else None
            # ),
            categorical_encoding=categorical_encoding,
            # (
            #     categorical_encoding.name
            #     if isinstance(categorical_encoding, Enum)
            #     else None
            # ),
            scale_inputs=scale_inputs,
            scale_outputs=scale_outputs,
        )

        for feature in self.get_features_to_be_transformed():
            if (
                isinstance(feature, CategoricalDescriptorInput)
                and self.descriptor_encoding == DescriptorEncodingEnum.DESCRIPTOR
            ):

                keys = [feature.key + "_" + str(t) for t in feature.descriptors]
                self.features2transformedFeatures[feature.key] = keys

            elif (
                isinstance(feature, CategoricalInput)
                and self.categorical_encoding == CategoricalEncodingEnum.ONE_HOT
            ):
                keys = [feature.key + "_" + str(t) for t in feature.categories]
                self.features2transformedFeatures[feature.key] = keys

        return

    def fit(self, input: pd.DataFrame):
        """Fit the encoders to provided input data

        Args:
            input (pd.DataFrame): The data, the encoders are fitted to.

        Raises:
            NotImplementedError:
                User-provided descriptor values are currently ignored.
                The descriptor values are set in the CategoricalDescriptorInputFeature.
            DomainError: Unknown input feature type. Features have to be of type continuous or categorical feature.
            DomainError: Output features cannot be categorical features currently.
            DomainError: Unknown output feature type

        Returns:
            transformer object: The fitted transformer
        """

        experiment = input.copy()

        for feature in self.get_features_to_be_transformed():
            if (
                isinstance(feature, CategoricalDescriptorInput)
                and self.descriptor_encoding == DescriptorEncodingEnum.DESCRIPTOR
            ):

                if all(np.isin(feature.descriptors, experiment.columns)):
                    # how should we deal with descriptors entered by the user, which are not in line with the values assigned in the features`?
                    # TODO: Check if user-provided decriptors are valid
                    # TODO: implement hierarchy which descriptors are preferred
                    raise NotImplementedError(
                        "User-provided descriptor values are currently ignored"
                    )

                enc = CategoricalDescriptorEncoder(
                    categories=[feature.categories],
                    descriptors=[feature.descriptors],
                    values=[feature.values],
                )
                values = pd.DataFrame(experiment[feature.key], columns=[feature.key])
                enc.fit(values)
                column_names = enc.get_feature_names_out()

                self.encoders[feature.key] = enc
                values = self.encoders[feature.key].transform(values)

                for loc, column_name in enumerate(column_names):

                    experiment[column_name] = values[:, loc]
                    var_min = min([val[loc] for val in feature.values])
                    var_max = max([val[loc] for val in feature.values])

                    # Scale inputs
                    self.fit_scaling(
                        column_name,
                        experiment,
                        var_min,
                        var_max,
                        scaler_type=self.scale_inputs,
                    )
                    experiment = experiment.drop(column_name, axis=1)

            elif (
                isinstance(feature, CategoricalInput)
                and self.categorical_encoding == CategoricalEncodingEnum.ORDINAL
            ):
                enc = OrdinalEncoder(categories=[feature.categories])  # type: ignore

                values = pd.DataFrame(experiment[feature.key], columns=[feature.key])
                enc.fit(values)

                self.encoders[feature.key] = enc

            elif (
                isinstance(feature, CategoricalInput)
                and self.categorical_encoding == CategoricalEncodingEnum.ONE_HOT
            ):
                # Create one-hot encoding columns & insert to DataSet
                # TODO: drop in oneHot encoder testen
                enc = OneHotEncoder(categories=[feature.categories])  # type: ignore

                values = pd.DataFrame(experiment[feature.key], columns=[feature.key])
                enc.fit(values)

                self.encoders[feature.key] = enc

            elif isinstance(feature, ContinuousInput):
                var_min, var_max = feature.lower_bound, feature.upper_bound
                self.fit_scaling(
                    feature.key,
                    experiment,
                    var_min,
                    var_max,
                    scaler_type=self.scale_inputs,
                )

            elif (
                self.categorical_encoding is None
                and self.descriptor_encoding is not None
            ):
                logging.warning(
                    "Descriptors should be encoded as categoricals. However, categoricals are selected to be not transformed. Thus, I will skip categoricals with descriptors as well."
                )
                pass

            else:
                raise DomainError(
                    f"Feature {feature.key} is not a continuous or categorical feature."
                )

        for feature in self.domain.get_features(OutputFeature):

            if isinstance(feature, OutputFeature):
                if not is_continuous(feature):
                    raise DomainError(
                        "Output features cannot be categorical features currently."
                    )

                self.fit_scaling(
                    feature.key, experiment, scaler_type=self.scale_outputs
                )

            else:
                raise DomainError(f"Feature {feature.key} is not in the dataset.")

        self.is_fitted = True
        return self

    def transform(self, experiment: pd.DataFrame):
        """Transform data inputs and outputs for a strategy given already fitted encoders

        Args:
            experiment (pd.DataFrame): Input data to be transformed

        Raises:
            DomainError:
                Unknown input feature type. Features have to be of type continuous or categorical feature.
            DomainError:
                Output features cannot be categorical features currently.
            DomainError:
                Unknown output feature type

        Returns:
            transformed_experiment (pd.DataFrame): the transformed input data
        """

        assert self.is_fitted is True, "Encoders are not initialized"
        transformed_experiment = experiment.copy()

        for feature in self.get_features_to_be_transformed():
            if (
                isinstance(feature, CategoricalDescriptorInput)
                and self.descriptor_encoding == DescriptorEncodingEnum.DESCRIPTOR
            ):

                values = pd.DataFrame(
                    transformed_experiment[feature.key], columns=[feature.key]
                )
                values = self.encoders[feature.key].transform(values)
                column_names = self.encoders[feature.key].get_feature_names_out()

                index = transformed_experiment.columns.get_loc(feature.key)
                for i, column_name in enumerate(column_names):
                    transformed_experiment.insert(index + i, column_name, values[:, i])

                    # Scale inputs
                    if self.encoders[column_name] is not None:
                        values_unscaled = np.atleast_2d(
                            transformed_experiment[column_name].to_numpy()
                        ).T
                        transformed_experiment[column_name] = self.encoders[
                            column_name
                        ].transform(values_unscaled)

                # Ensure descriptor features are floats
                transformed_experiment[column_names] = transformed_experiment[
                    column_names
                ].astype(float)

                # drop categorical column
                transformed_experiment = transformed_experiment.drop(
                    feature.key, axis=1
                )

            elif (
                isinstance(feature, CategoricalInput)
                and self.categorical_encoding == CategoricalEncodingEnum.ORDINAL
            ):

                values = pd.DataFrame(
                    transformed_experiment[feature.key], columns=[feature.key]
                )
                enc_values = self.encoders[feature.key].transform(values)

                transformed_experiment[feature.key] = enc_values.astype(
                    "int32"
                )  # categorical kernel needs int as input to avoid numerical trouble. Thus, these columns are also not scaled

            elif (
                isinstance(feature, CategoricalInput)
                and self.categorical_encoding == CategoricalEncodingEnum.ONE_HOT
            ):

                values = pd.DataFrame(
                    transformed_experiment[feature.key], columns=[feature.key]
                )
                enc_values = self.encoders[feature.key].transform(values).toarray()

                column_names = self.encoders[feature.key].get_feature_names_out()

                index = transformed_experiment.columns.get_loc(feature.key)
                for i, column_name in enumerate(column_names):
                    transformed_experiment.insert(
                        index + i, column_name, enc_values[:, i]
                    )

                # Drop old categorical column
                transformed_experiment = transformed_experiment.drop(
                    feature.key, axis=1
                )

            elif isinstance(feature, ContinuousInput):

                if self.encoders[feature.key] is not None:
                    values = np.atleast_2d(
                        transformed_experiment[feature.key].to_numpy()
                    ).T
                    transformed_experiment[feature.key] = self.encoders[
                        feature.key
                    ].transform(values)

            elif (
                self.categorical_encoding is None
                and self.descriptor_encoding is not None
            ):
                logging.warning(
                    "Descriptors should be encoded as categoricals. However, categoricals are selected to be not transformed. Thus, I will skip categoricals with descriptors as well."
                )
                pass

            else:
                raise DomainError(
                    f"Feature {feature.key} is not a continuous or categorical feature."
                )

        for feature in self.domain.get_features(OutputFeature):

            if isinstance(feature, OutputFeature):
                if not is_continuous(feature):
                    raise DomainError(
                        "Output features cannot be categorical features currently."
                    )

                if self.encoders[feature.key] is not None:
                    values = np.atleast_2d(
                        transformed_experiment[feature.key].to_numpy()
                    ).T
                    transformed_experiment[feature.key] = self.encoders[
                        feature.key
                    ].transform(values)

            else:
                raise DomainError(f"Feature {feature.key} is not in the dataset.")

        return transformed_experiment.copy()

    def fit_transform(self, experiment: pd.DataFrame):
        """A combination of self.fit and self.transform

        Args:
            experiment (pd.DataFrame): [description]

        Returns:
            transformed_experiment (pd.DataFrame): the transfered input data
        """
        self.fit(experiment)
        transformed_experiment = self.transform(experiment)

        return transformed_experiment

    def inverse_transform(self, transformed_candidate: pd.DataFrame):
        """backtransformation of data using the fitted encoders

        Args:
            transformed_candidate (pd.DataFrame): input data to be backtransformed

        Raises:
            DomainError:
                Feature is defined in the domain, but no feature data is provided in the dataset to be backtransformed.
            NotImplementedError:
                Acctually, categorical outputs are not supported.

        Returns:
            candidate (pd.DataFrame): backtransformed input data
        """

        assert self.is_fitted is True, "Encoders are not initialized"

        # Determine input and output columns in dataset
        candidate = transformed_candidate.copy()
        for feature in self.get_features_to_be_transformed():
            # Categorical variables with descriptors
            if (
                isinstance(feature, CategoricalDescriptorInput)
                and self.descriptor_encoding == DescriptorEncodingEnum.DESCRIPTOR
            ):
                enc = self.encoders[feature.key]
                column_names = enc.get_feature_names_out()

                for column_name in column_names:
                    candidate = self.un_scale(column_name, candidate)

                values = candidate[column_names].to_numpy()
                enc_values = enc.inverse_transform(values)

                index = candidate.columns.get_loc(column_names[0])
                candidate.insert(index, feature.key, enc_values)

                # Delete the descriptor columns
                candidate = candidate.drop(columns=column_names, axis=1)

            # Categorical features using one-hot encoding
            elif (
                isinstance(feature, CategoricalInput)
                and self.categorical_encoding == CategoricalEncodingEnum.ONE_HOT
            ):
                # Get encoder
                enc = self.encoders[feature.key]

                # Get array to be transformed
                column_names = enc.get_feature_names_out()
                values = candidate[column_names].to_numpy()

                # Do inverse transform
                index = candidate.columns.get_loc(column_names[0])
                enc_values = enc.inverse_transform(values)
                candidate.insert(index, feature.key, enc_values)

                # Add to dataset and drop one-hot encoding
                candidate = candidate.drop(column_names, axis=1)

            # Categorical features using ordinal encoding
            elif (
                isinstance(feature, CategoricalInput)
                and self.categorical_encoding == CategoricalEncodingEnum.ORDINAL
            ):
                # Get encoder
                enc = self.encoders[feature.key]

                # Get array to be transformed
                values = np.atleast_2d(candidate[feature.key].to_numpy()).T

                # Do inverse transform
                candidate[feature.key] = enc.inverse_transform(values)

            # Continuous features
            elif isinstance(feature, ContinuousInput):
                candidate = self.un_scale(feature.key, candidate)

            elif (
                self.categorical_encoding is None
                and self.descriptor_encoding is not None
            ):
                pass

            else:
                raise DomainError(f"Feature {feature.key} is not in the dataset.")

        for feature in self.domain.get_features(OutputFeature):

            if feature.key in transformed_candidate.columns:
                if isinstance(feature, ContinuousOutput):
                    candidate = self.un_scale(feature.key, candidate)
                else:
                    raise NotImplementedError(
                        "Acctually, only continuous outputs are implemented"
                    )
        return candidate

    def get_features_to_be_transformed(self):

        excludes = []
        if self.categorical_encoding is None:
            excludes.append(CategoricalInput)
        if self.descriptor_encoding is None:
            excludes.append(CategoricalDescriptorInput)

        return self.domain.get_features(InputFeature, excludes=excludes)

    def fit_scaling(
        self,
        key: str,
        experiment: pd.DataFrame,
        var_min: float = np.nan,
        var_max: float = np.nan,
        scaler_type: Optional[ScalerEnum] = None,
    ) -> "Transformer":  # TODO: switch to Self some time in the future which is available in Python 3.11
        """fitting the chosen scaler type to provided input data

        Args:
            key (str):
                column name in input data of the feature to be scaled
            experiment (pd.DataFrame):
                input data to be fitted to
            var_min (float, optional):
                Lower bound to be used for min max scaling.
                When not defined or nan, the minimum of the input data is used. Defaults to np.nan.
            var_max (float, optional):
                Upper bound to be used for min max scaling.
                When not defined or nan, the maximum of the input data is used. Defaults to np.nan.
            scaler_type (ScalerEnum, optional):
                Defines the type of scaling (Normalize/ standardize/ None). Defaults to None.

        Returns:
            transformer object: fitted encoders are added to self.encoders
        """
        values = np.atleast_2d(experiment[key].to_numpy()).T

        if scaler_type == ScalerEnum.STANDARDIZE:
            enc = StandardScaler()
            enc.fit(values)

        elif scaler_type == ScalerEnum.NORMALIZE:
            enc = MinMaxScaler()
            if np.isnan(var_min):
                var_min = min(values)
            if np.isnan(var_max):
                var_max = max(values)

            enc.fit(np.array([var_min, var_max]).reshape(-1, 1))

        else:
            enc = None

        self.encoders[key] = enc

        return self

    def un_scale(self, key: str, candidate: pd.DataFrame) -> pd.DataFrame:
        """uses the fitted encoders to back-scale the input data to original scale

        Args:
            key (str):
                column name in input data of the feature to be back-scaled
            candidate (pd.DataFrame):
                input data to be un-scaled

        Returns:
            candidate (pd.DataFrame):
                The un-scaled input data
        """

        if (key in self.encoders.keys()) and (self.encoders[key] is not None):
            values = np.atleast_2d(candidate[key].to_numpy()).T
            candidate[key] = self.encoders[key].inverse_transform(values)

        candidate[key] = candidate[key].astype(float)

        return candidate
