import itertools
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from bofire.domain.constraints import Constraint, LinearConstraint, NChooseKConstraint
from bofire.domain.features import (
    CategoricalInputFeature,
    ContinuousInputFeature,
    ContinuousOutputFeature,
    DesirabilityOutputFeature,
    Feature,
    InputFeature,
    OutputFeature,
)
from bofire.domain.util import BaseModel, filter_by_class, is_categorical, is_numeric
from pydantic import Field, validator


class Domain(BaseModel):

    input_features: Optional[List[InputFeature]] = Field(default_factory=lambda: [])
    output_features: Optional[List[OutputFeature]] = Field(default_factory=lambda: [])
    constraints: Optional[List[Constraint]] = Field(default_factory=lambda: [])
    experiments: Optional[pd.DataFrame]
    candidates: Optional[pd.DataFrame]
    """Representation of the optimization problem/domain

    Attributes:
        input_features (List[InputFeature], optional): List of input features. Defaults to [].
        output_features (List[OutputFeature], optional): List of output features. Defaults to [].
        constraints (List[Constraint], optional): List of constraints. Defaults to [].
    """

    @validator("output_features", always=True)
    def validate_unique_output_feature_keys(cls, v, values):
        """Validates if provided output feature keys are unique

        Args:
            v (List[OutputFeature]): List of all output features of the domain.
            values (List[InputFeature]): Dict containing a list of input features as single entry.

        Raises:
            ValueError: Feature keys are not unique.

        Returns:
            List[OutputFeature]: Returns the list of output features when no error is thrown.
        """
        if "input_features" not in values:
            return v
        features = v + values["input_features"]
        keys = [f.key for f in features]
        if len(set(keys)) != len(keys):
            raise ValueError("feature keys are not unique")
        return v

    @validator("constraints", always=True)
    def validate_constraints(cls, v, values):
        """Validate if all features included in the constraints are also defined as features for the domain.

        Args:
            v (List[Constraint]): List of constraints or empty if no constraints are defined
            values (List[InputFeature]): List of input features of the domain

        Raises:
            ValueError: Feature key in constraint is unknown.

        Returns:
            List[Constraint]: List of constraints defined for the domain
        """
        if "input_features" not in values:
            return v
        keys = [f.key for f in values["input_features"]]
        for c in v:
            if isinstance(c, LinearConstraint) or isinstance(c, NChooseKConstraint):
                for f in c.features:
                    if f not in keys:
                        raise ValueError(f"feature {f} in constraint unknown ({keys})")
        return v

    @validator("constraints", always=True)
    def validate_lower_bounds_in_nchoosek_constraints(cls, v, values):
        """Validate the lower bound as well if the chosen number of allowed features is continuous.

        Args:
            v (List[Constraint]): List of all constraints defined for the domain
            values (List[InputFeature]): _description_

        Returns:
            List[Constraint]: List of constraints defined for the domain
        """
        # gather continuous input_features in dictionary
        continuous_input_features_dict = {}
        for f in values["input_features"]:
            if type(f) is ContinuousInputFeature:
                continuous_input_features_dict[f.key] = f

        # check if unfixed continuous features appearing in NChooseK constraints have lower bound of 0
        for c in v:
            if isinstance(c, NChooseKConstraint):
                for f in c.features:
                    assert (
                        f in continuous_input_features_dict
                    ), f"{f} must be continuous."
                    assert (
                        continuous_input_features_dict[f].lower_bound == 0
                    ), f"lower bound of {f} must be 0 for NChooseK constraint."
        return v

    def to_config(self) -> Dict:
        """Serializables itself to a dictionary.

        Returns:
            Dict: Serialized version of the domain as dictionary.
        """
        config = {
            "input_features": [feat.to_config() for feat in self.input_features],
            "output_features": [feat.to_config() for feat in self.output_features],
            "constraints": [constraint.to_config() for constraint in self.constraints],
        }
        if self.experiments is not None:
            config["experiments"] = self.experiments.to_dict()
        if self.candidates is not None:
            config["candidates"] = self.candidates.to_dict()
        return config

    @classmethod
    def from_config(cls, config: Dict):
        """Instantiates a `Domain` object from a dictionary created by the `to_config`method.

        Args:
            config (Dict): Serialized version of a domain as dictionary.
        """
        d = cls(
            input_features=[
                Feature.from_config(feat) for feat in config["input_features"]
            ],
            output_features=[
                Feature.from_config(feat) for feat in config["output_features"]
            ],
            constraints=[
                Constraint.from_config(constraint)
                for constraint in config["constraints"]
            ],
        )
        if "experiments" in config.keys():
            d.add_experiments(experiments=config["experiments"])
        if "candidates" in config.keys():
            d.add_candidates(experiments=config["candidates"])
        return d

    def get_feature_reps_df(self) -> pd.DataFrame:
        """Returns a pandas dataframe describing the features contained in the optimization domain."""
        df = pd.DataFrame(
            index=self.get_feature_keys(Feature),
            columns=["Type", "Description"],
            data={
                "Type": [
                    feat.__class__.__name__ for feat in self.get_features(Feature)
                ],
                "Description": [feat.__str__() for feat in self.get_features(Feature)],
            },
        )
        return df

    def get_constraint_reps_df(self):
        """Provides a tabular overwiev of all constraints within the domain

        Returns:
            pd.DataFrame: DataFrame listing all constraints of the domain with a description
        """
        df = pd.DataFrame(
            index=range(len(self.get_constraints())),
            columns=["Type", "Description"],
            data={
                "Type": [feat.__class__.__name__ for feat in self.get_constraints()],
                "Description": [
                    constraint.__str__() for constraint in self.get_constraints()
                ],
            },
        )
        return df

    def get_constraints(
        self,
        includes: Union[Type, List[Type]] = Constraint,
        excludes: Union[Type, List[Type]] = None,
        exact: bool = False,
    ) -> List[Constraint]:
        """get constraints of the domain

        Args:
            includes (Union[Constraint, List[Constraint]], optional): Constraint class or list of specific constraint classes to be returned. Defaults to Constraint.
            excludes (Union[Type, List[Type]], optional): Constraint class or list of specific constraint classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[Constraint]: List of constraints in the domain fitting to the passed requirements.
        """
        return filter_by_class(
            self.constraints,
            includes=includes,
            excludes=excludes,
            exact=exact,
        )

    def get_features(
        self,
        includes: Union[Type, List[Type]] = Feature,
        excludes: Union[Type, List[Type]] = None,
        exact: bool = False,
    ) -> List[Feature]:
        """get features of the domain

        Args:
            includes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be returned. Defaults to Feature.
            excludes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[Feature]: List of features in the domain fitting to the passed requirements.
        """
        return list(
            sorted(
                filter_by_class(
                    self.input_features + self.output_features,
                    includes=includes,
                    excludes=excludes,
                    exact=exact,
                )
            )
        )

    def get_feature_keys(
        self,
        includes: Union[Type, List[Type]] = Feature,
        excludes: Union[Type, List[Type]] = None,
        exact: bool = False,
    ) -> List[str]:
        """Method to get feature keys of the domain

        Args:
            includes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be returned. Defaults to Feature.
            excludes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.

        Returns:
            List[str]: List of feature keys fitting to the passed requirements.
        """
        return [
            f.key
            for f in self.get_features(
                includes=includes,
                excludes=excludes,
                exact=exact,
            )
        ]

    def get_feature(self, key: str):
        """get a specific feature by its key

        Args:
            key (str): Feature key

        Returns:
            Feature: The feature with the passed key
        """
        return {f.key: f for f in self.input_features + self.output_features}[key]

    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the optimzation domain

        Args:
            constraint (Constraint): object of class Constraint, which is added to the list
        """
        self.constraints.append(constraint)

    def add_feature(self, feature: Feature) -> None:
        """add a feature to list domain.features

        Args:
            feature (Feature): object of class Feature, which is added to the list

        Raises:
            ValueError: if the feature key is already in the domain
            TypeError: if the feature type is neither Input nor Output feature
        """
        if (self.experiments is not None) or (self.candidates is not None):
            raise ValueError(
                "Feature cannot be added as experiments/candidates are already set."
            )
        if feature.key in self.get_feature_keys():
            raise ValueError(f"Feature with key {feature.key} already in domain.")
        if isinstance(feature, InputFeature):
            self.input_features.append(feature)
        elif isinstance(feature, OutputFeature):
            self.output_features.append(feature)
        else:
            raise TypeError(f"Cannot add feature of type {type(feature)}")

    def remove_feature_by_key(self, key):
        """removes a feature from domain indicated by its key

        Args:
            key (str): feature key

        Raises:
            KeyError: when the key is not found in the domain
            ValueError: when more than one feature with key is found
        """
        if (self.experiments is not None) or (self.candidates is not None):
            raise ValueError(
                f"Feature {key} cannot be removed as experiments/candidates are already set."
            )
        input_count = len([f for f in self.input_features if f.key == key])
        output_count = len([f for f in self.output_features if f.key == key])
        if input_count == 0 and output_count == 0:
            raise KeyError(f"no feature with key {key} found")
        if input_count + output_count > 1:
            raise ValueError(f"more than one feature with key {key} found")
        if input_count > 0:
            self.input_features = [f for f in self.input_features if f.key != key]
        if output_count > 0:
            self.output_features = [f for f in self.output_features if f.key != key]

    def get_categorical_combinations(
        self, include: Feature = InputFeature, exclude: Feature = None
    ):
        """get a list of tuples pairing the feature keys with a list of valid categories

        Args:
            include (Feature, optional): Features to be included. Defaults to InputFeature.
            exclude (Feature, optional): Features to be excluded, e.g. subclasses of the included features. Defaults to None.

        Returns:
            List[(str, List[str])]: Returns a list of tuples pairing the feature keys with a list of valid categories (str)
        """
        features = [
            f
            for f in self.get_features(includes=include, excludes=exclude)
            if isinstance(f, CategoricalInputFeature) and not f.is_fixed()
        ]
        list_of_lists = [
            [(f.key, cat) for cat in f.get_allowed_categories()] for f in features
        ]
        return list(itertools.product(*list_of_lists))

    # getting list of fixed values
    def get_nchoosek_combinations(self):
        """get all possible NChooseK combinations

        Returns:
            Tuple(used_features_list, unused_features_list): used_features_list is a list of lists containing features used in each NChooseK combination.
             unused_features_list is a list of lists containing features unused in each NChooseK combination.
        """

        if len(self.get_constraints(NChooseKConstraint)) == 0:
            used_continuous_features = self.get_feature_keys(ContinuousInputFeature)
            return used_continuous_features, []

        used_features_list_all = []

        # loops through each NChooseK constraint
        for con in self.get_constraints(NChooseKConstraint):
            used_features_list = []

            for n in range(con.min_count, con.max_count + 1):
                used_features_list.extend(itertools.combinations(con.features, n))

            if con.none_also_valid:
                used_features_list.append(tuple([]))

            used_features_list_all.append(used_features_list)

        used_features_list_all = list(
            itertools.product(*used_features_list_all)
        )  # product between NChooseK constraints

        # format into a list of used features
        used_features_list_formatted = []
        for used_features_list in used_features_list_all:

            used_features_list_flattened = [
                item for sublist in used_features_list for item in sublist
            ]
            used_features_list_formatted.append(list(set(used_features_list_flattened)))

        # sort lists
        used_features_list_sorted = []
        for used_features in used_features_list_formatted:
            used_features_list_sorted.append(sorted(used_features))

        # drop duplicates
        used_features_list_no_dup = []
        for used_features in used_features_list_sorted:
            if used_features not in used_features_list_no_dup:
                used_features_list_no_dup.append(used_features)

        # print(f"duplicates dropped: {len(used_features_list_sorted)-len(used_features_list_no_dup)}")

        # remove combinations not fulfilling constraints
        used_features_list_final = []
        for combo in used_features_list_no_dup:
            fulfil_constraints = (
                []
            )  # list of bools tracking if constraints are fulfilled
            for con in self.get_constraints(NChooseKConstraint):
                count = 0  # count of features in combo that are in con.features
                for f in combo:
                    if f in con.features:
                        count += 1
                if count >= con.min_count and count <= con.max_count:
                    fulfil_constraints.append(True)
                elif count == 0 and con.none_also_valid:
                    fulfil_constraints.append(True)
                else:
                    fulfil_constraints.append(False)
            if np.all(fulfil_constraints):
                used_features_list_final.append(combo)

        # print(f"violators dropped: {len(used_features_list_no_dup)-len(used_features_list_final)}")

        # features unused
        features_in_cc = []
        for con in self.get_constraints(NChooseKConstraint):
            features_in_cc.extend(con.features)
        features_in_cc = list(set(features_in_cc))
        features_in_cc.sort()
        unused_features_list = []
        for used_features in used_features_list_final:
            unused_features_list.append(
                [f_key for f_key in features_in_cc if f_key not in used_features]
            )

        # postprocess
        # used_features_list_final2 = []
        # unused_features_list2 = []
        # for used, unused in zip(used_features_list_final,unused_features_list):
        #     if len(used) == 3:
        #         used_features_list_final2.append(used), unused_features_list2.append(unused)

        return used_features_list_final, unused_features_list

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        """Method to check if all constraints are fulfilled on all rows of the provided dataframe

        Args:
            df_data (pd.DataFrame): Dataframe with data, the constraint validity should be tested on

        Returns:
            Boolean: True if all constraints are fulfilled for all rows, false if not
        """
        if len(self.constraints) == 0:
            return pd.Series([True] * len(experiments), index=experiments.index)
        return pd.concat(
            [c.satisfied(experiments) for c in self.constraints], axis=1
        ).all(axis=1)

    # TODO: needs to be tested
    def evaluate_constraints(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([c(experiments) for c in self.constraints], axis=1)

    # TODO: needs to be tested
    def evaluate_desirabilities(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                feat.desirability_function(experiments[feat.name])
                for feat in self.get_features(ContinuousOutputFeature)
            ],
            axis=1,
        )

    def preprocess_experiments_one_valid_output(
        self,
        output_feature_key: str,
        experiments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Method to get a dataframe where non-valid entries of the provided output feature are removed

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data
            output_feature_key (str): The feature based on which non-valid entries rows are removed

        Returns:
            pd.DataFrame: Dataframe with all experiments where only valid entries of the specific feature are included
        """
        if experiments is None:
            if self.experiments is not None:
                experiments = self.experiments
            else:
                return None
        clean_exp = experiments.loc[
            (experiments["valid_%s" % output_feature_key] == 1)
            & (experiments[output_feature_key].notna())
        ]
        # clean_exp = clean_exp.dropna()

        return clean_exp

    def preprocess_experiments_all_valid_outputs(
        self,
        experiments: Optional[pd.DataFrame] = None,
        output_feature_keys: Optional[List] = None,
    ) -> pd.DataFrame:
        """Method to get a dataframe where non-valid entries of all output feature are removed

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data
            output_feature_keys (Optional[List], optional): List of output feature keys which should be considered for removal of invalid values. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with all experiments where only valid entries of the selected features are included
        """
        if experiments is None:
            if self.experiments is not None:
                experiments = self.experiments
            else:
                return None
        if (output_feature_keys is None) or (len(output_feature_keys) == 0):
            output_feature_keys = self.get_feature_keys(OutputFeature)
        else:
            for key in output_feature_keys:
                feat = self.get_feature(key)
                assert isinstance(
                    feat, OutputFeature
                ), f"feat {key} is not an OutputFeature"

        clean_exp = experiments.query(
            " & ".join(["(`valid_%s` > 0)" % key for key in output_feature_keys])
        )
        clean_exp = clean_exp.dropna(subset=output_feature_keys)

        return clean_exp

    def preprocess_experiments_any_valid_output(
        self, experiments: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Method to get a dataframe where at least one output feature has a valid entry

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data

        Returns:
            pd.DataFrame: Dataframe with all experiments where at least one output feature has a valid entry
        """
        if experiments is None:
            if self.experiments is not None:
                experiments = self.experiments
            else:
                return None

        output_feature_keys = self.get_feature_keys(OutputFeature)

        # clean_exp = experiments.query(" or ".join(["(valid_%s > 0)" % key for key in output_feature_keys]))
        # clean_exp = clean_exp.query(" or ".join(["%s.notna()" % key for key in output_feature_keys]))

        clean_exp = experiments.query(
            " or ".join(
                [
                    "((`valid_%s` >0) & `%s`.notna())" % (key, key)
                    for key in output_feature_keys
                ]
            )
        )

        return clean_exp

    def coerce_invalids(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Coerces all invalid output measurements to np.nan

        Args:
            experiments (pd.DataFrame): Dataframe containing experimental data

        Returns:
            pd.DataFrame: coerced dataframe
        """
        # coerce invalid to nan
        for feat in self.get_feature_keys(OutputFeature):
            experiments.loc[experiments[f"valid_{feat}"] == 0, feat] = np.nan
        return experiments

    def aggregate_by_duplicates(
        self, experiments: pd.DataFrame, prec: int, delimiter: str = "-"
    ) -> Tuple[pd.DataFrame, list]:
        """Aggregate the dataframe by duplicate experiments

        Duplicates are identified based on the experiments with the same input features. Continuous input features
        are rounded before identifying the duplicates. Aggregation is performed by taking the average of the
        involved output features.

        Args:
            experiments (pd.DataFrame): Dataframe containing experimental data
            prec (int): Precision of the rounding of the continuous input features
            delimiter (str, optional): Delimiter used when combining the orig. labcodes to a new one. Defaults to "-".

        Returns:
            Tuple[pd.DataFrame, list]: Dataframe holding the aggregated experiments, list of lists holding the labcodes of the duplicates
        """
        # prepare the parent frame
        experiments = self.preprocess_experiments_any_valid_output(experiments).copy()
        if "labcode" not in experiments.columns:
            experiments["labcode"] = [
                str(i + 1).zfill(int(np.ceil(np.log10(experiments.shape[0]))))
                for i in range(experiments.shape[0])
            ]

        # round it
        experiments[self.get_feature_keys(ContinuousInputFeature)] = experiments[
            self.get_feature_keys(ContinuousInputFeature)
        ].round(prec)

        # coerce invalid to nan
        experiments = self.coerce_invalids(experiments)

        # group and aggregate
        agg = {feat: "mean" for feat in self.get_feature_keys(ContinuousOutputFeature)}
        agg["labcode"] = lambda x: delimiter.join(sorted(x.tolist()))
        for feat in self.get_feature_keys(OutputFeature):
            agg[f"valid_{feat}"] = lambda x: 1

        grouped = experiments.groupby(self.get_feature_keys(InputFeature))
        duplicated_labcodes = [
            sorted(group.labcode.values.tolist())
            for _, group in grouped
            if group.shape[0] > 1
        ]

        experiments = grouped.aggregate(agg).reset_index(drop=False)
        for feat in self.get_feature_keys(OutputFeature):
            experiments.loc[experiments[feat].isna(), f"valid_{feat}"] = 0

        experiments = experiments.sort_values(by="labcode")
        experiments = experiments.reset_index(drop=True)
        return experiments, sorted(duplicated_labcodes)

    def validate_experiments(
        self,
        experiments: pd.DataFrame,
        strict: bool = False,
    ) -> pd.DataFrame:
        """checks the experimental data on validity

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data

        Raises:
            ValueError: empty dataframe
            ValueError: the column for a specific feature is missing the provided data
            ValueError: there are labcodes with null value
            ValueError: there are labcodes with nan value
            ValueError: labcodes are not unique
            ValueError: the provided columns do no match to the defined domain
            ValueError: the provided columns do no match to the defined domain
            ValueError: inputFeature with null values
            ValueError: inputFeature with nan values

        Returns:
            pd.DataFrame: The provided dataframe with experimental data
        """

        if len(experiments) == 0:
            raise ValueError("no experiments provided (empty dataframe)")
        # check that each feature is a col
        feature_keys = self.get_feature_keys()
        for feature_key in feature_keys:
            if feature_key not in experiments:
                raise ValueError(f"no col in experiments for feature {feature_key}")
        # add valid_{key} cols if missing
        valid_keys = [
            f"valid_{output_feature_key}"
            for output_feature_key in self.get_feature_keys(OutputFeature)
        ]
        for valid_key in valid_keys:
            if valid_key not in experiments:
                experiments[valid_key] = True
        # check all cols
        expected = feature_keys + valid_keys
        cols = list(experiments.columns)
        # we allow here for a column named labcode used to identify experiments
        if "labcode" in cols:
            # test that labcodes are not na
            if experiments.labcode.isnull().values.any():
                raise ValueError("there are labcodes with null value")
            if experiments.labcode.isna().values.any():
                raise ValueError("there are labcodes with nan value")
            # test that labcodes are distinct
            if len(set(experiments.labcode.values.tolist())) != experiments.shape[0]:
                raise ValueError("labcodes are not unique")
            # we remove the labcode from the cols list to proceed as before
            cols.remove("labcode")
        if len(expected) != len(cols):
            raise ValueError(f"expected the following cols: `{expected}`, got `{cols}`")
        if len(set(expected + cols)) != len(cols):
            raise ValueError(f"expected the following cols: `{expected}`, got `{cols}`")
        # check values of continuous input features
        if experiments[self.get_feature_keys(InputFeature)].isnull().values.any():
            raise ValueError("there are null values")
        if experiments[self.get_feature_keys(InputFeature)].isna().values.any():
            raise ValueError("there are na values")
        # run the individual validators
        for feat in self.get_features(InputFeature):
            feat.validate_experimental(experiments[feat.key], strict=strict)
        return experiments

    def describe_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Method to get a tabular overview of how many measurements and how many valid entries are included in the input data for each output feature

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data

        Returns:
            pd.DataFrame: Dataframe with counts how many measurements and how many valid entries are included in the input data for each output feature
        """
        data = {}
        for feat in self.get_feature_keys(OutputFeature):
            data[feat] = [
                experiments.loc[experiments[feat].notna()].shape[0],
                experiments.loc[experiments[feat].notna(), "valid_%s" % feat].sum(),
            ]
        data["all"] = [
            experiments.shape[0],
            self.preprocess_experiments_all_valid_outputs(experiments).shape[0],
        ]
        return pd.DataFrame.from_dict(
            data, orient="index", columns=["measured", "valid"]
        )

    def validate_candidates(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Method to check the validty of porposed candidates

        Args:
            candidates (pd.DataFrame): Dataframe with suggested new experiments (candidates)

        Raises:
            ValueError: when a column is missing for a defined input feature
            ValueError: when a column is missing for a defined output feature
            ValueError: when a non-numerical value is proposed
            ValueError: when the constraints are not fulfilled
            ValueError: when an additional column is found

        Returns:
            pd.DataFrame: dataframe with suggested experiments (candidates)
        """
        # check that each input feature has a col and is valid in itself
        for feat in self.get_features(InputFeature):
            if feat.key not in candidates:
                raise ValueError(f"no col for input feature `{feat.key}`")
            feat.validate_candidental(candidates[feat.key])
        # for each output feature
        for key in self.get_feature_keys(DesirabilityOutputFeature):
            # check that pred, sd, and des cols are specified and numerical
            for col in [f"{key}_pred", f"{key}_sd", f"{key}_des"]:
                if col not in candidates:
                    raise ValueError("missing column {col}")
                if (not is_numeric(candidates[col])) and (
                    not candidates[col].isnull().values.all()
                ):
                    raise ValueError(
                        f"not all values of output feature `{key}` are numerical"
                    )
        # check if all constraints are fulfilled
        if self.is_fulfilled(candidates).all() == False:
            raise ValueError("Constraints not fulfilled.")
        # validate no additional cols exist
        if_count = len(self.get_features(InputFeature))
        of_count = len(self.get_features(DesirabilityOutputFeature))
        # input features, prediction, standard deviation and reward for each output feature, 3 additional usefull infos: reward, aquisition function, strategy
        if len(candidates.columns) != if_count + 3 * of_count:
            raise ValueError("additional columns found")
        return candidates

    @property
    def experiment_column_names(self):
        """the columns in the experimental dataframe

        Returns:
            List[str]: List of columns in the experiment dataframe (output feature keys + valid_output feature keys)
        """
        return self.get_feature_keys() + [
            f"valid_{output_feature_key}"
            for output_feature_key in self.get_feature_keys(OutputFeature)
        ]

    @property
    def candidate_column_names(self):
        """the columns in the candidate dataframe

        Returns:
            List[str]: List of columns in the candidate dataframe (input feature keys + input feature keys_pred, input feature keys_sd, input feature keys_des)
        """
        return (
            self.get_feature_keys(InputFeature)
            + [
                f"{output_feature_key}_pred"
                for output_feature_key in self.get_feature_keys(
                    OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]
                )
            ]
            + [
                f"{output_feature_key}_sd"
                for output_feature_key in self.get_feature_keys(
                    OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]
                )
            ]
            + [
                f"{output_feature_key}_des"
                for output_feature_key in self.get_feature_keys(
                    OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]
                )
            ]
        )

    def add_candidates(self, candidates: pd.DataFrame):
        candidates = self.validate_candidates(candidates)
        if candidates is None:
            self.candidates = candidates
        else:
            self._candidates = pd.concat(
                (self._candidates, candidates), ignore_index=True
            )

    def add_experiments(self, experiments: pd.DataFrame):
        experiments = self.validate_experiments(experiments)
        if experiments is None:
            self.experiments = None
        else:
            self.experiments = pd.concat(
                (self.experiments, experiments), ignore_index=True
            )


def get_subdomain(
    domain: Domain,
    feature_keys: List,
):
    """removes all features not defined as argument creating a subdomain of the provided domain

    Args:
        domain (Domain): the original domain wherefrom a subdomain should be created
        feature_keys (List): List of features that shall be included in the subdomain

    Raises:
        Assert: when in total less than 2 features are provided
        ValueError: when a provided feature key is not present in the provided domain
        Assert: when no output feature is provided
        Assert: when no input feature is provided
        ValueError: _description_

    Returns:
        Domain: A new domain containing only parts of the original domain
    """
    assert len(feature_keys) >= 2, "At least two features have to be provided."
    output_feature_keys = []
    input_feature_keys = []
    subdomain = deepcopy(domain)
    for key in feature_keys:
        try:
            feat = domain.get_feature(key)
        except KeyError:
            raise ValueError(f"Feature {key} not present in domain.")
        if isinstance(feat, InputFeature):
            input_feature_keys.append(key)
        else:
            output_feature_keys.append(key)
    assert (
        len(output_feature_keys) > 0
    ), "At least one output feature has to be provided."
    assert len(input_feature_keys) > 0, "At least one input feature has to be provided."
    # loop over constraints and make sure that all features used in constraints are in the input_feature_keys
    for c in domain.constraints:
        for key in c.features:
            if key not in input_feature_keys:
                raise ValueError(
                    f"Removed input feature {key} is used in a constraint."
                )

    for key in set(domain.get_feature_keys(Feature)) - set(feature_keys):
        subdomain.remove_feature_by_key(key)
    return subdomain


class DomainError(Exception):
    """A class defining a specific domain error"""

    pass
