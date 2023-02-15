import collections.abc
import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
from pydantic import Field, parse_obj_as, validator

from bofire.domain.constraints import (
    AnyConstraint,
    Constraints,
    LinearConstraint,
    NChooseKConstraint,
)
from bofire.domain.features import (
    AnyInputFeature,
    AnyOutputFeature,
    ContinuousInput,
    ContinuousOutput,
    Feature,
    Features,
    InputFeature,
    InputFeatures,
    OutputFeature,
    OutputFeatures,
)
from bofire.domain.objectives import Objective
from bofire.domain.util import PydanticBaseModel, is_numeric, isinstance_or_union


class Domain(PydanticBaseModel):
    # The types describe what we expect to be passed as arguments.
    # They will be converted to InputFeatures and OutputFeatures, respectively.
    input_features: Union[Sequence[AnyInputFeature], InputFeatures] = Field(
        default_factory=lambda: InputFeatures()
    )
    output_features: Union[Sequence[AnyOutputFeature], OutputFeatures] = Field(
        default_factory=lambda: OutputFeatures()
    )

    constraints: Union[Sequence[AnyConstraint], Constraints] = Field(
        default_factory=lambda: Constraints()
    )

    experiments: Optional[pd.DataFrame] = None
    candidates: Optional[pd.DataFrame] = None

    """Representation of the optimization problem/domain

    Attributes:
        input_features (List[InputFeature], optional): List of input features. Defaults to [].
        output_features (List[OutputFeature], optional): List of output features. Defaults to [].
        constraints (List[Constraint], optional): List of constraints. Defaults to [].
    """

    @property
    def outputs(self) -> OutputFeatures:
        """Returns output features as OutputFeatures"""
        return cast(OutputFeatures, self.output_features)

    @property
    def inputs(self) -> InputFeatures:
        """Returns input features as InputFeatures"""
        return cast(InputFeatures, self.input_features)

    @property
    def cnstrs(self) -> Constraints:
        return cast(Constraints, self.constraints)

    @validator("input_features", always=True, pre=True)
    def validate_input_features_list(cls, v, values):
        if isinstance(v, collections.abc.Sequence):
            v = InputFeatures(features=v)
            return v
        if isinstance_or_union(v, AnyInputFeature):
            return InputFeatures(features=[v])
        else:
            return v

    @validator("output_features", always=True, pre=True)
    def validate_output_features_list(cls, v, values):
        if isinstance(v, collections.abc.Sequence):
            return OutputFeatures(features=v)
        if isinstance_or_union(v, AnyOutputFeature):
            return OutputFeatures(features=[v])
        else:
            return v

    @validator("constraints", always=True, pre=True)
    def validate_constraints_list(cls, v, values):
        if isinstance(v, list):
            return Constraints(constraints=v)
        if isinstance_or_union(v, AnyConstraint):
            return Constraints(constraints=[v])
        else:
            return v

    @validator("output_features", always=True)
    def validate_unique_feature_keys(cls, v: OutputFeatures, values) -> OutputFeatures:
        """Validates if provided input and output feature keys are unique

        Args:
            v (OutputFeatures): List of all output features of the domain.
            value (Dict[str, InputFeatures]): Dict containing a list of input features as single entry.

        Raises:
            ValueError: Feature keys are not unique.

        Returns:
            OutputFeatures: Keeps output features as given.
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
    def validate_linear_constraints(cls, v, values):
        """Validate if all features included in linear constraints are continuous ones.

        Args:
            v (List[Constraint]): List of constraints or empty if no constraints are defined
            values (List[InputFeature]): List of input features of the domain

        Raises:
            ValueError: _description_


        Returns:
           List[Constraint]: List of constraints defined for the domain
        """
        if "input_features" not in values:
            return v

        # gather continuous input_features in dictionary
        continuous_input_features_dict = {}
        for f in values["input_features"]:
            if type(f) is ContinuousInput:
                continuous_input_features_dict[f.key] = f

        # check if non continuous input features appear in linear constraints
        for c in v:
            if isinstance(c, LinearConstraint):
                for f in c.features:
                    assert (
                        f in continuous_input_features_dict
                    ), f"{f} must be continuous."
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
            if type(f) is ContinuousInput:
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
            index=range(len(self.constraints)),
            columns=["Type", "Description"],
            data={
                "Type": [feat.__class__.__name__ for feat in self.constraints],
                "Description": [
                    constraint.__str__() for constraint in self.constraints
                ],
            },
        )
        return df

    def get_features(
        self,
        includes: Union[Type[Feature], List[Type[Feature]]] = Feature,
        excludes: Union[Type[Feature], List[Type[Feature]], None] = None,
        exact: bool = False,
    ) -> Features:
        """get features of the domain

        Args:
            includes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be returned. Defaults to Feature.
            excludes (Union[Type, List[Type]], optional): Feature class or list of specific feature classes to be excluded from the return. Defaults to None.
            exact (bool, optional): Boolean to distinguish if only the exact class listed in includes and no subclasses inherenting from this class shall be returned. Defaults to False.
            by_attribute (str, optional): If set it is filtered by the attribute specified in by `by_attribute`. Defaults to None.

        Returns:
            List[Feature]: List of features in the domain fitting to the passed requirements.
        """
        assert isinstance(self.input_features, InputFeatures)
        features = self.input_features + self.output_features
        return features.get(includes, excludes, exact)

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
        assert isinstance(self.input_features, InputFeatures)
        return {f.key: f for f in self.input_features + self.output_features}[key]

    # getting list of fixed values
    def get_nchoosek_combinations(self):
        """get all possible NChooseK combinations

        Returns:
            Tuple(used_features_list, unused_features_list): used_features_list is a list of lists containing features used in each NChooseK combination.
             unused_features_list is a list of lists containing features unused in each NChooseK combination.
        """

        if len(self.cnstrs.get(NChooseKConstraint)) == 0:
            used_continuous_features = self.get_feature_keys(ContinuousInput)
            return used_continuous_features, []

        used_features_list_all = []

        # loops through each NChooseK constraint
        for con in self.cnstrs.get(NChooseKConstraint):
            assert isinstance(con, NChooseKConstraint)
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
            for con in self.cnstrs.get(NChooseKConstraint):
                assert isinstance(con, NChooseKConstraint)
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
        for con in self.cnstrs.get(NChooseKConstraint):
            assert isinstance(con, NChooseKConstraint)
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

        preprocessed = self.outputs.preprocess_experiments_any_valid_output(experiments)
        assert preprocessed is not None
        experiments = preprocessed.copy()
        if "labcode" not in experiments.columns:
            experiments["labcode"] = [
                str(i + 1).zfill(int(np.ceil(np.log10(experiments.shape[0]))))
                for i in range(experiments.shape[0])
            ]

        # round it
        experiments[self.get_feature_keys(ContinuousInput)] = experiments[
            self.get_feature_keys(ContinuousInput)
        ].round(prec)

        # coerce invalid to nan
        experiments = self.coerce_invalids(experiments)

        # group and aggregate
        agg: Dict[str, Any] = {
            feat: "mean" for feat in self.get_feature_keys(ContinuousOutput)
        }
        agg["labcode"] = lambda x: delimiter.join(sorted(x.tolist()))
        for feat in self.get_feature_keys(OutputFeature):
            agg[f"valid_{feat}"] = lambda x: 1

        grouped = experiments.groupby(self.get_feature_keys(InputFeature))
        duplicated_labcodes = [
            sorted(group.labcode.to_numpy().tolist())
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
            if experiments.labcode.isnull().to_numpy().any():
                raise ValueError("there are labcodes with null value")
            if experiments.labcode.isna().to_numpy().any():
                raise ValueError("there are labcodes with nan value")
            # test that labcodes are distinct
            if (
                len(set(experiments.labcode.to_numpy().tolist()))
                != experiments.shape[0]
            ):
                raise ValueError("labcodes are not unique")
            # we remove the labcode from the cols list to proceed as before
            cols.remove("labcode")
        if len(expected) != len(cols):
            raise ValueError(f"expected the following cols: `{expected}`, got `{cols}`")
        if len(set(expected + cols)) != len(cols):
            raise ValueError(f"expected the following cols: `{expected}`, got `{cols}`")
        # check values of continuous input features
        if experiments[self.get_feature_keys(InputFeature)].isnull().to_numpy().any():
            raise ValueError("there are null values")
        if experiments[self.get_feature_keys(InputFeature)].isna().to_numpy().any():
            raise ValueError("there are na values")
        # run the individual validators
        for feat in self.get_features(InputFeature):
            assert isinstance(feat, InputFeature)
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
        preprocessed = self.outputs.preprocess_experiments_all_valid_outputs(
            experiments
        )
        assert preprocessed is not None
        data["all"] = [
            experiments.shape[0],
            preprocessed.shape[0],
        ]
        return pd.DataFrame.from_dict(
            data, orient="index", columns=["measured", "valid"]
        )

    def validate_candidates(
        self, candidates: pd.DataFrame, only_inputs: bool = False
    ) -> pd.DataFrame:
        """Method to check the validty of porposed candidates

        Args:
            candidates (pd.DataFrame): Dataframe with suggested new experiments (candidates)
            only_inputs (bool,optional): If True, only the input columns are validated. Defaults to False.

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
        assert isinstance(self.input_features, InputFeatures)
        self.input_features.validate_inputs(candidates)
        # check if all constraints are fulfilled
        if not self.cnstrs.is_fulfilled(candidates).all():
            raise ValueError("Constraints not fulfilled.")
        # for each continuous output feature with an attached objective object
        if not only_inputs:
            assert isinstance(self.output_features, OutputFeatures)
            for key in self.output_features.get_keys_by_objective(Objective):
                # check that pred, sd, and des cols are specified and numerical
                for col in [f"{key}_pred", f"{key}_sd", f"{key}_des"]:
                    if col not in candidates:
                        raise ValueError(f"missing column {col}")
                    if (not is_numeric(candidates[col])) and (
                        not candidates[col].isnull().to_numpy().all()
                    ):
                        raise ValueError(
                            f"not all values of output feature `{key}` are numerical"
                        )
            # validate no additional cols exist
            if_count = len(self.get_features(InputFeature))
            of_count = len(self.output_features.get_keys_by_objective(Objective))
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
        assert isinstance(self.output_features, OutputFeatures)
        return (
            self.get_feature_keys(InputFeature)
            + [
                f"{output_feature_key}_pred"
                for output_feature_key in self.output_features.get_keys_by_objective(
                    Objective
                )
            ]
            + [
                f"{output_feature_key}_sd"
                for output_feature_key in self.output_features.get_keys_by_objective(
                    Objective
                )
            ]
            + [
                f"{output_feature_key}_des"
                for output_feature_key in self.output_features.get_keys_by_objective(
                    Objective
                )
            ]
        )

    def set_candidates(self, candidates: pd.DataFrame):
        candidates = self.validate_candidates(candidates)
        self.candidates = candidates

    def add_candidates(self, candidates: pd.DataFrame):
        candidates = self.validate_candidates(candidates)
        if candidates is None:
            self.candidates = candidates
        else:
            self.candidates = pd.concat(
                (self.candidates, candidates), ignore_index=True
            )

    @property
    def num_candidates(self) -> int:
        if self.candidates is None:
            return 0
        return len(self.candidates)

    def set_experiments(self, experiments: pd.DataFrame):
        experiments = self.validate_experiments(experiments)
        self.experiments = experiments

    def add_experiments(self, experiments: pd.DataFrame):
        experiments = self.validate_experiments(experiments)
        if experiments is None:
            self.experiments = None
        elif self.experiments is None:
            self.experiments = experiments
        else:
            self.experiments = pd.concat(
                (self.experiments, experiments), ignore_index=True
            )

    def _set_constraints_unvalidated(
        self, constraints: Union[Sequence[AnyConstraint], Constraints]
    ):
        """Hack for reduce_domain"""
        self.constraints = Constraints(constraints=[])
        if isinstance(constraints, Constraints):
            constraints = constraints.constraints
        self.constraints.constraints = constraints

    @property
    def num_experiments(self) -> int:
        if self.experiments is None:
            return 0
        return len(self.experiments)

    @staticmethod
    def from_dict(dict_: dict):
        return parse_obj_as(Domain, dict_)


def get_subdomain(
    domain: Domain,
    feature_keys: List,
) -> Domain:
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
    output_features = []
    input_features = []
    for key in feature_keys:
        try:
            feat = domain.get_feature(key)
        except KeyError:
            raise ValueError(f"Feature {key} not present in domain.")
        if isinstance(feat, InputFeature):
            input_features.append(feat)
        else:
            output_features.append(feat)
    assert len(output_features) > 0, "At least one output feature has to be provided."
    assert len(input_features) > 0, "At least one input feature has to be provided."
    input_features = InputFeatures(features=input_features)
    # loop over constraints and make sure that all features used in constraints are in the input_feature_keys
    for c in domain.constraints:
        # TODO: fix type hint
        for key in c.features:  # type: ignore
            if key not in input_features.get_keys():
                raise ValueError(
                    f"Removed input feature {key} is used in a constraint."
                )
    subdomain = deepcopy(domain)
    subdomain.input_features = input_features
    subdomain.output_features = output_features
    return subdomain


class DomainError(Exception):
    """A class defining a specific domain error"""

    pass
