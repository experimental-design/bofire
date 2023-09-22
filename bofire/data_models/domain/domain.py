import collections.abc
import itertools
import warnings
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import numpy as np
import pandas as pd
from pydantic import Field, validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import (
    AnyConstraint,
    ConstraintNotFulfilledError,
    LinearConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.constraints import Constraints
from bofire.data_models.domain.features import Features, Inputs, Outputs
from bofire.data_models.features.api import (
    AnyInput,
    AnyOutput,
    ContinuousInput,
    ContinuousOutput,
    Feature,
    Input,
    Output,
)
from bofire.data_models.objectives.api import Objective


def isinstance_or_union(obj, of):
    if get_origin(of) is Union:
        of = get_args(of)
    return isinstance(obj, of)


def is_numeric(s: Union[pd.Series, pd.DataFrame]) -> bool:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").notnull().all()
    return s.apply(lambda s: pd.to_numeric(s, errors="coerce").notnull().all()).all()  # type: ignore


class Domain(BaseModel):
    type: Literal["Domain"] = "Domain"

    inputs: Inputs = Field(default_factory=lambda: Inputs())
    outputs: Outputs = Field(default_factory=lambda: Outputs())

    constraints: Constraints = Field(default_factory=lambda: Constraints())

    """Representation of the optimization problem/domain

    Attributes:
        inputs (List[Input], optional): List of input features. Defaults to [].
        outputs (List[Output], optional): List of output features. Defaults to [].
        constraints (List[Constraint], optional): List of constraints. Defaults to [].
    """

    @classmethod
    def from_lists(
        cls,
        inputs: Optional[Sequence[AnyInput]] = None,
        outputs: Optional[Sequence[AnyOutput]] = None,
        constraints: Optional[Sequence[AnyConstraint]] = None,
    ):
        inputs = [] if inputs is None else inputs
        outputs = [] if outputs is None else outputs
        constraints = [] if constraints is None else constraints
        return cls(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=outputs),
            constraints=Constraints(constraints=constraints),
        )

    @validator("inputs", always=True, pre=True)
    def validate_inputs_list(cls, v, values):
        if isinstance(v, collections.abc.Sequence):
            v = Inputs(features=v)
            return v
        if isinstance_or_union(v, AnyInput):
            return Inputs(features=[v])
        else:
            return v

    @validator("outputs", always=True, pre=True)
    def validate_outputs_list(cls, v, values):
        if isinstance(v, collections.abc.Sequence):
            return Outputs(features=v)
        if isinstance_or_union(v, AnyOutput):
            return Outputs(features=[v])
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

    @validator("outputs", always=True)
    def validate_unique_feature_keys(cls, v: Outputs, values) -> Outputs:
        """Validates if provided input and output feature keys are unique

        Args:
            v (Outputs): List of all output features of the domain.
            value (Dict[str, Inputs]): Dict containing a list of input features as single entry.

        Raises:
            ValueError: Feature keys are not unique.

        Returns:
            Outputs: Keeps output features as given.
        """
        if "inputs" not in values:
            return v
        features = v + values["inputs"]
        keys = [f.key for f in features]
        if len(set(keys)) != len(keys):
            raise ValueError("feature keys are not unique")
        return v

    @validator("constraints", always=True)
    def validate_constraints(cls, v, values):
        """Validate if all features included in the constraints are also defined as features for the domain.

        Args:
            v (List[Constraint]): List of constraints or empty if no constraints are defined
            values (List[Input]): List of input features of the domain

        Raises:
            ValueError: Feature key in constraint is unknown.

        Returns:
            List[Constraint]: List of constraints defined for the domain
        """
        if "inputs" not in values:
            return v
        keys = [f.key for f in values["inputs"]]
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
            values (List[Input]): List of input features of the domain

        Raises:
            ValueError: _description_


        Returns:
           List[Constraint]: List of constraints defined for the domain
        """
        if "inputs" not in values:
            return v

        # gather continuous inputs in dictionary
        continuous_inputs_dict = {}
        for f in values["inputs"]:
            if isinstance(f, ContinuousInput):
                continuous_inputs_dict[f.key] = f

        # check if non continuous input features appear in linear constraints
        for c in v:
            if isinstance(c, LinearConstraint):
                for f in c.features:
                    assert f in continuous_inputs_dict, f"{f} must be continuous."
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
        assert isinstance(self.inputs, Inputs)
        features = self.inputs + self.outputs
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
        assert isinstance(self.inputs, Inputs)
        return {f.key: f for f in self.inputs + self.outputs}[key]

    # TODO: tidy this up
    def get_nchoosek_combinations(self, exhaustive: bool = False):  # noqa: C901
        """get all possible NChooseK combinations

        Args:
            exhaustive (bool, optional): if True all combinations are returned. Defaults to False.

        Returns:
            Tuple(used_features_list, unused_features_list): used_features_list is a list of lists containing features used in each NChooseK combination.
                unused_features_list is a list of lists containing features unused in each NChooseK combination.
        """

        if len(self.constraints.get(NChooseKConstraint)) == 0:
            used_continuous_features = self.get_feature_keys(ContinuousInput)
            return used_continuous_features, []

        used_features_list_all = []

        # loops through each NChooseK constraint
        for con in self.constraints.get(NChooseKConstraint):
            assert isinstance(con, NChooseKConstraint)
            used_features_list = []

            if exhaustive:
                for n in range(con.min_count, con.max_count + 1):
                    used_features_list.extend(itertools.combinations(con.features, n))

                if con.none_also_valid:
                    used_features_list.append(())
            else:
                used_features_list.extend(
                    itertools.combinations(con.features, con.max_count)
                )

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
            for con in self.constraints.get(NChooseKConstraint):
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
        for con in self.constraints.get(NChooseKConstraint):
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
        for feat in self.get_feature_keys(Output):
            experiments.loc[experiments[f"valid_{feat}"] == 0, feat] = np.nan
        return experiments

    def aggregate_by_duplicates(
        self,
        experiments: pd.DataFrame,
        prec: int,
        delimiter: str = "-",
        method: Literal["mean", "median"] = "mean",
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
        if method not in ["mean", "median"]:
            raise ValueError(f"Unknown aggregation type provided: {method}")

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
            feat: method for feat in self.get_feature_keys(ContinuousOutput)
        }
        agg["labcode"] = lambda x: delimiter.join(sorted(x.tolist()))
        for feat in self.get_feature_keys(Output):
            agg[f"valid_{feat}"] = lambda x: 1

        grouped = experiments.groupby(self.get_feature_keys(Input))
        duplicated_labcodes = [
            sorted(group.labcode.to_numpy().tolist())
            for _, group in grouped
            if group.shape[0] > 1
        ]

        experiments = grouped.aggregate(agg).reset_index(drop=False)
        for feat in self.get_feature_keys(Output):
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
            ValueError: Input with null values
            ValueError: Input with nan values

        Returns:
            pd.DataFrame: The provided dataframe with experimental data
        """

        if len(experiments) == 0:
            raise ValueError("no experiments provided (empty dataframe)")
        # we allow here for a column named labcode used to identify experiments
        if "labcode" in experiments.columns:
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
        # run the individual validators
        experiments = self.inputs.validate_experiments(
            experiments=experiments, strict=strict
        )
        experiments = self.outputs.validate_experiments(experiments=experiments)
        return experiments

    def describe_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Method to get a tabular overview of how many measurements and how many valid entries are included in the input data for each output feature

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data

        Returns:
            pd.DataFrame: Dataframe with counts how many measurements and how many valid entries are included in the input data for each output feature
        """
        data = {}
        for feat in self.get_feature_keys(Output):
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
        self,
        candidates: pd.DataFrame,
        only_inputs: bool = False,
        tol: float = 1e-5,
        raise_validation_error: bool = True,
    ) -> pd.DataFrame:
        """Method to check the validty of porposed candidates

        Args:
            candidates (pd.DataFrame): Dataframe with suggested new experiments (candidates)
            only_inputs (bool,optional): If True, only the input columns are validated. Defaults to False.
            tol (float,optional): tolerance parameter for constraints. A constraint is considered as not fulfilled if the violation
                is larger than tol. Defaults to 1e-6.
            raise_validation_error (bool, optional): If true an error will be raised if candidates violate constraints,
                otherwise only a warning will be displayed. Defaults to True.

        Raises:
            ValueError: when a column is missing for a defined input feature
            ValueError: when a column is missing for a defined output feature
            ValueError: when a non-numerical value is proposed
            ValueError: when an additional column is found
            ConstraintNotFulfilledError: when the constraints are not fulfilled and `raise_validation_error = True`

        Returns:
            pd.DataFrame: dataframe with suggested experiments (candidates)
        """
        # check that each input feature has a col and is valid in itself
        assert isinstance(self.inputs, Inputs)
        candidates = self.inputs.validate_candidates(candidates)
        # check if all constraints are fulfilled
        if not self.constraints.is_fulfilled(candidates, tol=tol).all():
            if raise_validation_error:
                raise ConstraintNotFulfilledError(
                    f"Constraints not fulfilled: {candidates}"
                )
            warnings.warn("Not all constraints are fulfilled.")
        # for each continuous output feature with an attached objective object
        if not only_inputs:
            assert isinstance(self.outputs, Outputs)
            candidates = self.outputs.validate_candidates(candidates=candidates)
        return candidates

    @property
    def experiment_column_names(self):
        """the columns in the experimental dataframe

        Returns:
            List[str]: List of columns in the experiment dataframe (output feature keys + valid_output feature keys)
        """
        return self.get_feature_keys() + [
            f"valid_{output_feature_key}"
            for output_feature_key in self.get_feature_keys(Output)
        ]

    @property
    def candidate_column_names(self):
        """the columns in the candidate dataframe

        Returns:
            List[str]: List of columns in the candidate dataframe (input feature keys + input feature keys_pred, input feature keys_sd, input feature keys_des)
        """
        assert isinstance(self.outputs, Outputs)
        return (
            self.get_feature_keys(Input)
            + [
                f"{output_feature_key}_pred"
                for output_feature_key in self.outputs.get_keys_by_objective(Objective)
            ]
            + [
                f"{output_feature_key}_sd"
                for output_feature_key in self.outputs.get_keys_by_objective(Objective)
            ]
            + [
                f"{output_feature_key}_des"
                for output_feature_key in self.outputs.get_keys_by_objective(Objective)
            ]
        )

    def _set_constraints_unvalidated(
        self, constraints: Union[Sequence[AnyConstraint], Constraints]
    ):
        """Hack for reduce_domain"""
        self.constraints = Constraints(constraints=[])
        if isinstance(constraints, Constraints):
            constraints = constraints.constraints
        self.constraints.constraints = constraints
