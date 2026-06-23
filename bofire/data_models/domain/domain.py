import collections.abc
import warnings
from collections.abc import Sequence
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import (
    AnyConstraint,
    Constraint,
    ConstraintNotFulfilledError,
    InterpointConstraint,
)
from bofire.data_models.domain.constraints import Constraints
from bofire.data_models.domain.features import Inputs, Outputs
from bofire.data_models.features.api import (
    AnyInput,
    AnyOutput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
    Input,
    Output,
)
from bofire.data_models.objectives.api import Objective


def is_numeric(s: Union[pd.Series, pd.DataFrame]) -> bool:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").notnull().all()
    return s.apply(lambda s: pd.to_numeric(s, errors="coerce").notnull().all()).all()


class Domain(BaseModel):
    """Representation of the optimization problem/domain

    Attributes:
        inputs (List[Input], optional): List of input features. Defaults to [].
        outputs (List[Output], optional): List of output features. Defaults to [].
        constraints (List[Constraint], optional): List of constraints. Defaults to [].
        context (str, optional): Free-text context providing additional information
            about the optimization problem. Useful for agentic optimization where an
            LLM agent can leverage this description to better understand the overall
            problem, its goals, and any domain-specific knowledge.
    """

    type: Literal["Domain"] = "Domain"

    inputs: Inputs = Field(default_factory=lambda: Inputs())
    outputs: Outputs = Field(default_factory=lambda: Outputs())
    constraints: Constraints = Field(default_factory=lambda: Constraints())
    context: Optional[str] = None

    def to_description(self) -> str:
        """Render a human-readable description of the optimization problem.

        Covers problem context, objectives, and constraints. Feature details
        are handled separately by ``Inputs.to_pydantic_model()`` which embeds
        bounds, types, and context into the dynamic output schema.
        """
        lines = []

        if self.context:
            lines.append(f"## Problem Context\n{self.context}")

        lines.append("\n## Objectives")
        for feat in self.outputs:
            lines.append(f"- {feat.to_description()}")

        if len(self.constraints) > 0:
            lines.append("\n## Constraints (candidates MUST satisfy all of these)")
            for c in self.constraints:
                lines.append(f"- {c.to_description()}")

        return "\n".join(lines)

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

    @field_validator("inputs", mode="before")
    @classmethod
    def validate_inputs_list(cls, v):
        if isinstance(v, collections.abc.Sequence):
            v = Inputs(features=v)
            return v
        if isinstance(v, Input):
            return Inputs(features=[v])
        return v

    @field_validator("outputs", mode="before")
    @classmethod
    def validate_outputs_list(cls, v):
        if isinstance(v, collections.abc.Sequence):
            return Outputs(features=v)
        if isinstance(v, Output):
            return Outputs(features=[v])
        return v

    @field_validator("constraints", mode="before")
    @classmethod
    def validate_constraints_list(cls, v):
        if isinstance(v, list):
            return Constraints(constraints=v)
        if isinstance(v, Constraint):
            return Constraints(constraints=[v])
        return v

    @model_validator(mode="after")
    def validate_unique_feature_keys(self):
        """Validates if provided input and output feature keys are unique

        Args:
            v (Outputs): List of all output features of the domain.
            value (Dict[str, Inputs]): Dict containing a list of input features as single entry.

        Raises:
            ValueError: Feature keys are not unique.

        Returns:
            Outputs: Keeps output features as given.

        """
        keys = self.outputs.get_keys() + self.inputs.get_keys()
        if len(set(keys)) != len(keys):
            raise ValueError("Feature keys are not unique")
        return self

    @model_validator(mode="after")
    def validate_constraints(self):
        """Validate that the constraints defined in the domain fit to the input features.

        Args:
            v (List[Constraint]): List of constraints or empty if no constraints are defined
            values (List[Input]): List of input features of the domain

        Raises:
            ValueError: Feature key in constraint is unknown.

        Returns:
            List[Constraint]: List of constraints defined for the domain

        """
        for c in self.constraints.get():
            c.validate_inputs(self.inputs)
        return self

    def coerce_invalids(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Coerces all invalid output measurements to np.nan

        Args:
            experiments (pd.DataFrame): Dataframe containing experimental data

        Returns:
            pd.DataFrame: coerced dataframe

        """
        # coerce invalid to nan
        for feat in self.outputs.get_keys(Output):
            experiments.loc[experiments[f"valid_{feat}"] == 0, feat] = np.nan
        return experiments

    def aggregate_by_duplicates(
        self,
        experiments: pd.DataFrame,
        prec: int,
        delimiter: str = "-",
        method: Literal["mean", "median"] = "mean",
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, list]:
        """Aggregate the dataframe by duplicate experiments

        Duplicates are identified based on the experiments with the same input
        features. Continuous input features are rounded before identifying the
        duplicates. Continuous output features are aggregated by taking the
        average (or median) of the involved values; categorical output
        features are aggregated by majority vote. Ties in the majority vote
        are broken by a random pick; pass ``random_state`` to make this
        reproducible.

        Args:
            experiments (pd.DataFrame): Dataframe containing experimental data
            prec (int): Precision of the rounding of the continuous input features
            delimiter (str, optional): Delimiter used when combining the orig.
                labcodes to a new one. Defaults to "-".
            method (Literal["mean", "median"], optional): Which aggregation
                method to use for continuous outputs. Defaults to "mean".
                Categorical outputs always use majority vote regardless of
                this argument.
            random_state (int, optional): Seed used only when breaking ties in
                the categorical majority vote. Defaults to None
                (non-deterministic).

        Returns:
            Tuple[pd.DataFrame, list]: Dataframe holding the aggregated
                experiments, list of lists holding the labcodes of the duplicates

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

        # round it if continuous inputs are present
        if len(self.inputs.get(ContinuousInput)) > 0:
            experiments[self.inputs.get_keys(ContinuousInput)] = experiments[
                self.inputs.get_keys(ContinuousInput)
            ].round(prec)

        # coerce invalid to nan
        experiments = self.coerce_invalids(experiments)

        # group and aggregate
        agg: Dict[str, Any] = dict.fromkeys(
            self.outputs.get_keys(ContinuousOutput), method
        )

        seed = np.random.default_rng(random_state)

        def _make_categorical_aggregator(feat: str):
            def _aggregate(values: pd.Series):
                non_na = values.dropna()
                if len(non_na) == 0:
                    return np.nan
                counts = non_na.value_counts()
                top_count = counts.iloc[0]
                winners = counts[counts == top_count].index.tolist()
                if len(winners) > 1:
                    return seed.choice(winners)
                return winners[0]

            return _aggregate

        for feat in self.outputs.get_keys(CategoricalOutput):
            agg[feat] = _make_categorical_aggregator(feat)
        agg["labcode"] = lambda x: delimiter.join(sorted(x.tolist()))
        for feat in self.outputs.get_keys(Output):
            agg[f"valid_{feat}"] = lambda x: 1

        grouped = experiments.groupby(self.inputs.get_keys(Input))
        duplicated_labcodes = [
            sorted(group.labcode.to_numpy().tolist())
            for _, group in grouped
            if group.shape[0] > 1
        ]

        experiments = grouped.aggregate(agg).reset_index(drop=False)
        for feat in self.outputs.get_keys(Output):
            experiments.loc[experiments[feat].isna(), f"valid_{feat}"] = 0

        experiments = experiments.sort_values(by="labcode")
        experiments = experiments.reset_index(drop=True)
        return experiments, sorted(duplicated_labcodes)

    def validate_experiments(
        self,
        experiments: pd.DataFrame,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Checks the experimental data on validity

        Args:
            experiments (pd.DataFrame): Dataframe with experimental data
            strict (bool, optional): Boolean to distinguish if the occurrence of
                fixed features in the dataset should be considered or not.
                Defaults to False.

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
            experiments=experiments,
            strict=strict,
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
        for feat in self.outputs.get_keys(Output):
            data[feat] = [
                experiments.loc[experiments[feat].notna()].shape[0],
                experiments.loc[experiments[feat].notna(), "valid_%s" % feat].sum(),
            ]
        preprocessed = self.outputs.preprocess_experiments_all_valid_outputs(
            experiments,
        )
        assert preprocessed is not None
        data["all"] = [
            experiments.shape[0],
            preprocessed.shape[0],
        ]
        return pd.DataFrame.from_dict(
            data,
            orient="index",
            columns=["measured", "valid"],
        )

    def validate_candidates(
        self,
        candidates: pd.DataFrame,
        only_inputs: bool = False,
        tol: float = 1e-5,
        raise_validation_error: bool = True,
    ) -> pd.DataFrame:
        """Method to check the validty of proposed candidates

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
                    f"Constraints not fulfilled: {candidates}",
                )
            warnings.warn("Not all constraints are fulfilled.")
        # for each continuous output feature with an attached objective object
        if not only_inputs:
            assert isinstance(self.outputs, Outputs)
            candidates = self.outputs.validate_candidates(candidates=candidates)
        return candidates

    def is_fulfilled(
        self,
        experiments: pd.DataFrame,
        tol: float = 1e-6,
        exlude_interpoint: bool = True,
    ) -> pd.Series:
        """Check if all constraints are fulfilled on all rows of the provided dataframe
        both constraints and inputs are checked.

        Args:
            experiments: Dataframe with data, the constraint validity should be tested on
            tol: Tolerance for checking the constraints. Defaults to 1e-6.
            exlude_interpoint: If True, InterpointConstraints are excluded from the check. Defaults to True.

        Returns:
            Boolean series indicating if all constraints are fulfilled for all rows.
        """
        constraints = (
            self.constraints.get(excludes=[InterpointConstraint])
            if exlude_interpoint
            else self.constraints.get()
        )
        return constraints.is_fulfilled(experiments, tol) & self.inputs.is_fulfilled(
            experiments
        )

    @property
    def experiment_column_names(self):
        """The columns in the experimental dataframe

        Returns:
            List[str]: List of columns in the experiment dataframe (output feature keys + valid_output feature keys)

        """
        return (self.inputs + self.outputs).get_keys() + [
            f"valid_{output_feature_key}"
            for output_feature_key in self.outputs.get_keys(Output)
        ]

    @property
    def candidate_column_names(self):
        """The columns in the candidate dataframe

        Returns:
            List[str]: List of columns in the candidate dataframe (input feature keys + input feature keys_pred, input feature keys_sd, input feature keys_des)

        """
        assert isinstance(self.outputs, Outputs)
        return (
            self.inputs.get_keys(Input)
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
