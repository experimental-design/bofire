from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
    Input,
)
from bofire.data_models.strategies.doe import (
    AnyDoEOptimalityCriterion,
    DoEOptimalityCriterion,
)
from bofire.strategies.doe.design import find_local_max_ipopt, get_n_experiments
from bofire.strategies.doe.utils import get_formula_from_string, n_zero_eigvals
from bofire.strategies.strategy import Strategy


def map_discrete_to_continuous(
    inputs: List[DiscreteInput],
) -> Tuple[List[ContinuousInput], List[ContinuousInput], Constraints]:
    """
    Maps a discrete input to a constrained set of continuous inputs. Each discrete value is represented by a
    continuous variable between 0 and 1. As only one can be active at a time, the sum of all continuous variables
    is constrained to be 1. The constraint is added to the domain as a NChooseKConstraint. The number of discrete
    values is the number of continuous variables.

    Args:
        domain (Domain): _description_
    """

    def generate_value_key(input: DiscreteInput, d: float):
        return f"aux_{input.key}_{d}".replace(".", "_")

    # Create a new list of inputs
    new_inputs = []
    new_auxiliary_inputs = []
    new_constraints = []
    # Iterate over the inputs of the original domain
    for input in inputs:
        # If the input is a DiscreteInput, replace it with a ContinuousInput
        assert isinstance(
            input, DiscreteInput
        ), f"Expected {DiscreteInput.__name__}, got {type(input)}"

        # Create a new ContinuousInput for each value
        new_inputs_for_input = []
        new_aux_inputs_for_input = []
        new_inputs_for_input.append(
            ContinuousInput(key=input.key, bounds=[0, input.values[-1]])
        )
        new_inputs += new_inputs_for_input
        for d in input.values:
            new_aux_inputs_for_input.append(
                ContinuousInput(
                    key=generate_value_key(input, d),
                    bounds=[0, 1],
                )
            )
        new_auxiliary_inputs += new_aux_inputs_for_input
        # Create a new list of constraints
        new_constraints.append(
            LinearEqualityConstraint(
                features=[i.key for i in new_aux_inputs_for_input],
                coefficients=[1] * len(new_aux_inputs_for_input),
                rhs=1,
            )
        )
        new_constraints.append(
            NChooseKConstraint(
                features=[i.key for i in new_aux_inputs_for_input],
                min_count=0,
                max_count=1,
                none_also_valid=True,
            )
        )
        new_constraints.append(
            LinearEqualityConstraint(
                features=[
                    i.key for i in new_inputs_for_input + new_aux_inputs_for_input
                ],
                coefficients=[1.0] + [-value for value in input.values],
                rhs=0.0,
            )
        )
    return (
        new_inputs,
        new_auxiliary_inputs,
        Constraints(constraints=new_constraints),
    )


def map_categorical_to_continuous(
    categorical_inputs: List[CategoricalInput],
) -> Tuple[Dict[str, Dict[str, str]], List[ContinuousInput], Constraints]:
    """
    Maps a categorical input to a constrained set of continuous inputs. Each category is represented by a
    continuous variable between 0 and 1. As only one can be active at a time, the sum of all continuous variables
    is constrained to be 1. The constraint is added to the domain as a NChooseKConstraint. The number of categories
    is the number of continuous variables.

    Args:
        domain (Domain): _description_
    """

    # Create a new list of inputs
    def generate_value_key(input: CategoricalInput, c: str):
        return f"aux_{input.key}_{c}"

    new_constraints = []
    new_auxiliary_inputs = []
    mappings_aux_to_category = {}
    # Iterate over the inputs of the original domain
    for input in categorical_inputs:
        # If the input is a CategoricalInput, replace it with a ContinuousInput
        assert isinstance(
            input, CategoricalInput
        ), f"Expected {CategoricalInput.__name__}, got {type(input)}"
        # Create a new ContinuousInput for each category
        new_auxiliary_inputs_for_input = []
        mapping_aux_to_category = {}
        for c in input.categories:
            new_auxiliary_inputs_for_input.append(
                ContinuousInput(key=generate_value_key(input=input, c=c), bounds=[0, 1])
            )
            mapping_aux_to_category[generate_value_key(input, c)] = c
        new_auxiliary_inputs += new_auxiliary_inputs_for_input
        mappings_aux_to_category[input.key] = mapping_aux_to_category
        # Create a new list of constraints
        new_constraints.append(
            LinearEqualityConstraint(
                features=[i.key for i in new_auxiliary_inputs_for_input],
                coefficients=[1] * len(new_auxiliary_inputs_for_input),
                rhs=1,
            )
        )
    return (
        mappings_aux_to_category,
        new_auxiliary_inputs,
        Constraints(constraints=new_constraints),
    )


def create_continuous_domain(
    domain: Domain,
) -> Tuple[
    Domain, Dict[str, Dict[str, str]], List[ContinuousInput], List[ContinuousInput]
]:
    """
    Creates a domain from the inputs, constraints and outputs.

    Args:
        inputs (Inputs): The inputs of the domain.
        constraints (Constraints): The constraints of the domain.
        outputs (Outputs): The outputs of the domain.

    Returns:
        Domain: The created domain.
    """
    # Create a new domain
    (
        mapped_discrete_inputs,
        mapped_aux_inputs_for_discrete,
        mapped_constraints_for_discrete,
    ) = map_discrete_to_continuous(
        [input for input in domain.inputs if isinstance(input, DiscreteInput)]
    )
    (
        mappings_categorical_inputs,
        mapped_aux_categorical_inputs,
        mapped_constraints_for_categorical,
    ) = map_categorical_to_continuous(
        [input for input in domain.inputs if isinstance(input, CategoricalInput)]
    )
    mapped_continous_inputs = [
        input for input in domain.inputs if isinstance(input, ContinuousInput)
    ]
    # Combine the inputs and constraints
    all_inputs = (
        mapped_discrete_inputs
        + mapped_aux_categorical_inputs
        + mapped_continous_inputs
        + mapped_aux_inputs_for_discrete
    )
    all_constraints = (
        mapped_constraints_for_discrete
        + mapped_constraints_for_categorical
        + domain.constraints
    )
    # Create the domain
    domain = Domain(
        inputs=Inputs(features=all_inputs),
        constraints=all_constraints,
        outputs=domain.outputs,
    )
    # Return the domain
    return (
        domain,
        mappings_categorical_inputs,
        mapped_aux_inputs_for_discrete,
        mapped_aux_categorical_inputs,
    )


def project_df_to_orginal_domain(
    df: pd.DataFrame,
    mapped_aux_inputs_for_discrete: Optional[List[ContinuousInput]] = None,
    mapped_aux_categorical_inputs: Optional[List[ContinuousInput]] = None,
    mappings_categorical_inputs: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Projects the dataframe to the original domain by removing the auxiliary inputs and mapping the discrete inputs
    back to their original values.

    Args:
        df (pd.DataFrame): The dataframe to project.
        mapped_aux_inputs_for_discrete (Inputs): The auxiliary inputs for discrete inputs.
        mapped_aux_categorical_inputs (Inputs): The auxiliary inputs for categorical inputs.
        mappings_categorical_inputs (Dict[str, Dict[str, str]]): The mappings for categorical inputs.

    Returns:
        pd.DataFrame: The projected dataframe.
    """
    # set the categorical inputs according to the auxiliary inputs
    if (mapped_aux_categorical_inputs is not None) and (
        mappings_categorical_inputs is not None
    ):
        for input_key, mapping in mappings_categorical_inputs.items():
            aux_keys = mapping.keys()
            df[input_key] = df[aux_keys].idxmax(axis=1)
            df[input_key] = df[input_key].map(mapping)
        # drop the auxiliary inputs
        df = df.drop(columns=[input.key for input in mapped_aux_categorical_inputs])
    # drop the auxiliary inputs
    if mapped_aux_inputs_for_discrete is not None:
        df = df.drop(columns=[input.key for input in mapped_aux_inputs_for_discrete])
    return df


class DoEStrategy(Strategy):
    """Strategy for design of experiments. This strategy is used to generate a set of
    experiments for a given domain.
    The experiments are generated via minimization of a user defined optimality criterion.

    """

    def __init__(
        self,
        data_model: data_models.DoEStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.data_model = data_model
        self._partially_fixed_candidates = None
        self._fixed_candidates = None

    @property
    def formula(self):
        if isinstance(self.data_model.criterion, DoEOptimalityCriterion):
            return get_formula_from_string(
                self.data_model.criterion.formula, self.data_model.domain
            )
        return None

    def set_candidates(self, candidates: pd.DataFrame):
        original_columns = self.domain.inputs.get_keys(includes=Input)
        to_many_columns = []
        for col in candidates.columns:
            if col not in original_columns:
                to_many_columns.append(col)
        if len(to_many_columns) > 0:
            raise AttributeError(
                f"provided candidates have columns: {*to_many_columns,},  which do not exist in original domain",
            )

        to_few_columns = []
        for col in original_columns:
            if col not in candidates.columns:
                to_few_columns.append(col)
        if len(to_few_columns) > 0:
            raise AttributeError(
                f"provided candidates are missing columns: {*to_few_columns,} which exist in original domain",
            )

        self._candidates = candidates

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:  # type: ignore
        (
            domain,
            mappings_categorical_inputs,
            mapped_aux_inputs_for_discrete,
            mapped_aux_categorical_inputs,
        ) = create_continuous_domain(domain=self.domain)

        # here we adapt the (partially) fixed experiments to the new domain
        fixed_experiments_count = 0
        _candidate_count = candidate_count

        if self.candidates is not None:
            adapted_partially_fixed_candidates = (
                self._transform_candidates_to_new_domain(
                    domain,
                    self.candidates,
                )
            )
        else:
            adapted_partially_fixed_candidates = None

        if self.candidates is not None:
            fixed_experiments_count = self.candidates.notnull().all(axis=1).sum()
            _candidate_count = candidate_count + fixed_experiments_count

        design = find_local_max_ipopt(
            domain,
            n_experiments=_candidate_count,
            fixed_experiments=None,
            partially_fixed_experiments=adapted_partially_fixed_candidates,
            ipopt_options=self.data_model.ipopt_options,
            criterion=self.data_model.criterion,
        )
        design = project_df_to_orginal_domain(
            design,
            mapped_aux_inputs_for_discrete=mapped_aux_inputs_for_discrete,
            mappings_categorical_inputs=mappings_categorical_inputs,
            mapped_aux_categorical_inputs=mapped_aux_categorical_inputs,
        )
        return design.iloc[fixed_experiments_count:, :].reset_index(
            drop=True,
        )

    def get_required_number_of_experiments(self) -> Optional[int]:
        if self.formula:
            return get_n_experiments(self.formula) - n_zero_eigvals(
                domain=self.data_model.domain, model_type=self.formula
            )
        else:
            ValueError(
                f"Only {AnyDoEOptimalityCriterion} type have required number of experiments."
            )

    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise

        """
        return True

    def _transform_candidates_to_new_domain(
        self, domain: Domain, candidates: pd.DataFrame
    ) -> pd.DataFrame:
        new_candidates = candidates.copy()
        new_candidates[
            [key for key in domain.inputs.get_keys() if key not in candidates.columns]
        ] = None
        return new_candidates
