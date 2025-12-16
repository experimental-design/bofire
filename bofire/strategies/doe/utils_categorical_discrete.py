from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)


def map_discrete_to_continuous(
    inputs: List[DiscreteInput],
) -> Tuple[
    Dict[str, List[str]], List[ContinuousInput], List[ContinuousInput], Constraints
]:
    """
    Maps a discrete input to a constrained set of continuous inputs. Each discrete value is represented by a
    continuous variable between 0 and 1. As only one can be active at a time, the sum of all continuous variables
    is constrained to be 1. The constraint is added to the domain as a NChooseKConstraint. The number of discrete
    values is the number of continuous variables.
    Args:
        inputs: A list of discrete inputs.
    Returns:
        A tuple containing:
            - A dictionary mapping the discrete input key to a list of auxiliary variable keys.
            - A list of continuous inputs representing the discrete values.
            - A list of auxiliary continuous inputs for the discrete values.
            - A Constraints object containing the constraints for the discrete inputs.
    """

    def generate_value_key(input: DiscreteInput, d: float):
        return (
            f"aux_{input.key}_{str(d).replace('.', '__decpt__').replace('-','__neg__')}"
        )

    # Create a new list of inputs
    new_inputs = []
    new_auxiliary_inputs = []
    new_constraints = []
    mappings_discrete_var_key_to_aux_var_keys = {}
    # Iterate over the inputs of the original domain
    for input in inputs:
        # If the input is a DiscreteInput, replace it with a ContinuousInput
        mapping_aux_to_discrete_value = []
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
            mapping_aux_to_discrete_value += [generate_value_key(input, d)]
        new_auxiliary_inputs += new_aux_inputs_for_input
        mappings_discrete_var_key_to_aux_var_keys[input.key] = (
            mapping_aux_to_discrete_value
        )
        # Create a new list of constraints
        new_constraints.append(
            LinearEqualityConstraint(
                features=[i.key for i in new_aux_inputs_for_input],
                coefficients=[1] * len(new_aux_inputs_for_input),
                rhs=1,
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
        mappings_discrete_var_key_to_aux_var_keys,
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
        categorical_inputs: A list of categorical inputs.
    Returns:
        A tuple containing:
            - A dictionary mapping the categorical input key to a dictionary of auxiliary variable keys and their
              corresponding categories.
            - A list of continuous inputs representing the categorical values.
            - A Constraints object containing the constraints for the categorical inputs.
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
    Domain,
    Dict[str, Dict[str, str]],
    Dict[str, List[str]],
    List[ContinuousInput],
    List[ContinuousInput],
    List[ContinuousInput],
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
        mappings_discrete_var_key_to_aux_var_keys,
        relaxed_discrete_inputs,
        aux_vars_for_discrete,
        aux_constraints_for_discrete_var_relaxation,
    ) = map_discrete_to_continuous(
        [input for input in domain.inputs if isinstance(input, DiscreteInput)]
    )
    (
        mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mapped_aux_categorical_inputs,
        aux_constraints_for_categorical_var_relaxation,
    ) = map_categorical_to_continuous(
        [input for input in domain.inputs if isinstance(input, CategoricalInput)]
    )
    mapped_continous_inputs = [
        input for input in domain.inputs if isinstance(input, ContinuousInput)
    ]
    # Combine the inputs and constraints
    all_inputs = (
        relaxed_discrete_inputs
        + mapped_aux_categorical_inputs
        + mapped_continous_inputs
        + aux_vars_for_discrete
    )
    all_constraints = (
        aux_constraints_for_discrete_var_relaxation
        + aux_constraints_for_categorical_var_relaxation
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
        mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mappings_discrete_var_key_to_aux_var_keys,
        aux_vars_for_discrete,
        mapped_aux_categorical_inputs,
        mapped_continous_inputs,
    )


def filter_out_discrete_auxilliary_vars(
    df: pd.DataFrame,
    aux_vars_for_discrete: Optional[List[ContinuousInput]] = None,
) -> pd.DataFrame:
    """
    Projects the dataframe to the original domain by removing the auxiliary inputs and mapping the discrete inputs
    back to their original values.
    Args:
        df (pd.DataFrame): The dataframe to project.
        aux_vars_for_discrete (Inputs): The auxiliary inputs for discrete inputs.
        mapped_aux_categorical_inputs (Inputs): The auxiliary inputs for categorical inputs.
         mappings_categorical_var_key_to_aux_var_key_state_pairs (Dict[str, Dict[str, str]]): The mappings for categorical inputs.
    Returns:
        pd.DataFrame: The projected dataframe.
    """
    # drop the auxiliary inputs
    if aux_vars_for_discrete is not None:
        df = df.drop(columns=[input.key for input in aux_vars_for_discrete])
    return df


def filter_out_categorical_and_categorical_auxilliary_vars(
    df: pd.DataFrame,
    mapped_aux_categorical_inputs: Optional[List[ContinuousInput]] = None,
    mappings_categorical_var_key_to_aux_var_key_state_pairs: Optional[
        Dict[str, Dict[str, str]]
    ] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Projects the dataframe to the original domain by removing the auxiliary inputs and mapping the discrete inputs
    back to their original values.
    Args:
        df (pd.DataFrame): The dataframe to project.
        mappings_categorical_var_key_to_aux_var_key_state_pairs (Dict[str, Dict[str, str]]): The mappings for categorical inputs.
    Returns:
       Tuple[pd.DataFrame, pd.DataFrame]: The projected dataframe.
    The first dataframe contains the variables not associated with the categorical inputs.
    The second dataframe contains the categorical inputs.
    """
    df_categorical = pd.DataFrame(index=df.index)
    # set the categorical inputs according to the auxiliary inputs
    if (mapped_aux_categorical_inputs is not None) and (
        mappings_categorical_var_key_to_aux_var_key_state_pairs is not None
    ):
        for (
            input_key,
            mapping,
        ) in mappings_categorical_var_key_to_aux_var_key_state_pairs.items():
            aux_keys = mapping.keys()
            df[input_key] = df[aux_keys].idxmax(axis=1)
            df[input_key] = df[input_key].map(mapping)
        # drop the auxiliary inputs
        df = df.drop(columns=[input.key for input in mapped_aux_categorical_inputs])
    # drop the categorical inputs
    if mappings_categorical_var_key_to_aux_var_key_state_pairs is not None:
        categorical_inputs = list(
            mappings_categorical_var_key_to_aux_var_key_state_pairs.keys()
        )
        df_categorical = df[categorical_inputs].copy()
        df = df.drop(columns=categorical_inputs)
    return df, df_categorical


def project_candidates_into_domain(
    domain: Domain,
    candidates: pd.DataFrame,
    mapping_discrete_input_to_discrete_aux: Dict[str, List[str]],
    keys_continuous_inputs: List[str],
    scip_params: Optional[Dict] = None,
) -> pd.DataFrame:
    def n_choosek_on_boolean(x, k):
        return cp.sum(x) == k

    def linear_equality_constraint(w, x, y):
        return cp.sum(w @ x) == y

    def linear_inequality_constraint(w, x, y):
        return cp.sum(w @ x) <= y

    def lower_bound(w, x):
        return x >= w

    def upper_bound(w, x):
        return x <= w

    candidates_rounded = candidates.copy()
    n_experiments, _ = candidates_rounded.shape
    # sort all columns to the following order:
    # 1. auxiliary inputs for discrete input 1
    # 2. auxiliary inputs for discrete input 2
    # 3. ...
    # 4. original inputs
    cp_variables = []
    columns = []
    for u, k in mapping_discrete_input_to_discrete_aux.items():
        for i in range(len(k)):
            columns.append(k[i])
        columns.append(u)
    columns += keys_continuous_inputs
    constraints = []
    b = []

    for i in range(n_experiments):
        map_original_inputs_to_cp_variables = {}
        for u, k in mapping_discrete_input_to_discrete_aux.items():
            x = cp.Variable(len(k), boolean=True)

            # add the nchoosek constraint for the discrete input
            # this constraint ensures that only one of the auxiliary inputs is selected
            constraints += [n_choosek_on_boolean(x, 1)]

            cp_variables += [x]
            b += list(candidates_rounded[k].iloc[i, :].values)
            x_u = cp.Variable(1)
            cp_variables += [x_u]
            map_original_inputs_to_cp_variables[u] = x_u

            # enforce that the sum of the auxiliary inputs times the allowed discrete values is equal to the discrete input
            constraints += [
                linear_equality_constraint(domain.inputs.get_by_key(u).values, x, x_u)  # type: ignore
            ]
            b += [candidates_rounded[u].iloc[i]]
        if len(keys_continuous_inputs) > 0:
            y = cp.Variable(len(keys_continuous_inputs))
            cp_variables += [y]
            map_original_inputs_to_cp_variables.update(
                {k: y[i] for i, k in enumerate(keys_continuous_inputs)}
            )
            # add upper and lower bounds for the continuous inputs
            lower_bounds = [
                domain.inputs.get_by_key(continuous_input).bounds[0]  # type: ignore
                for continuous_input in keys_continuous_inputs
            ]
            upper_bounds = [
                domain.inputs.get_by_key(continuous_input).bounds[1]  # type: ignore
                for continuous_input in keys_continuous_inputs
            ]
            constraints += [lower_bound(x=y, w=lower_bounds)]
            constraints += [upper_bound(x=y, w=upper_bounds)]
            b += list(candidates_rounded[keys_continuous_inputs].iloc[i, :].values)

        # get linear inequality contraints in domain
        for constraint in domain.constraints.constraints:
            if isinstance(constraint, LinearInequalityConstraint):
                variables = [
                    map_original_inputs_to_cp_variables[feature]
                    for feature in constraint.features
                ]
                if constraint.rhs is not None:
                    constraints += [
                        linear_inequality_constraint(
                            constraint.coefficients,
                            cp.hstack(variables),
                            constraint.rhs,
                        )
                    ]
                else:
                    raise ValueError(
                        "The right-hand side of the linear inequality constraint must be provided."
                    )
            elif isinstance(constraint, LinearEqualityConstraint):
                variables = [
                    map_original_inputs_to_cp_variables[feature]
                    for feature in constraint.features
                ]
                if constraint.rhs is not None:
                    constraints += [
                        linear_equality_constraint(
                            constraint.coefficients,
                            cp.hstack(variables),
                            constraint.rhs,
                        )
                    ]
                else:
                    raise ValueError(
                        "The right-hand side of the linear equality constraint must be provided."
                    )

    # Create the objective function
    objective = cp.Minimize(
        cp.sum_squares(b - cp.hstack(cp_variables))  # type: ignore
    )
    prob = cp.Problem(objective=objective, constraints=constraints)
    if scip_params is None:
        scip_params = {"numerics/feastol": 1e-8}
    prob.solve(solver="SCIP", scip_params=scip_params)
    return pd.DataFrame(
        index=candidates.index,
        data=np.concatenate([var.value for var in cp_variables], axis=0).reshape(
            candidates.shape[0], candidates.shape[1]
        ),
        columns=columns,
    )


def encode_candidates_to_relaxed_domain(
    candidates: pd.DataFrame,
    mappings_categorical_var_key_to_aux_var_key_state_pairs: Dict[str, Dict[str, str]],
    mapping_discrete_input_to_discrete_aux: Dict[str, List[str]],
    domain: Domain,
) -> pd.DataFrame:
    """
    Encode candidates from the original domain to the relaxed domain with proper auxiliary variable values.

    For categorical inputs, creates one-hot encoding with aux_{key}_{category} naming.
    For discrete inputs, creates auxiliary variables with values based on the discrete input value.
    Continuous inputs are kept as-is.

    Args:
        candidates: DataFrame with candidates in original domain (e.g., x1=5, cat='A')
        mappings_categorical_var_key_to_aux_var_key_state_pairs: Mapping from categorical keys to
            their auxiliary variable names and category values
        mapping_discrete_input_to_discrete_aux: Mapping from discrete input keys to their
            auxiliary variable names
        domain: The original (non-relaxed) domain

    Returns:
        DataFrame with candidates in relaxed domain (e.g., x1=5, aux_cat_A=1, aux_cat_B=0)
    """
    encoded = candidates.copy()

    # Encode categorical inputs
    for (
        cat_key,
        aux_mapping,
    ) in mappings_categorical_var_key_to_aux_var_key_state_pairs.items():
        if cat_key in encoded.columns:
            # For each category, create a column with 1 if that category is active, 0 otherwise
            for aux_key, category_value in aux_mapping.items():
                encoded[aux_key] = (encoded[cat_key] == category_value).astype(float)

    # Encode discrete inputs
    for discrete_key, aux_keys in mapping_discrete_input_to_discrete_aux.items():
        if discrete_key in encoded.columns:
            discrete_input = domain.inputs.get_by_key(discrete_key)
            assert isinstance(discrete_input, DiscreteInput)

            # For each allowed discrete value, create an auxiliary variable
            for aux_key, discrete_value in zip(aux_keys, discrete_input.values):
                # Set to 1 if the discrete input equals this value, 0 otherwise
                encoded[aux_key] = (encoded[discrete_key] == discrete_value).astype(
                    float
                )

    return encoded
