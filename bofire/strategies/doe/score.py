import pandas as pd

from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.doe import AnyOptimalityCriterion
from bofire.strategies.doe.objective import get_objective_function
from bofire.strategies.doe.utils_categorical_discrete import (
    create_continuous_domain,
    encode_candidates_to_relaxed_domain,
)


def get_optimality_score(
    domain: Domain,
    design: pd.DataFrame,
    criterion: AnyOptimalityCriterion,
) -> float:
    """Evaluates an optimality criterion on a given design.

    The value is the one a `DoEStrategy` with the same `criterion` minimizes, so it can
    be used to compare designs, e.g. a strategy's result against a hand-made design.
    Smaller is always better, which means that some scores are a transformed version of
    the textbook quantity:

    - `DOptimalityCriterion`: $-\\log\\det(X^T X + \\delta I)$
    - `AOptimalityCriterion`: $\\mathrm{tr}((X^T X + \\delta I)^{-1})$
    - `EOptimalityCriterion`: $-\\lambda_{min}(X^T X + \\delta I)$
    - `GOptimalityCriterion`: $\\max \\mathrm{diag}(X (X^T X + \\delta I)^{-1} X^T)$
    - `KOptimalityCriterion`: condition number of $X^T X + \\delta I$
    - `IOptimalityCriterion`: average prediction variance over the design space
    - `SpaceFillingCriterion`: negative sum of the smallest pairwise distances

    Here $X$ is the model matrix of the design, built from the criterion's `formula`.
    `delta` and `transform_range` are taken from the criterion.

    Args:
        domain: Domain the design lives in. Discrete and categorical inputs are
            handled as in `DoEStrategy`.
        design: Design with one row per experiment and one column per input of the
            domain. Further columns are ignored.
        criterion: The criterion to evaluate.

    Returns:
        The value of the criterion for the design.

    Raises:
        ValueError: If the design is empty or lacks a column for an input of the domain.
    """
    if len(design) == 0:
        raise ValueError("The design has to contain at least one experiment.")
    missing = [key for key in domain.inputs.get_keys() if key not in design.columns]
    if missing:
        raise ValueError(f"The design is missing columns for the inputs: {missing}.")

    (
        relaxed_domain,
        mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mapping_discrete_input_to_discrete_aux,
        *_,
    ) = create_continuous_domain(domain=domain)

    relaxed_design = encode_candidates_to_relaxed_domain(
        candidates=design,
        mappings_categorical_var_key_to_aux_var_key_state_pairs=mappings_categorical_var_key_to_aux_var_key_state_pairs,
        mapping_discrete_input_to_discrete_aux=mapping_discrete_input_to_discrete_aux,
        domain=domain,
    )[relaxed_domain.inputs.get_keys()]

    objective_function = get_objective_function(
        criterion,
        domain=relaxed_domain,
        n_experiments=len(design),
        inputs_for_formula=domain.inputs,
    )
    return float(objective_function(relaxed_design.to_numpy(dtype=float).flatten()))
