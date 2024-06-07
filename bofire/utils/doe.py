import itertools
import re
import string
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import CategoricalInput, ContinuousInput


def get_confounding_matrix(
    inputs: Inputs,
    design: pd.DataFrame,
    powers: Optional[List[int]] = None,
    interactions: Optional[List[int]] = None,
):
    """Analyzes the confounding of a design and returns the confounding matrix.

    Only takes continuous features into account.

    Args:
        inputs (Inputs): Input features.
        design (pd.DataFrame): Design matrix.
        powers (List[int], optional): List of powers of the individual factors/features that should be considered.
            Integers has to be larger than 1. Defaults to [].
        interactions (List[int], optional): List with interaction levels to be considered.
            Integers has to be larger than 1. Defaults to [2].

    Returns:
        _type_: _description_
    """
    from sklearn.preprocessing import MinMaxScaler

    if len(inputs.get(CategoricalInput)) > 0:
        warnings.warn("Categorical input features will be ignored.")

    keys = inputs.get_keys(ContinuousInput)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_design = pd.DataFrame(
        data=scaler.fit_transform(design[keys]),
        columns=keys,
    )

    # add powers
    if powers is not None:
        for p in powers:
            assert p > 1, "Power has to be at least of degree two."
            for key in keys:
                scaled_design[f"{key}**{p}"] = scaled_design[key] ** p

    # add interactions
    if interactions is None:
        interactions = [2]

    for i in interactions:
        assert i > 1, "Interaction has to be at least of degree two."
        assert i < len(keys) + 1, f"Interaction has to be smaller than {len(keys)+1}."
        for combi in itertools.combinations(keys, i):
            scaled_design[":".join(combi)] = scaled_design[list(combi)].prod(axis=1)

    return scaled_design.corr()


def ff2n(n_factors: int) -> np.ndarray:
    """Computes the full factorial design for a given number of factors.

    Args:
        n_factors: The number of factors.

    Returns:
        The full factorial design.
    """
    return np.array(list(itertools.product([-1, 1], repeat=n_factors)))


def validate_generator(n_factors: int, generator: str) -> str:
    """Validates the generator and thows an error if it is not valid."""

    if len(generator.split(" ")) != n_factors:
        raise ValueError("Generator does not match the number of factors.")
    # clean it and transform it into a list
    generators = [item for item in re.split(r"\-|\s|\+", generator) if item]
    lengthes = [len(i) for i in generators]

    # Indices of single letters (main factors)
    idx_main = [i for i, item in enumerate(lengthes) if item == 1]

    if len(idx_main) == 0:
        raise ValueError("At least one unconfounded main factor is needed.")

    # Check that single letters (main factors) are unique
    if len(idx_main) != len({generators[i] for i in idx_main}):
        raise ValueError("Main factors are confounded with each other.")

    # Check that single letters (main factors) follow the alphabet
    if (
        "".join(sorted([generators[i] for i in idx_main]))
        != string.ascii_lowercase[: len(idx_main)]
    ):
        raise ValueError(
            f'Use the letters `{" ".join(string.ascii_lowercase[: len(idx_main)])}` for the main factors.'
        )

    # Indices of letter combinations.
    idx_combi = [i for i, item in enumerate(generators) if item != 1]

    # check that main factors come before combinations
    if min(idx_combi) > max(idx_main):
        raise ValueError("Main factors have to come before combinations.")

    # Check that letter combinations are unique
    if len(idx_combi) != len({generators[i] for i in idx_combi}):
        raise ValueError("Generators are not unique.")

    # Check that only letters are used in the combinations that are also single letters (main factors)
    if not all(
        set(item).issubset({generators[i] for i in idx_main})
        for item in [generators[i] for i in idx_combi]
    ):
        raise ValueError("Generators are not valid.")

    return generator


def fracfact(gen) -> np.ndarray:
    """Computes the fractional factorial design for a given generator.

    Args:
        gen: The generator.

    Returns:
        The fractional factorial design.
    """
    gen = validate_generator(n_factors=gen.count(" ") + 1, generator=gen)

    generators = [item for item in re.split(r"\-|\s|\+", gen) if item]
    lengthes = [len(i) for i in generators]

    # Indices of single letters (main factors)
    idx_main = [i for i, item in enumerate(lengthes) if item == 1]

    # Indices of letter combinations.
    idx_combi = [i for i, item in enumerate(generators) if item != 1]

    # Check if there are "-" operators in gen
    idx_negative = [
        i for i, item in enumerate(gen.split(" ")) if item[0] == "-"
    ]  # remove empty strings

    # Fill in design with two level factorial design
    H1 = ff2n(len(idx_main))
    H = np.zeros((H1.shape[0], len(lengthes)))
    H[:, idx_main] = H1

    # Recognize combinations and fill in the rest of matrix H2 with the proper
    # products
    for k in idx_combi:
        # For lowercase letters
        xx = np.array([ord(c) for c in generators[k]]) - 97

        H[:, k] = np.prod(H1[:, xx], axis=1)

    # Update design if gen includes "-" operator
    if len(idx_negative) > 0:
        H[:, idx_negative] *= -1

    # Return the fractional factorial design
    return H


def get_alias_structure(gen: str, order: int = 4) -> List[str]:
    """Computes the alias structure of the design matrix. Works only for generators
    with positive signs.

    Args:
        gen: The generator.
        order: The order up to wich the alias structure should be calculated. Defaults to 4.

    Returns:
        The alias structure of the design matrix.
    """
    design = fracfact(gen)

    n_experiments, n_factors = design.shape

    all_names = string.ascii_lowercase + "I"
    factors = range(n_factors)
    all_combinations = itertools.chain.from_iterable(
        (
            itertools.combinations(factors, n)
            for n in range(1, min(n_factors, order) + 1)
        )
    )
    aliases = {n_experiments * "+": [(26,)]}  # 26 is mapped to I

    for combination in all_combinations:
        # positive sign
        contrast = np.prod(
            design[:, combination], axis=1
        )  # this is the product of the combination
        scontrast = "".join(np.where(contrast == 1, "+", "-").tolist())
        aliases[scontrast] = aliases.get(scontrast, [])
        aliases[scontrast].append(combination)  # type: ignore

    aliases_list = []
    for alias in aliases.values():
        aliases_list.append(
            sorted(alias, key=lambda a: (len(a), a))
        )  # sort by length and then by the combination
    aliases_list = sorted(
        aliases_list, key=lambda list: ([len(a) for a in list], list)
    )  # sort by the length of the alias

    aliases_readable = []

    for alias in aliases_list:
        aliases_readable.append(
            " = ".join(["".join([all_names[f] for f in a]) for a in alias])
        )

    return aliases_readable


def get_generator(n_factors: int, n_generators: int) -> str:
    """Computes a generator for a given number of factors and generators.

    Args:
        n_factors: The number of factors.
        n_generators: The number of generators.

    Returns:
        The generator.
    """
    if n_generators == 0:
        return " ".join(list(string.ascii_lowercase[:n_factors]))
    n_base_factors = n_factors - n_generators
    if n_generators == 1:
        if n_base_factors == 1:
            raise ValueError(
                "Design not possible, as main factors are confounded with each other."
            )
        return " ".join(
            list(string.ascii_lowercase[:n_base_factors])
            + [string.ascii_lowercase[:n_base_factors]]
        )
    n_base_factors = n_factors - n_generators
    if n_base_factors - 1 < 2:
        raise ValueError(
            "Design not possible, as main factors are confounded with each other."
        )
    generators = [
        "".join(i)
        for i in (
            itertools.combinations(
                string.ascii_lowercase[:n_base_factors], n_base_factors - 1
            )
        )
    ]
    if len(generators) > n_generators:
        generators = generators[:n_generators]
    elif (n_generators - len(generators) == 1) and (n_base_factors > 1):
        generators += [string.ascii_lowercase[:n_base_factors]]
    elif n_generators - len(generators) >= 1:
        raise ValueError(
            "Design not possible, as main factors are confounded with each other."
        )
    return " ".join(list(string.ascii_lowercase[:n_base_factors]) + generators)
