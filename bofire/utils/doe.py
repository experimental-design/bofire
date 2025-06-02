import itertools
import re
import string
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy
import statsmodels.formula.api as smf
from formulaic import Formula
from formulaic.errors import FormulaSyntaxError

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.strategies.api import DoEStrategy, FractionalFactorialStrategy
from bofire.strategies.doe.utils import linear_and_interactions_formula
from bofire.utils.default_fracfac_generators import (
    default_blocking_generators,
    default_fracfac_generators,
)


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
        assert i < len(keys) + 1, f"Interaction has to be smaller than {len(keys) + 1}."
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
    lengths = [len(i) for i in generators]

    # Indices of single letters (main factors)
    idx_main = [i for i, item in enumerate(lengths) if item == 1]

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
            f"Use the letters `{' '.join(string.ascii_lowercase[: len(idx_main)])}` for the main factors.",
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


def fracfact(gen: str) -> np.ndarray:
    """Computes the fractional factorial design for a given generator.

    Args:
        gen: The generator.

    Returns:
        The fractional factorial design.

    """
    gen = validate_generator(n_factors=gen.count(" ") + 1, generator=gen)

    generators = [item for item in re.split(r"\-|\s|\+", gen) if item]
    lengths = [len(i) for i in generators]

    # Indices of single letters (main factors)
    idx_main = [i for i, item in enumerate(lengths) if item == 1]

    # Indices of letter combinations.
    idx_combi = [i for i, item in enumerate(generators) if item != 1]

    # Check if there are "-" operators in gen
    idx_negative = [
        i for i, item in enumerate(gen.split(" ")) if item[0] == "-"
    ]  # remove empty strings

    # Fill in design with two level factorial design
    H1 = ff2n(len(idx_main))
    H = np.zeros((H1.shape[0], len(lengths)))
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
        order: The order up to which the alias structure should be calculated. Defaults to 4.

    Returns:
        The alias structure of the design matrix.

    """
    design = fracfact(gen)

    n_experiments, n_factors = design.shape

    all_names = string.ascii_lowercase + "I"
    factors = range(n_factors)
    all_combinations = itertools.chain.from_iterable(
        itertools.combinations(factors, n) for n in range(1, min(n_factors, order) + 1)
    )
    aliases = {n_experiments * "+": [(26,)]}  # 26 is mapped to I

    for combination in all_combinations:
        # positive sign
        contrast = np.prod(
            design[:, combination],
            axis=1,
        )  # this is the product of the combination
        scontrast = "".join(np.where(contrast == 1, "+", "-").tolist())
        aliases[scontrast] = aliases.get(scontrast, [])
        aliases[scontrast].append(combination)  # type: ignore

    aliases_list = []
    for alias in aliases.values():
        aliases_list.append(
            sorted(alias, key=lambda a: (len(a), a)),
        )  # sort by length and then by the combination
    aliases_list = sorted(
        aliases_list,
        key=lambda list: ([len(a) for a in list], list),
    )  # sort by the length of the alias

    aliases_readable = []

    for alias in aliases_list:
        aliases_readable.append(
            " = ".join(["".join([all_names[f] for f in a]) for a in alias]),
        )

    return aliases_readable


def get_default_generator(n_factors: int, n_generators: int) -> str:
    """Returns the default generator for a given number of factors and generators.

    In case the combination is not available, the function will raise an error.

    Args:
        n_factors: The number of factors.
        n_generators: The number of generators.

    Returns:
        The generator.

    """
    if n_generators == 0:
        return " ".join(list(string.ascii_lowercase[:n_factors]))
    df_generators = default_fracfac_generators
    n_base_factors = n_factors - n_generators
    if df_generators.loc[
        (df_generators.n_factors == n_factors)
        & (df_generators.n_generators == n_generators)
    ].empty:
        raise ValueError("No generator available for the requested combination.")
    generators = (
        df_generators.loc[
            (df_generators.n_factors == n_factors)
            & (df_generators.n_generators == n_generators),
            "generator",
        ]
        .to_list()[0]
        .split(";")
    )
    assert len(generators) == n_generators, "Number of generators does not match."
    generators = [generator.split("=")[1].strip().lower() for generator in generators]
    return " ".join(list(string.ascii_lowercase[:n_base_factors]) + generators)


def compute_generator(n_factors: int, n_generators: int) -> str:
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
                "Design not possible, as main factors are confounded with each other.",
            )
        return " ".join(
            list(string.ascii_lowercase[:n_base_factors])
            + [string.ascii_lowercase[:n_base_factors]],
        )
    n_base_factors = n_factors - n_generators
    if n_base_factors - 1 < 2:
        raise ValueError(
            "Design not possible, as main factors are confounded with each other.",
        )
    generators = [
        "".join(i)
        for i in (
            itertools.combinations(
                string.ascii_lowercase[:n_base_factors],
                n_base_factors - 1,
            )
        )
    ]
    if len(generators) > n_generators:
        generators = generators[:n_generators]
    elif (n_generators - len(generators) == 1) and (n_base_factors > 1):
        generators += [string.ascii_lowercase[:n_base_factors]]
    elif n_generators - len(generators) >= 1:
        raise ValueError(
            "Design not possible, as main factors are confounded with each other.",
        )
    return " ".join(list(string.ascii_lowercase[:n_base_factors]) + generators)


def get_generator(n_factors: int, n_generators: int) -> str:
    """Returns a generator for a given number of factors and generators.

    If the requested combination is available in the default generators, it will return
    this one. Otherwise, it will compute a new one using `get_bofire_generator`.

    Args:
        n_factors: The number of factors.
        n_generators: The number of generators.

    Returns:
        The generator.

    """
    try:
        return get_default_generator(n_factors, n_generators)
    except ValueError:
        return compute_generator(n_factors, n_generators)


def get_block_generator(
    n_factors: int, n_generators: int, n_repetitions: int, n_blocks: int
) -> str:
    """Gets the block generator for a given number of factors, generators, repetitions, and blocks.

    Should be only used if blocking cannot be reached by repetitions only.

    Args:
        n_factors: number of factors
        n_generators: number of generators/reducing factors
        n_repetitions: number of repetitions
        n_blocks: number of blocks that should be realized

    Raises:
        ValueError: If blocking can be reached by repetitions only.

    Returns:
        The blocking generator.
    """
    if n_repetitions % n_blocks == 0:
        raise ValueError("Blocking can be reached by repetitions only.")

    possible_blocks = sorted(
        set(
            default_blocking_generators.loc[
                default_blocking_generators.n_factors == n_factors
            ].n_blocks.to_list()
        )
    )

    if n_blocks in possible_blocks:
        return default_blocking_generators.loc[
            (default_blocking_generators.n_factors == n_factors)
            & (default_blocking_generators.n_blocks == n_blocks)
        ].block_generator.to_list()[0]

    for b in possible_blocks:
        if b * n_repetitions % n_blocks == 0:
            return default_blocking_generators.loc[
                (default_blocking_generators.n_factors == n_factors)
                & (default_blocking_generators.n_blocks == b)
            ].block_generator.to_list()[0]

    raise ValueError("No block generator available for the requested combination.")


def get_n_blocks(n_factors: int, n_generators: int, n_repetitions: int) -> List[int]:
    """Computes the number of possible blocks for a given number of factors, generators, and repetitions.

    Args:
        n_factors: number of factors
        n_generators: number of generators/reducing factors
        n_repetitions: number of repetitions

    Returns:
        List[int]: List of possible number of blocks.
    """
    n_blocks = []
    # check if no repetitions are planned
    if n_repetitions == 1:
        return sorted(
            set(
                default_blocking_generators.loc[
                    default_blocking_generators.n_factors == n_factors
                ].n_blocks.to_list()
            )
        )
    else:
        # check if blocking can be reached just by repetitions
        for i in range(2, n_repetitions + 1):
            if n_repetitions % i == 0:
                n_blocks.append(i)

        # check if blocking can be reached by a combination of explicit blocks and repetitions
        n_blocks += (
            default_blocking_generators.loc[
                default_blocking_generators.n_factors == n_factors, "n_blocks"
            ].to_numpy()
            * (n_repetitions)
        ).tolist() + default_blocking_generators.loc[
            default_blocking_generators.n_factors == n_factors, "n_blocks"
        ].to_list()

        return sorted(set(n_blocks))


def apply_block_generator(design: np.ndarray, gen: str) -> List[int]:
    """Applies blocking to a design matrix.

    Args:
        design: The design matrix.
        gen: The generator.

    Returns:
        List of integers which assigns an experiment in the design matrix to a block.

    """
    generators = [i.strip().lower() for i in gen.split(";")]

    # Fill in design with two level factorial design
    blocking_design = np.zeros((design.shape[0], len(generators)))

    # Recognize combinations and fill in the rest of matrix H2 with the proper
    # products
    for i, g in enumerate(generators):
        # For lowercase letters
        xx = np.array([ord(c) for c in g]) - 97
        blocking_design[:, i] = np.prod(design[:, xx], axis=1)

    # Create a list to store the block assignments
    blocks = []

    # Iterate over each row in the design matrix
    for row in blocking_design:
        # Find the index of the unique row in the blocking design
        block = np.where((blocking_design == row).all(axis=1))[0][0]
        blocks.append(block)

    return blocks


class Term(BaseModel):
    name: str
    effect: float
    coefficient: float
    uncoded_coefficient: float
    error: float
    t_value: float
    p_value: float
    vif_value: Optional[float] = None


class DoEAnalysisResult(BaseModel):
    terms: List[Term]
    critical_t_value: float
    r_squared: float
    adjusted_r_squared: float
    f_value: float
    aic: float
    bic: float
    fitted_values: List[float]
    residues: List[float]


class DoEAnalysis:
    def __init__(
        self,
        experiments: pd.DataFrame,
        strategy_data: Union[DoEStrategy, FractionalFactorialStrategy],
        formula: Optional[str] = None,
    ):
        self._strategy_data = strategy_data

        if len(self.domain.outputs.get(ContinuousOutput)) != 1:
            raise ValueError("Domain has to have exactly one continuous output.")

        if len(self.domain.inputs.get([ContinuousInput, DiscreteInput])) != len(
            self.domain.inputs
        ):
            raise ValueError("All inputs have to be continuous or discrete.")

        if formula is not None:
            try:
                Formula(formula)
            except FormulaSyntaxError:
                raise ValueError(f"Invalid formula: {formula}")
            self._formula = formula
        else:
            if isinstance(strategy_data, DoEStrategy):
                self._formula = strategy_data.optimality_criterion.formula
            else:
                self._formula = linear_and_interactions_formula(self.domain.inputs)

        # filter the experiments
        self._experiments = (
            self.domain.outputs.preprocess_experiments_all_valid_outputs(experiments)
        )
        if isinstance(self._strategy_data, FractionalFactorialStrategy):
            if FractionalFactorialStrategy.n_center >= 1:
                self._experiments["centerpoint"] = 0
                is_center = np.all(
                    np.isclose(
                        self.coded_experiments[self.domain.inputs.get_keys()],
                        0,
                        atol=1e-6,
                    ),
                    axis=1,
                )
                self._experiments.loc[is_center, "centerpoint"] = 1

        # check that there are more experiments than terms in the formula
        n_terms = len(Formula(self._formula).terms + 1)
        if len(self._experiments) <= n_terms:
            raise ValueError(
                f"Number of experiments ({len(experiments)}) has to be larger than the number of terms in the formula ({n_terms}).",
            )

    @property
    def include_centerpoint(self) -> bool:
        """Check if the centerpoint is included in the experiments."""
        return "centerpoint" in self.experiments.columns

    @property
    def domain(self) -> Domain:
        return self._strategy_data.domain

    @property
    def formula(self) -> str:
        return self._formula

    @property
    def experiments(self) -> pd.DataFrame:
        return self._experiments

    @property
    def coded_experiments(self) -> pd.DataFrame:
        from sklearn.preprocessing import MinMaxScaler

        coded_experiments = self.experiments.copy()
        coded_experiments[self.domain.inputs.get_keys()] = MinMaxScaler(
            feature_range=(-1, 1)
        ).fit_transform(coded_experiments[self.domain.inputs.get_keys()])
        return coded_experiments

    def _get_formula_for_ols(self) -> str:
        if self.include_centerpoint:
            return f"{self.domain.outputs.get_keys()[0]} ~ {self.formula} + centerpoint"
        return f"{self.domain.outputs.get_keys()[0]} ~ {self.formula}"

    def _ols(self):
        model = smf.ols(
            formula=self._get_formula_for_ols(),
            data=self.experiments,
        ).fit()
        return model.fit()

    def _coded_ols(self):
        model = smf.ols(
            formula=self._get_formula_for_ols(),
            data=self.coded_experiments,
        ).fit()
        return model.fit()

    def __call__(self) -> DoEAnalysisResult:
        results = self._ols()
        coded_results = self._coded_ols()

        # extract the terms
        terms = []
        for term in coded_results.model.exog_names:
            terms.append(
                Term(
                    name=term,
                    effect=coded_results.params[term] * 2,
                    coefficient=coded_results.params[term],
                    uncoded_coefficient=results.params[term],
                    error=coded_results.bse[term],
                    t_value=coded_results.tvalues[term],
                    p_value=coded_results.pvalues[term],
                    vif_value=None,
                )
            )

        # doe analysis
        doe_analysis_result = DoEAnalysisResult(
            terms=terms,
            critical_t_value=abs(scipy.stats.t.ppf(0.975, coded_results.df_resid)),
            r_squared=coded_results.rsquared,
            adjusted_r_squared=coded_results.rsquared_adj,
            f_value=coded_results.fvalue,
            aic=coded_results.aic,
            bic=coded_results.bic,
            fitted_values=coded_results.fittedvalues.tolist(),
            residues=coded_results.resid.tolist(),
        )
        return doe_analysis_result
