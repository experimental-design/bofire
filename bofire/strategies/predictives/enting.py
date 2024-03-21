import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import entmoot.constraints as entconstr  # type: ignore
    import pyomo.environ as pyo  # type: ignore
    from entmoot.models.enting import Enting  # type: ignore
    from entmoot.optimizers.pyomo_opt import PyomoOptimizer  # type: ignore
    from entmoot.problem_config import ProblemConfig  # type: ignore
except ImportError:
    warnings.warn("entmoot not installed, BoFire's `EntingStrategy` cannot be used.")

from typing import Union

from pydantic import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    AnyInput,
    AnyOutput,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.strategies.predictives.predictive import PredictiveStrategy


def domain_to_problem_config(
    domain: Domain, seed: Optional[int] = None
) -> Tuple["ProblemConfig", "pyo.ConcreteModel"]:
    """Convert a set of features and constraints from BoFire to ENTMOOT.

    Problems in BoFire are defined as `Domain`s. Before running an ENTMOOT strategy,
    the problem must be converted to an `entmoot.ProblemConfig`.

    Args:
        domain (Domain): the definition of the optimization problem.
        seed (int, optional): random seed for ENTMOOT problem config.

    Returns:
        A tuple (problem_config, model_pyo), where problem_config is the problem definition
        in an ENTMOOT format, and model_pyo is the Pyomo model containing constraints.
    """
    # entmoot expects int, not np.int64
    seed = int(seed) if not (isinstance(seed, int) or seed is None) else seed
    problem_config = ProblemConfig(seed)

    for input_feature in domain.inputs.get():
        _bofire_feat_to_entmoot(problem_config, input_feature)  # type: ignore

    for output_feature in domain.outputs.get_by_objective(
        includes=[MinimizeObjective, MaximizeObjective]
    ):
        _bofire_output_to_entmoot(problem_config, output_feature)  # type: ignore

    constraints = []
    for constraint in domain.constraints.get():
        constraints.append(_bofire_constraint_to_entmoot(problem_config, constraint))  # type: ignore

    # apply constraints to model
    model_pyo = problem_config.get_pyomo_model_core()
    model_pyo.problem_constraints = pyo.ConstraintList()
    entconstr.ConstraintList(constraints).apply_pyomo_constraints(
        model_pyo, problem_config.feat_list, model_pyo.problem_constraints
    )

    return problem_config, model_pyo


def _bofire_feat_to_entmoot(
    problem_config: "ProblemConfig",
    feature: AnyInput,
) -> None:
    """Given a Bofire `Input`, create an ENTMOOT `FeatureType`.

    Args:
        problem_config (ProblemConfig): An ENTMOOT problem definition, modified in-place.
        feature (AnyInput): An input feature to be added to the problem_config object.
    """
    feat_type = None
    bounds = None
    name = feature.key

    if isinstance(feature, ContinuousInput):
        feat_type = "real"
        bounds = (feature.lower_bound, feature.upper_bound)

    elif isinstance(feature, DiscreteInput):
        x = feature.values
        assert (
            np.all(np.diff(x) == 1) and x[0] % 1 == 0
        ), "Discrete values must be consecutive integers"
        feat_type = "binary" if np.array_equal(x, np.array([0, 1])) else "integer"
        bounds = (int(feature.lower_bound), int(feature.upper_bound))

    elif isinstance(feature, CategoricalInput):
        feat_type = "categorical"
        bounds = tuple(feature.categories)

    else:
        raise NotImplementedError(f"Did not recognise input {feature}")

    problem_config.add_feature(feat_type, bounds, name)


def _bofire_output_to_entmoot(
    problem_config: "ProblemConfig", feature: AnyOutput
) -> None:
    """Given a Bofire `Output`, create an ENTMOOT `MinObjective`.

    If the output feature has a maximise objective, this is added to the problem config as a
    `MinObjective`, and a factor of -1 is introduced in `EntingStrategy`.

    Args:
        problem_config (ProblemConfig): An ENTMOOT problem definition, modified in-place.
        feature (AnyOutput): An output feature to be added to the problem_config object.
    """
    if isinstance(feature.objective, MinimizeObjective):  # type: ignore
        problem_config.add_min_objective(name=feature.key)

    elif isinstance(feature.objective, MaximizeObjective):  # type: ignore
        problem_config.add_max_objective(name=feature.key)

    else:
        raise NotImplementedError(f"Did not recognise output {feature}")


def _bofire_constraint_to_entmoot(
    problem_config: "ProblemConfig",
    constraint: Union[
        LinearEqualityConstraint, LinearInequalityConstraint, NChooseKConstraint
    ],
) -> None:
    """Convert a Bofire `Constraint` to an ENTMOOT `Constraint`.

    Args:
        problem_config (ProblemConfig): An ENTMOOT problem definition.
        constraint (Union[LinearEqualityConstraint, LinearInequalityConstraint, NChooseKConstraint]): A constraint to be applied to the Pyomo model.
    """

    if isinstance(constraint, LinearEqualityConstraint):
        ent_constraint = entconstr.LinearEqualityConstraint(
            feature_keys=constraint.features,
            coefficients=constraint.coefficients,
            rhs=constraint.rhs,
        )

    elif isinstance(constraint, LinearInequalityConstraint):
        ent_constraint = entconstr.LinearInequalityConstraint(
            feature_keys=constraint.features,
            coefficients=constraint.coefficients,
            rhs=constraint.rhs,
        )

    elif isinstance(constraint, NChooseKConstraint):
        ent_constraint = entconstr.NChooseKConstraint(
            feature_keys=constraint.features,
            min_count=constraint.min_count,
            max_count=constraint.max_count,
            none_also_valid=constraint.none_also_valid,
        )

    else:
        raise NotImplementedError("Only linear and nchoosek constraints are supported.")

    return ent_constraint


def _dump_enting_params(data_model: data_models.EntingStrategy) -> dict:
    """Dump the model in the nested structure required for ENTMOOT.

    Returns:
        dict: the nested dictionary of entmoot params.
    """
    return {
        "unc_params": {
            "beta": data_model.beta,
            "bound_coeff": data_model.bound_coeff,
            "acq_sense": data_model.acq_sense,
            "dist_trafo": data_model.dist_trafo,
            "dist_metric": data_model.dist_metric,
            "cat_metric": data_model.cat_metric,
        },
        "tree_train_params": {
            "train_params": {
                "num_boost_round": data_model.num_boost_round,
                "max_depth": data_model.max_depth,
                "min_data_in_leaf": data_model.min_data_in_leaf,
                "min_data_per_group": data_model.min_data_per_group,
                "verbose": data_model.verbose,
            },
        },
    }


def _dump_solver_params(data_model: data_models.EntingStrategy) -> dict:
    """Dump the solver parameters for pyomo.

    Returns:
        dict: the nested dictionary of solver params.
    """
    return {
        "solver_name": data_model.solver_name,
        "verbose": data_model.solver_verbose,
        **data_model.solver_params,
    }


class EntingStrategy(PredictiveStrategy):
    """Strategy for selecting new candidates using ENTMOOT"""

    def __init__(
        self,
        data_model: data_models.EntingStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self._init_problem_config()
        self._enting = Enting(self._problem_config, _dump_enting_params(data_model))
        self._solver_params = _dump_solver_params(data_model)
        self._kappa_fantasy = data_model.kappa_fantasy

    def _init_problem_config(self) -> None:
        cfg = domain_to_problem_config(self.domain, self.seed)
        self._problem_config: ProblemConfig = cfg[0]
        self._model_pyo: pyo.ConcreteModel = cfg[1]

    @property
    def input_preprocessing_specs(self):
        return {}

    def _postprocess_candidate(self, candidate: List) -> pd.DataFrame:
        """Converts a single candidate to a pandas Dataframe with prediction.

        Args:
            candidate (List): List containing the features of the candidate.

        Returns:
            pd.DataFrame: Dataframe with candidate.
        """
        keys = [feat.name for feat in self._problem_config.feat_list]
        df_candidate = pd.DataFrame(
            data=[candidate],
            columns=keys,
        )

        preds = self.predict(df_candidate)

        return pd.concat((df_candidate, preds), axis=1)

    def _fantasy_as_experiment(self, candidates: pd.DataFrame):
        """Fit the model with fantasy candidates.

        The Enting strategy generates a globally optimal candidate. Therefore,
        to generate batch proposals, we sequentially generate 'fantasy' observations
        of the candidate, by adding a multiple of the standard deviation to the
        mean prediction. This behaviour is defined by the `kappa_fantasy` parameter.

        Args:
            candidates (pd.DataFrame): The candidate(s) to make a fantasy observation for.
        """
        kappa = self._kappa_fantasy
        # overestimate for minimisation, underestimate for maximisation
        signs = {
            output.key: -1 if isinstance(output.objective, MaximizeObjective) else 1  # type: ignore
            for output in self.domain.outputs.get_by_objective()
        }
        as_experiment = candidates.assign(
            **{
                key: candidates[f"{key}_pred"]
                + kappa * signs[key] * candidates[f"{key}_sd"]
                for key in self.domain.outputs.get_keys()
            },
            valid_y=True,
        )

        return as_experiment

    def _ask(self, candidate_count: PositiveInt = 1) -> pd.DataFrame:
        """Generates candidates.

        If `candidate_count == 1`, then the globally optimal solution is returned.
        If `candidate_count > 1`, then we use fantasy observations to make sequential
        proposals. Note that since this sequentially generates candidates, it is
        much faster to generate a batch in a single function call, such that each candidate
        is only predicted once.

        If you are using subsequent calls to `EntingStrategy.ask()`, then you must add the candidates to the pending list of candidates, by calling `.ask(pending=True)`.

        Args:
            candidate_count (PositiveInt, optional): Number of candidates to be generated. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame with a candidates.
        """
        # First, fit the model on fantasies generated for any pending candidates
        # This ensures that new points are far from pending candidates
        experiments_plus_fantasy = (
            self.experiments.copy() if self.experiments is not None else pd.DataFrame()
        )
        if self.candidates is not None:
            for i in range(len(self.candidates)):
                # iterate using indices so that each `candidate` is a DataFrame
                candidate = self.candidates[i : i + 1]
                # add prediction from model
                preds = self.predict(candidate)
                candidate = pd.concat((candidate, preds), axis=1)
                as_experiment = self._fantasy_as_experiment(candidate)
                experiments_plus_fantasies = pd.concat(
                    (experiments_plus_fantasy, as_experiment)
                )
                self._fit(experiments_plus_fantasies)

        new_candidates = []
        # Subsequently generate candidates, using fantasies if appropriate
        for i in range(candidate_count):
            opt_pyo = PyomoOptimizer(self._problem_config, params=self._solver_params)
            res = opt_pyo.solve(tree_model=self._enting, model_core=self._model_pyo)
            candidate = self._postprocess_candidate(res.opt_point)
            new_candidates.append(candidate)
            # only retrain with fantasy if not last candidate in batch
            if i < candidate_count - 1:
                as_experiment = self._fantasy_as_experiment(candidate)
                experiments_plus_fantasies = pd.concat(
                    (experiments_plus_fantasy, as_experiment)
                )
                self._fit(experiments_plus_fantasies)

        self._fit(self.experiments)
        return pd.concat(new_candidates)

    def _fit(self, experiments: pd.DataFrame):
        input_keys = self.domain.inputs.get_keys()
        output_keys = self.domain.outputs.get_keys()

        experiments = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            experiments
        )

        X = experiments[input_keys].to_numpy()
        y = experiments[output_keys].to_numpy()
        self._enting.fit(X, y)

    def _predict(self, transformed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = transformed.to_numpy()
        pred = self._enting.predict(X)
        # pred has shape [([mu1], std1), ([mu2], std2), ... ]
        m, v = zip(*pred)
        mean = np.array(m)
        std = np.sqrt(np.array(v)).reshape(-1, 1)
        # std is given combined - copy for each objective
        std = np.tile(std, mean.shape[1])
        return mean, std

    def has_sufficient_experiments(self) -> bool:
        if self.experiments is None:
            return False
        return (
            len(
                self.domain.outputs.preprocess_experiments_all_valid_outputs(
                    experiments=self.experiments
                )
            )
            > 1
        )
