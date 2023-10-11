from typing import List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from entmoot.models.enting import Enting
from entmoot.optimizers.pyomo_opt import PyomoOptimizer
from entmoot.problem_config import ProblemConfig
from pydantic import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.features.api import TInputTransformSpecs
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.strategies.predictives.predictive import PredictiveStrategy
from bofire.utils.entmoot import domain_to_problem_config


class EntingStrategy(PredictiveStrategy):
    """Strategy for selecting new candidates using ENTMOOT"""

    def __init__(
        self,
        data_model: data_models.EntingStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self._init_problem_config()
        self._enting = Enting(self._problem_config, data_model.enting_params)
        self._solver_params = data_model.solver_params
        self._learn_from_candidates_coeff = data_model.learn_from_candidates_coeff

    def _init_problem_config(self) -> None:
        cfg = domain_to_problem_config(self.domain)
        self._problem_config: ProblemConfig = cfg[0]
        self._model_pyo: pyo.ConcreteModel = cfg[1]

    @property
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
        # TODO: implement this properly
        # return self.surrogate_specs.input_preprocessing_specs  # type: ignore
        return {}  # type: ignore

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

    def _add_candidate_as_experiment(self, candidate: pd.DataFrame):
        gamma = self._learn_from_candidates_coeff
        # overestimate for minimisation, underestimate for maximisation
        signs = {
            output.key: -1 if isinstance(output.objective, MaximizeObjective) else 1
            for output in self.domain.outputs
        }
        as_experiment = candidate.assign(
            **{
                key: candidate[f"{key}_pred"]
                + gamma * signs[key] * candidate[f"{key}_sd"]
                for key in self.domain.outputs.get_keys()
            }
        )
        self.tell(as_experiment)

    def _ask(self, candidate_count: PositiveInt = 1) -> pd.DataFrame:
        """Generate a single optimal candidate.

        Args:
            candidate_count (PositiveInt, optional): Number of candidates to be generated. Defaults to 1.

        Returns:
            pd.DataFrame: DataFrame with a single candidate (proposed experiment).
        """
        candidate_lst = []
        for _ in range(candidate_count):
            opt_pyo = PyomoOptimizer(self._problem_config, params=self._solver_params)
            res = opt_pyo.solve(tree_model=self._enting, model_core=self._model_pyo)
            candidate = self._postprocess_candidate(res.opt_point)
            candidate_lst.append(candidate)
            # only refit if generating multiple
            if candidate_count > 1:
                self._add_candidate_as_experiment(candidate)

        return pd.concat(candidate_lst)

    def _fit(self, experiments: pd.DataFrame):
        input_keys = self.domain.inputs.get_keys()
        output_keys = self.domain.outputs.get_keys()

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
