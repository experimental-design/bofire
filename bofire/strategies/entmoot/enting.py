import pandas as pd
import pyomo.environ as pyo
from entmoot.models.enting import Enting
from entmoot.optimizers.pyomo_opt import PyomoOptimizer
from entmoot.problem_config import ProblemConfig
from pydantic import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.strategies.entmoot.problem_config import domain_to_problem_config
from bofire.strategies.strategy import Strategy


class EntingStrategy(Strategy):
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

    def _init_problem_config(self) -> None:
        cfg = domain_to_problem_config(self.domain)
        self._problem_config: ProblemConfig = cfg[0]
        self._model_pyo: pyo.ConcreteModel = cfg[1]

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        if candidate_count > 1:
            raise NotImplementedError("Can currently only handle one at a time")
        opt_pyo = PyomoOptimizer(self._problem_config, params=self._solver_params)
        res = opt_pyo.solve(tree_model=self._enting, model_core=self._model_pyo)
        candidate = res.opt_point
        objective_value = res.opt_val
        unc_unscaled = res.unc_unscaled

        keys = [feat.name for feat in self._problem_config.feat_list]
        candidates = pd.DataFrame(
            data=[candidate + [objective_value, unc_unscaled]],
            index=range(candidate_count),
            columns=keys + ["y_pred", "y_sd"],
        )

        return candidates

    def _tell(self):
        input_keys = self.domain.inputs.get_keys()
        output_keys = self.domain.outputs.get_keys()

        X = self._experiments[input_keys].values
        y = self._experiments[output_keys].values
        self._enting.fit(X, y)

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
