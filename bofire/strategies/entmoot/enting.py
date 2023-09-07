
from typing import Tuple
import pyomo.environ as pyo
import pandas as pd
from pydantic import PositiveInt
from entmoot.models.enting import Enting
from entmoot.problem_config import ProblemConfig
from bofire.strategies.strategy import Strategy
from bofire.data_models.domain.api import Domain
from bofire.strategies.entmoot.problem_config import domain_to_problem_config
import bofire.data_models.strategies.api as data_models 

class EntingStrategy(Strategy):
    """Strategy for selecting new candidates using ENTMOOT
    """

    def __init__(
        self,
        data_model: data_models.EntingStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self._init_problem_config()
        self._enting = Enting(self._problem_config, kwargs)

    def _init_problem_config(self) -> None:
        cfg = domain_to_problem_config(self.domain)
        self._problem_config: ProblemConfig = cfg[0]
        self._model_pyo: pyo.ConcreteModel = cfg[1]

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        # Uses the Entmoot sampler, which is uniform and does not use constraints
        candidates = self._problem_config.get_rnd_sample_numpy(candidate_count)

        keys = [feat.name for feat in self._problem_config.feat_list]
        samples = pd.DataFrame(
            data=candidates.reshape(candidate_count, -1),
            index=range(candidate_count),
            columns=keys
        )

        return samples
    
    def _tell(self):
        input_keys = self.domain.inputs.get_keys()
        output_keys = self.domain.outputs.get_keys()

        X = self._experiments[input_keys].values
        y = self._experiments[output_keys].values
        self._enting.fit(X, y)

    def has_sufficient_experiments(self) -> bool:
        return True