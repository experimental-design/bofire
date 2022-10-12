import inspect
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import pandas as pd
import torch
from everest.domain.constraints import ConcurrencyConstraint, Constraint
from everest.domain.features import (CategoricalDescriptorInputFeature,
                                     CategoricalInputFeature,
                                     ContinuousInputFeature,
                                     ContinuousOutputFeature,
                                     ContinuousOutputFeature_woDesFunc,
                                     InputFeature, OutputFeature)
from everest.strategies.botorch import tkwargs
from everest.strategies.botorch.base import BotorchBasicBoStrategy
from everest.strategies.botorch.utils.objectives import (
    AdditiveObjective, MultiplicativeObjective)

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import (qExpectedImprovement,
                                             qNoisyExpectedImprovement,
                                             qProbabilityOfImprovement,
                                             qUpperConfidenceBound)

# this is aichembuddy's logger
logger = logging.getLogger("everest.botorch.sobo")
logger.setLevel(logging.WARNING)

# we set also the log level of botorch to info to see what is going on there
from botorch.settings import log_level

log_level(logging.WARNING)


class AcquisitionFunctionEnum(Enum):
    QNEI = "QNEI"
    QUCB = "QUCB"
    QEI = "QEI"
    QPI = "QPI"


class BoTorchSoboStrategy(BotorchBasicBoStrategy):
    acquisition_function: AcquisitionFunctionEnum
    acqf: Optional[AcquisitionFunction]
    name: str = "botorch.sobo"

    def _init_acqf(self, df_pending=None) -> None:
        assert self.is_fitted == True, "Model not trained."

        # init acqf
        if self.acquisition_function == AcquisitionFunctionEnum.QNEI:
            self.init_qNEI()
        elif self.acquisition_function == AcquisitionFunctionEnum.QUCB:
            self.init_qUCB()
        elif self.acquisition_function == AcquisitionFunctionEnum.QEI:
            self.init_qEI()
        elif self.acquisition_function == AcquisitionFunctionEnum.QPI:
            self.init_qPI()
        else:
            raise NotImplementedError("ACQF %s is not implemented." % self.acquisition_function)
        
        self.acqf.set_X_pending(df_pending)
        return

    def _init_objective(self):
        self.objective = MultiplicativeObjective(targets=[var.desirability_function for var in self.domain.get_features(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])])
        return

    def get_fbest(self, experiments = None):
        if experiments is None: experiments = self.experiments
        df_valid = self.domain.preprocess_experiments_all_valid_outputs(experiments)
        samples = torch.from_numpy(df_valid[self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])].values).to(**tkwargs)
        return self.objective.forward(samples=samples).detach().numpy().max()

    def init_qNEI(self):
        
        logger.info("Use NEI as ACQF")

        clean_experiments = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)
        transformed = self.transformer.transform(clean_experiments)
        t_features, targets = self.get_training_tensors(transformed, self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]))

        self.acqf = qNoisyExpectedImprovement(
            model=self.model,
            X_baseline=t_features,
            sampler=self.sampler,
            objective=self.objective
        ) 
        return

    def init_qUCB(self,beta=0.2):
        # TODO: handle beta
        logger.info("Use UCB with beta=%4.2f as ACQF" % beta)
        self.acqf= qUpperConfidenceBound(self.model, beta, self.sampler, objective=self.objective)

        return

    def init_qEI(self):
        df = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)
        #df = self.experiments.query(" & ".join(["(valid_%s > 0)" % t for t in self.domain.get_feature_keys(OutputFeature)]) + " & des.notna()")
        best_f = self.get_fbest()
        
        self.acqf = qExpectedImprovement(self.model, best_f, self.sampler,objective=self.objective)
        
        logger.info("Use EI as ACQF. Best_f = %12.6f"  % (best_f))

        return

    def init_qPI(self):
        df = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)
        #df = self.experiments.query(" & ".join(["(valid_%s > 0)" % t for t in self.domain.get_feature_keys(OutputFeature)]) + " & des.notna()")
        best_f = self.get_fbest()
        self.acqf = qProbabilityOfImprovement(self.model,best_f,self.sampler,objective=self.objective) 
        logger.info("Use PI as ACQF. Best_f = %12.6f"  % (best_f))
                           
        return

    @classmethod
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        return True


class BoTorchSoboAdditiveStrategy(BoTorchSoboStrategy):

    name: str = "botorch.sobo.additive"

    def _init_objective(self):
        self.objective = AdditiveObjective(targets=[var.desirability_function for var in self.domain.get_features(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])])
        return



class BoTorchSoboMultiplicativeStrategy(BoTorchSoboStrategy):

    name: str = "botorch.sobo.multiplicative"





    