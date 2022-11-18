from enum import Enum
from typing import Dict, List, Optional, Type

import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)

from bofire.domain.constraints import Constraint
from bofire.domain.features import ContinuousOutput, Feature
from bofire.domain.objectives import Objective
from bofire.strategies.botorch import tkwargs
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.strategies.botorch.utils.objectives import (
    AdditiveObjective,
    MultiplicativeObjective,
)


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
        self.objective = MultiplicativeObjective(targets=[var.objective for var in self.domain.output_features.get_by_objective(excludes=None)])
        return

    def get_fbest(self, experiments = None):
        if experiments is None: experiments = self.experiments
        df_valid = self.domain.preprocess_experiments_all_valid_outputs(experiments)
        samples = torch.from_numpy(df_valid[self.domain.output_features.get_keys_by_objective(excludes=None)].values).to(**tkwargs)
        return self.objective.forward(samples=samples).detach().numpy().max()

    def init_qNEI(self):

        clean_experiments = self.domain.preprocess_experiments_all_valid_outputs(self.experiments)
        transformed = self.transformer.transform(clean_experiments)
        t_features, targets = self.get_training_tensors(transformed, self.domain.output_features.get_keys_by_objective(excludes=None))

        self.acqf = qNoisyExpectedImprovement(
            model=self.model,
            X_baseline=t_features,
            sampler=self.sampler,
            objective=self.objective
        ) 
        return

    def init_qUCB(self,beta=0.2):
        # TODO: handle beta
        self.acqf= qUpperConfidenceBound(self.model, beta, self.sampler, objective=self.objective)

        return

    def init_qEI(self):
        
        best_f = self.get_fbest()
        
        self.acqf = qExpectedImprovement(self.model, best_f, self.sampler,objective=self.objective)

        return

    def init_qPI(self):
                
        best_f = self.get_fbest()
        self.acqf = qProbabilityOfImprovement(self.model,best_f,self.sampler,objective=self.objective) 
                           
        return
  
    
    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        """Method to check if a objective type is implemented for the strategy

        Args:
            my_type (Type[Objective]): Objective class

        Returns:
            bool: True if the objective type is valid for the strategy chosen, False otherwise
        """
        return True

class BoTorchSoboAdditiveStrategy(BoTorchSoboStrategy):

    name: str = "botorch.sobo.additive"

    def _init_objective(self):
        self.objective = AdditiveObjective(targets=[var.objective for var in self.domain.output_features.get_by_objective(excludes=None)])
        return

class BoTorchSoboMultiplicativeStrategy(BoTorchSoboStrategy):

    name: str = "botorch.sobo.multiplicative"





    