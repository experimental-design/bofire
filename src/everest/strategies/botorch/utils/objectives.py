from abc import abstractmethod

import numpy as np
import torch
from botorch.acquisition import MCAcquisitionObjective
from everest.domain.desirability_functions import (
    DeltaIdentityDesirabilityFunction, MaxIdentityDesirabilityFunction,
    MaxSigmoidDesirabilityFunction, MinIdentityDesirabilityFunction,
    MinSigmoidDesirabilityFunction, TargetDesirabilityFunction,
    ConstantDesirabilityFunction)
from everest.strategies.botorch import tkwargs


class Objective(MCAcquisitionObjective):
    def __init__(self, targets):
        super().__init__()
        self._targets = targets
        return

    @property
    def targets(self):
        return self._targets

    @property
    def num_targets(self):
        return len(self.targets)

    def reward_constant(self, x, w):
        return torch.ones(x.shape).to(**tkwargs)*w
    
    @abstractmethod
    def reward_target(self,x,w,target_value,tolerance,steepness)-> np.array:
        pass
    
    @abstractmethod
    def reward_min(self,x: np.array, w: np.array, tp: np.array, steepness: np.array,**kwargs)-> np.array:
        pass

    @abstractmethod
    def reward_max(self,x: np.array, w: np.array, tp: np.array, steepness: np.array,**kwargs)-> np.array:
        pass

    @abstractmethod
    def reward_max_identity(self, x: np.array, w: np.array, lower_bound: np.array, upper_bound: np.array) -> np.array:
        pass

    @abstractmethod
    def reward_min_identity(self, x: np.array,w: np.array, lower_bound: np.array, upper_bound: np.array)-> np.array:
        pass
    
    @abstractmethod
    def reward_delta_identity(x: np.array, w: np.array, ref_point: np.array)-> np.array:
        pass

    @abstractmethod  
    def forward(self, samples, X=None):
        pass

    def reward(self,x,desirability_function):
        if isinstance(desirability_function, TargetDesirabilityFunction):
            return self.reward_target(x,**desirability_function.dict()), 0.
        if isinstance(desirability_function, MinSigmoidDesirabilityFunction):
            return self.reward_min(x,**desirability_function.dict()), 0.
        if isinstance(desirability_function, MaxSigmoidDesirabilityFunction):
            return self.reward_max(x,**desirability_function.dict()), 0.
        if isinstance(desirability_function, MaxIdentityDesirabilityFunction):
            return self.reward_max_identity(x,**desirability_function.dict()), 0.
        if isinstance(desirability_function, MinIdentityDesirabilityFunction):
            return self.reward_min_identity(x,**desirability_function.dict()), 0.
        if isinstance(desirability_function, ConstantDesirabilityFunction):
            return self.reward_constant(x,**desirability_function.dict()), 0.
        if isinstance(desirability_function, DeltaIdentityDesirabilityFunction):
            # other option is to use non negative costs --> we first try the identity option
            return self.reward_delta_identity(x,**desirability_function.dict()), 0.
        # we don't have the case ignore?
        # if desirability_function == "ignore":
        #     return torch.ones(x.shape), 0.
        else:
            raise NotImplementedError

class MultiplicativeObjective(Objective):
    def __init__(self, targets):
        super().__init__(targets)
        return
    
    def reward_target(self,x,w,target_value,tolerance,steepness):
        return (1./(1.+torch.exp(-1*steepness*(x-(target_value-tolerance)))) *(1. - 1./(1.+torch.exp(-1.*steepness*(x-(target_value+tolerance))))))**w
        
    def reward_min(self,x,w,tp,steepness,**kwargs):
        return ((1. - 1./(1.+torch.exp(-1.*steepness*(x-tp)))))**w
    
    def reward_max(self,x,w,tp,steepness,**kwargs):
        return (1./(1.+torch.exp(-1.*steepness*(x-tp))))**w

    def reward_max_identity(self, x,w,lower_bound,upper_bound,**kwargs):
        return ((x - lower_bound)/(upper_bound - lower_bound))**w

    def reward_min_identity(self,x,w,lower_bound,upper_bound,**kwargs):
        return -1.*((x - lower_bound)/(upper_bound - lower_bound))**w

    def reward_delta_identity(self, x, w, ref_point, scale, **kwargs):
        return ((ref_point-x)*scale)**w
        
    def forward(self, samples, X=None):
        infeasible_cost = 0.
        reward = 1.

        for i in range(self.num_targets):
            r, infeasible = self.reward(samples[...,i],self.targets[i])
            reward *= r
            infeasible_cost += infeasible #TODO: where do we check the feasibility of the samples? #TODO: 
        return reward-infeasible_cost


class AdditiveObjective(Objective):

    def __init__(self, targets):
        super().__init__(targets)
        return
    
    def reward_target(self,x,w,target_value,tolerance,steepness):
        return w*1./(1.+torch.exp(-1*steepness*(x-(target_value-tolerance)))) *(1. - 1./(1.+torch.exp(-1.*steepness*(x-(target_value+tolerance)))))
        
    def reward_min(self,x,w,tp,steepness,**kwargs):
        return w*(1. - 1./(1.+torch.exp(-1.*steepness*(x-tp))))
    
    def reward_max(self,x,w,tp,steepness,**kwargs):
        return w*1./(1.+torch.exp(-1.*steepness*(x-tp)))
    
    def reward_max_identity(self, x,w,lower_bound,upper_bound, **kwargs):
        return (x - lower_bound)/(upper_bound - lower_bound)*w

    def reward_min_identity(self, x,w,lower_bound,upper_bound, **kwargs):
        return -1.*(x - lower_bound)/(upper_bound - lower_bound)*w

    def reward_delta_identity(self, x,scale,w, ref_point, **kwargs):
        return ((ref_point-x)*scale)*w
    
    def forward(self, samples, X=None):

        infeasible_cost = 0.
        reward = 0.

        for i in range(self.num_targets):
            r, infeasible = self.reward(samples[...,i],self.targets[i])
            reward += r
            infeasible_cost += infeasible
        return reward-infeasible_cost
