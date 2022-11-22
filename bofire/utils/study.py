import time
from abc import abstractmethod
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, PrivateAttr, validator
from tqdm import tqdm

from bofire.domain import Domain
from bofire.strategies.strategy import Strategy
from bofire.utils.multiobjective import (
    compute_hypervolume,
    get_pareto_front,
    get_ref_point_mask,
    infer_ref_point,
)


class Study(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    domain: Optional[Domain]
    experiments: Optional[pd.DataFrame]
    candidates: Optional[pd.DataFrame]
    meta: Optional[pd.DataFrame]
    strategy: Optional[Strategy]
    best_possible_f: Optional[float] = None

    def __init__(self,**data):
        super().__init__(**data)
        self.domain = self.setup_domain()
        self.reset()
        return

    @abstractmethod
    def setup_domain(self):
        pass

    @abstractmethod
    def run_candidate_experiments(self, candidates):
        pass

    def get_fbest(self, experiments: Optional[pd.DataFrame]=None):
        return np.nan
        
    def reset(self):
        self.experiments = pd.DataFrame(columns=self.domain.experiment_column_names,dtype='float64')
        self.candidates = pd.DataFrame(columns=self.domain.candidate_column_names,dtype='float64')
        self.meta = pd.DataFrame(columns=["ask_time","experiment_time","tell_time","batch", "fbest", "batchbest"],dtype='float64')
        return

    def optimize(
        self,
        strategy: Strategy,
        num_iterations:int = 100,
        batch_size:int = 1,
        progress_bar:bool = True
    ):
        self.strategy = strategy
        self.strategy.init_domain(self.domain)
        #if len(self.experiments) > 0: self.strategy.tell(self.experiments)

        with tqdm(range(num_iterations),disable=True if progress_bar == False else False, postfix={"fbest":"?",}) as pbar:
            for i in pbar:
                start = time.time()
                candidates, extras = strategy.ask(
                    candidate_count=batch_size,
                    allow_insufficient_experiments=True,
                )
                #self.candidates = self.candidates.append(candidates,ignore_index=True)
                self.candidates = pd.concat((self.candidates,candidates), ignore_index=True)
                ask_time = time.time() - start

                start = time.time()
                experiments =  self.run_candidate_experiments(candidates)
                #self.experiments = self.experiments.append(experiments,ignore_index=True)
                self.experiments = pd.concat((self.experiments, experiments), ignore_index=True)
                experiment_time = time.time() - start

                start = time.time()
                self.strategy.tell(
                    experiments=self.experiments, replace=True
                )
                tell_time = time.time() - start

                fbest = self.get_fbest()
                self.meta = pd.concat((self.meta, pd.DataFrame(data=[[ask_time,experiment_time,tell_time,i,fbest] for j in range(len(candidates))], columns=["ask_time","experiment_time","tell_time","batch","fbest"])),ignore_index=True)
                pbar.set_postfix({"fbest":fbest})
        return


class MetricsEnum(Enum):
    HYPERVOLUME = "HYPERVOLUME"
    STRATEGY = "STRATEGY"


class PoolStudy(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    domain: Domain
    num_starting_experiments: int
    experiments:  pd.DataFrame
    meta: Optional[pd.DataFrame]
    strategy: Optional[Strategy]
    starting_point_generator: Optional[Callable]
    metrics: Optional[MetricsEnum] = None
    ref_point: Optional[dict]

    @validator("domain")
    def validate_feature_count(cls, domain: Domain):
        if len(domain.input_features) == 0:
            raise ValueError("no input feature specified")
        if len(domain.output_features) == 0:
            raise ValueError("no output feature specified")
        return domain

    @validator("experiments")
    def validate_experiments(cls, experiments, values):
        experiments = values["domain"].validate_experiments(experiments)
        # # we pick only those experiments where at least one output is valid
        cleaned = values["domain"].preprocess_experiments_any_valid_output(experiments).copy().reset_index(drop=True)
        if len(experiments) < 2: 
            raise ValueError("too less experiments available for PoolStudy.")
        return cleaned

    @validator("metrics")
    def validate_metrics(cls, metrics, values):
        if metrics is MetricsEnum.HYPERVOLUME and len(values["domain"].output_features.get_by_objective(excludes=None)) < 2:
            raise ValueError("For metrics HYPERVOLUME at least two output features has to be defined.")
        return metrics

    @validator("ref_point")
    def validate_ref_point(cls,ref_point, values):
        if ref_point is None: return None
        if len(ref_point) != len(values["domain"].output_features.get_by_objective(excludes=None)):
            raise ValueError("Length of refpoint does not match number of output features.")
        for feat in values["domain"].output_features.get_keys_by_objective(excludes=None):
            assert feat in ref_point.keys()
        return ref_point

    @staticmethod
    def generate_uniform(experiments: pd.DataFrame, num_starting_experiments):
        assert num_starting_experiments > 0
        return np.random.choice(np.arange(experiments.shape[0]), size=num_starting_experiments, replace=False)

    def __init__(self, **data):
        super().__init__(**data)
        self.meta = pd.DataFrame(index = range(self.experiments.shape[0]), columns = ["iteration"], data=np.nan)
        if self.starting_point_generator is None:
            start_idx = self.generate_uniform(self.experiments, self.num_starting_experiments)
        else:
            start_idx = self.starting_point_generator(self.experiments, self.num_starting_experiments)
        #
        if (self.metrics==MetricsEnum.HYPERVOLUME) and (self.ref_point is None):
            self.ref_point = infer_ref_point(self.domain, self.experiments, return_masked=False)
        self.meta.loc[start_idx,"iteration"] = 0

    @property
    def picked_experiments(self):
        return self.experiments.loc[self.meta.iteration.notna()]

    @property
    def num_picked_experiments(self):
        return self.picked_experiments.shape[0]

    @property
    def open_experiments(self):
        return self.experiments.loc[self.meta.iteration.isna()]

    @property
    def num_open_experiments(self):
        return self.open_experiments.shape[0]

    @property
    def num_iterations(self):
        return int(self.meta.iteration.max())

    def get_fbest(self, experiments: pd.DataFrame):
        if self.metrics is None:
            return np.nan
        elif self.metrics == MetricsEnum.HYPERVOLUME:
            opt_exps = get_pareto_front(self.domain,experiments)
            return compute_hypervolume(domain=self.domain, optimal_experiments=opt_exps, ref_point=self.ref_point)
        elif self.metrics == MetricsEnum.STRATEGY:
            return self.strategy.get_fbest(experiments)

    @property
    def expected_random(self):
        """Expected value for number of random picks until finding the best one.

        According to https://arxiv.org/abs/1404.1161 it is defined as
        E(X) = (N+1)/(K+1) where N is the number of possible solutions and 
        K the number of good solutions.
        """
        K = get_pareto_front(self.experiments).shape[0] if self.metrics == MetricsEnum.HYPERVOLUME else 1
        return (self.experiments.shape[0]+1)/(K+1)

    def optimize(
        self,
        strategy: Strategy,
        num_iterations:int = 100,
        candidate_count:int = 1,
        progress_bar:bool = True,
        early_stopping = False,
    ):
        self.strategy = strategy(self.domain)

        if candidate_count > 1: raise ValueError("batch_size > 1 not yet implemented.")
        if num_iterations < 1: raise ValueError("At least one iteration has to be performed!")
        if num_iterations > self.num_open_experiments:
            num_iterations = self.num_open_experiments
        fbest = self.get_fbest(self.experiments)
        with tqdm(range(num_iterations),disable=True if progress_bar == False else False, postfix={"dist2best":"?"}) as pbar:
            for i in pbar:
                strategy.tell(self.picked_experiments)
                acqf_values = strategy._choose_from_pool(self.open_experiments)
                picked_idx = self.open_experiments.iloc[acqf_values.argmax()].name
                self.meta.loc[picked_idx, "iteration"] = i+1
                cfbest = self.get_fbest(self.picked_experiments)
                pbar.set_postfix({"dist2best":fbest - cfbest})
                if np.allclose(fbest,cfbest) and early_stopping: break
