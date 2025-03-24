import copy
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_list,
    optimize_acqf_mixed,
)
import torch
from torch import Tensor
from pymoo.core.problem import Problem as PymooProblem
from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import default as pymoo_default_termination
from pymoo.core.repair import Repair as PymooRepair

import cvxopt

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
    ProductConstraint,
)
from bofire.data_models.domain.api import Domain, Constraints
from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    DiscreteInput,
    Input,
)
from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.strategies.api import (
    AcquisitionOptimizer as AcquisitionOptimizerDataModel,
)
from bofire.data_models.strategies.api import (
    BotorchOptimizer as BotorchOptimizerDataModel,
    GeneticAlgorithm as GeneticAlgorithmDataModel,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import (
    ShortestPathStrategy as ShortestPathStrategyDataModel,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.random import RandomStrategy
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.utils.torch_tools import (
    get_initial_conditions_generator,
    get_interpoint_constraints,
    get_linear_constraints,
    get_nonlinear_constraints,
    tkwargs,
)

class AcquisitionOptimizer(ABC):
    def __init__(self, data_model: AcquisitionOptimizerDataModel):
        self.prefer_exhaustive_search_for_purely_categorical_domains = (
            data_model.prefer_exhaustive_search_for_purely_categorical_domains
        )

    def optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,  # this is the preprocessing specs for the inputs
        experiments: Optional[pd.DataFrame] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimizes the acquisition function(s) for the given domain and input preprocessing specs.

        Args:
            candidate_count (int): Number of candidates that should be returned.
            acqfs (List[AcquisitionFunction]): List of acquisition functions that should be optimized.
            domain (Domain): The domain of the optimization problem.
            input_preprocessing_specs (InputTransformSpecs): The input preprocessing specs.
            experiments

        Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.

        """
        # we check here if we have a fully combinatorial search space
        # and use _optimize_acqf_discrete in this case
        if self.prefer_exhaustive_search_for_purely_categorical_domains:
            if len(
                domain.inputs.get(includes=[DiscreteInput, CategoricalInput]),
            ) == len(domain.inputs):
                if len(acqfs) > 1:
                    raise NotImplementedError(
                        "Multiple Acqfs are currently not supported for purely combinatorial search spaces.",
                    )
                return self._optimize_acqf_discrete(
                    candidate_count=candidate_count,
                    acqf=acqfs[0],
                    domain=domain,
                    input_preprocessing_specs=input_preprocessing_specs,
                    experiments=experiments,
                )

        return self._optimize(
            candidate_count=candidate_count,
            acqfs=acqfs,
            domain=domain,
            input_preprocessing_specs=input_preprocessing_specs,
            experiments=experiments,
        )

    @abstractmethod
    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,  # this is the preprocessing specs for the inputs
        experiments: Optional[pd.DataFrame] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimizes the acquisition function(s) for the given domain and input preprocessing specs.

        Args:
            candidate_count (int): Number of candidates that should be returned.
            acqfs (List[AcquisitionFunction]): List of acquisition functions that should be optimized.
            domain (Domain): The domain of the optimization problem.
            input_preprocessing_specs (InputTransformSpecs): The input preprocessing specs.

        Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.

        """
        pass

    def _features2idx(
        self, domain: Domain, input_preprocessing_specs: InputTransformSpecs
    ) -> Dict[str, Tuple[int]]:
        features2idx, _ = domain.inputs._get_transform_info(
            input_preprocessing_specs,
        )
        return features2idx

    def get_bounds(
        self, domain: Domain, input_preprocessing_specs: InputTransformSpecs
    ) -> torch.Tensor:
        lower, upper = domain.inputs.get_bounds(
            specs=input_preprocessing_specs,
        )
        return torch.tensor([lower, upper]).to(**tkwargs)

    def get_fixed_features(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
    ) -> Dict[int, float]:
        """Provides the values of all fixed features

        Raises:
            NotImplementedError: [description]

        Returns:
            fixed_features (dict): Dictionary of fixed features, keys are the feature indices, values the transformed feature values

        """
        # does this go to the actual optimizer implementation, or is this optimizer agnostic?
        # -> maybe agnostic, and categorical_method, and
        fixed_features = {}
        features2idx = self._features2idx(domain, input_preprocessing_specs)

        for _, feat in enumerate(domain.inputs.get(Input)):
            assert isinstance(feat, Input)
            if feat.fixed_value() is not None:
                fixed_values = feat.fixed_value(
                    transform_type=input_preprocessing_specs.get(feat.key),  # type: ignore
                )
                for j, idx in enumerate(features2idx[feat.key]):
                    fixed_features[idx] = fixed_values[j]  # type: ignore

        return fixed_features

    def _include_exclude_categorical_combinations(
        self, domain: Domain
    ) -> Tuple[Union[List[Type[Input]], None], Union[List[Type[Input]], None]]:
        """Returns include and exclude arguments for get_categorical_combinations methods.

        Returns:
            Tuple[List[Type[Input]], List[Type[Input]]]: Tuple of include and exclude arguments.

        """
        return [Input], None

    def get_categorical_combinations(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
    ) -> List[Dict[int, float]]:
        """Provides all possible combinations of fixed values

        Returns:
            list_of_fixed_features List[dict]: Each dict contains a combination of fixed values

        """
        # this is botorch specific, it should go to the new class

        fixed_basis = self.get_fixed_features(
            domain,
            input_preprocessing_specs,
        )

        include, exclude = self._include_exclude_categorical_combinations(domain)

        combos = domain.inputs.get_categorical_combinations(
            include=include,
            exclude=exclude,  # type: ignore
        )
        # now build up the fixed feature list
        if len(combos) == 1:
            return [fixed_basis]
        features2idx = self._features2idx(domain, input_preprocessing_specs)
        list_of_fixed_features = []

        for combo in combos:
            fixed_features = copy.deepcopy(fixed_basis)

            for pair in combo:
                feat, val = pair
                feature = domain.inputs.get_by_key(feat)
                if (
                    isinstance(feature, CategoricalDescriptorInput)
                    and input_preprocessing_specs[feat]
                    == CategoricalEncodingEnum.DESCRIPTOR
                ):
                    index = feature.categories.index(val)

                    for j, idx in enumerate(features2idx[feat]):
                        fixed_features[idx] = feature.values[index][j]

                elif isinstance(feature, CategoricalMolecularInput):
                    preproc = input_preprocessing_specs[feat]
                    if not isinstance(preproc, MolFeatures):
                        raise ValueError(
                            f"preprocessing for {feat} must be of type AnyMolFeatures"
                        )
                    transformed = feature.to_descriptor_encoding(
                        preproc, pd.Series([val])
                    )
                    for j, idx in enumerate(features2idx[feat]):
                        fixed_features[idx] = transformed.values[0, j]
                elif isinstance(feature, CategoricalInput):
                    # it has to be onehot in this case
                    transformed = feature.to_onehot_encoding(pd.Series([val]))
                    for j, idx in enumerate(features2idx[feat]):
                        fixed_features[idx] = transformed.values[0, j]

                elif isinstance(feature, DiscreteInput):
                    fixed_features[features2idx[feat][0]] = val  # type: ignore

            list_of_fixed_features.append(fixed_features)
        return list_of_fixed_features

    def _optimize_acqf_discrete(
        self,
        candidate_count: int,
        acqf: AcquisitionFunction,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        experiments: Optional[pd.DataFrame] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Optimizes the acquisition function for a discrete search space.

        Args:
            candidate_count: Number of candidates that should be returned.
            acqf: Acquisition function that should be optimized.

        Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
        """
        # assert self.experiments is not None
        choices = pd.DataFrame.from_dict(
            [  # type: ignore
                {e[0]: e[1] for e in combi}
                for combi in domain.inputs.get_categorical_combinations()
            ],
        )
        # adding categorical features that are fixed
        for feat in domain.inputs.get_fixed():
            choices[feat.key] = feat.fixed_value()[0]  # type: ignore
        # compare the choices with the training data and remove all that are also part
        # of the training data
        merged = choices.merge(
            experiments[domain.inputs.get_keys()],
            on=list(choices.columns),
            how="left",
            indicator=True,
        )
        filtered_choices = merged[merged["_merge"] == "left_only"].copy()
        filtered_choices.drop(columns=["_merge"], inplace=True)

        # translate the filtered choice to torch
        t_choices = torch.from_numpy(
            domain.inputs.transform(
                filtered_choices,
                specs=input_preprocessing_specs,
            ).values,
        ).to(**tkwargs)
        return optimize_acqf_discrete(
            acq_function=acqf,
            q=candidate_count,
            unique=True,
            choices=t_choices,
        )
        # return self._postprocess_candidates(candidates=candidates)


class BotorchOptimizer(AcquisitionOptimizer):
    def __init__(self, data_model: BotorchOptimizerDataModel):
        self.n_restarts = data_model.n_restarts
        self.n_raw_samples = data_model.n_raw_samples
        self.maxiter = data_model.maxiter
        self.batch_limit = data_model.batch_limit

        # just for completeness here, we should drop the support for FREE and only go over ones that are also
        # allowed, for more speedy optimization we can user other solvers
        # so this can be remomved
        self.categorical_method = data_model.categorical_method
        self.discrete_method = data_model.discrete_method
        self.descriptor_method = data_model.descriptor_method

        self.local_search_config = data_model.local_search_config

        super().__init__(data_model)

    def _setup(self):
        pass

    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        experiments: Optional[pd.DataFrame] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # this is the implementation of the optimizer, here goes _optimize_acqf_continuous

        # for continuous and mixed search spaces, here different optimizers could
        # be used, so we have to abstract the stuff below
        (
            bounds,
            local_bounds,
            ic_generator,
            ic_gen_kwargs,
            nonlinears,
            fixed_features,
            fixed_features_list,
        ) = self._setup_ask(domain, input_preprocessing_specs, experiments)

        # do the global opt
        candidates, global_acqf_val = self._optimize_acqf_continuous(
            domain=domain,
            input_preprocessing_specs=input_preprocessing_specs,
            candidate_count=candidate_count,
            acqfs=acqfs,
            bounds=bounds,
            ic_generator=ic_generator,  # type: ignore
            ic_gen_kwargs=ic_gen_kwargs,
            nonlinear_constraints=nonlinears,  # type: ignore
            fixed_features=fixed_features,
            fixed_features_list=fixed_features_list,
        )

        if (
            self.local_search_config is not None
            and has_local_search_region(domain)
            and candidate_count == 1
        ):
            local_candidates, local_acqf_val = self._optimize_acqf_continuous(
                domain=domain,
                input_preprocessing_specs=input_preprocessing_specs,
                candidate_count=candidate_count,
                acqfs=acqfs,
                bounds=local_bounds,
                ic_generator=ic_generator,  # type: ignore
                ic_gen_kwargs=ic_gen_kwargs,
                nonlinear_constraints=nonlinears,  # type: ignore
                fixed_features=fixed_features,
                fixed_features_list=fixed_features_list,
            )
            if self.local_search_config.is_local_step(
                local_acqf_val.item(),
                global_acqf_val.item(),
            ):
                return local_candidates, local_acqf_val

            raise NotImplementedError("Johannes to have a look at this")
            sp = ShortestPathStrategy(
                data_model=ShortestPathStrategyDataModel(
                    domain=self.domain,
                    start=self.experiments.iloc[-1].to_dict(),
                    end=self._postprocess_candidates(candidates).iloc[-1].to_dict(),
                ),
            )

            step = pd.DataFrame(sp.step(sp.start)).T
            return pd.concat((step, self.predict(step)), axis=1)

        return candidates, global_acqf_val

    def _optimize_acqf_continuous(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        bounds: Tensor,
        ic_generator: Callable,
        ic_gen_kwargs: Dict,
        nonlinear_constraints: List[Callable[[Tensor], float]],
        fixed_features: Optional[Dict[int, float]],
        fixed_features_list: Optional[List[Dict[int, float]]],
    ) -> Tuple[Tensor, Tensor]:
        if len(acqfs) > 1:
            candidates, acqf_vals = optimize_acqf_list(
                acq_function_list=acqfs,
                bounds=bounds,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearEqualityConstraint,
                ),
                inequality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearInequalityConstraint,
                ),
                nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                fixed_features=fixed_features,
                fixed_features_list=fixed_features_list,
                ic_gen_kwargs=ic_gen_kwargs,
                ic_generator=ic_generator,
                options=self._get_optimizer_options(domain),  # type: ignore
            )
        elif fixed_features_list:
            candidates, acqf_vals = optimize_acqf_mixed(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearEqualityConstraint,
                ),
                inequality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearInequalityConstraint,
                ),
                nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                fixed_features_list=fixed_features_list,
                ic_generator=ic_generator,
                ic_gen_kwargs=ic_gen_kwargs,
                options=self._get_optimizer_options(domain),  # type: ignore
            )
        else:
            interpoints = get_interpoint_constraints(
                domain=domain,
                n_candidates=candidate_count,
            )
            candidates, acqf_vals = optimize_acqf(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearEqualityConstraint,
                )
                + interpoints,
                inequality_constraints=get_linear_constraints(
                    domain=domain,
                    constraint=LinearInequalityConstraint,
                ),
                fixed_features=fixed_features,
                nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                return_best_only=True,
                options=self._get_optimizer_options(domain),  # type: ignore
                ic_generator=ic_generator,
                **ic_gen_kwargs,
            )
        return candidates, acqf_vals

    def _get_optimizer_options(self, domain: Domain) -> Dict[str, int]:
        """Returns a dictionary of settings passed to `optimize_acqf` controlling
        the behavior of the optimizer.

        Returns:
            Dict[str, int]: The dictionary with the settings.

        """
        return {
            "batch_limit": (  # type: ignore
                self.batch_limit
                if len(
                    domain.constraints.get([NChooseKConstraint, ProductConstraint]),
                )
                == 0
                else 1
            ),
            "maxiter": self.maxiter,
        }

    def _setup_ask(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        experiments: Optional[pd.DataFrame] = None,
    ):
        """Generates argument that can by passed to one of botorch's `optimize_acqf` method."""
        # this is botorch optimizer dependent code and should be moved to the optimizer
        # the bounds should be removed and we get in _ask

        num_categorical_features = len(
            domain.inputs.get([CategoricalInput, DiscreteInput]),
        )
        num_categorical_combinations = len(
            domain.inputs.get_categorical_combinations(),
        )
        bounds = self.get_bounds(domain, input_preprocessing_specs)

        # setup local bounds
        assert experiments is not None
        local_lower, local_upper = domain.inputs.get_bounds(
            specs=input_preprocessing_specs,
            reference_experiment=experiments.iloc[-1],
        )
        local_bounds = torch.tensor([local_lower, local_upper]).to(**tkwargs)

        # setup nonlinears
        if len(domain.constraints.get([NChooseKConstraint, ProductConstraint])) == 0:
            ic_generator = None
            ic_gen_kwargs = {}
            nonlinear_constraints = None
        else:
            # TODO: implement LSR-BO also for constraints --> use local bounds
            ic_generator = gen_batch_initial_conditions
            ic_gen_kwargs = {
                "generator": get_initial_conditions_generator(
                    strategy=RandomStrategy(
                        data_model=RandomStrategyDataModel(domain=domain),
                    ),
                    transform_specs=input_preprocessing_specs,
                ),
            }
            nonlinear_constraints = get_nonlinear_constraints(domain)
        # setup fixed features
        if (
            (num_categorical_features == 0)
            or (num_categorical_combinations == 1)
            or (
                all(
                    enc == CategoricalMethodEnum.FREE
                    for enc in [
                        self.categorical_method,
                        self.descriptor_method,
                        self.discrete_method,
                    ]
                )
            )
        ):
            fixed_features = self.get_fixed_features(
                domain,
                input_preprocessing_specs,
            )
            fixed_features_list = None
        else:
            fixed_features = None
            fixed_features_list = self.get_categorical_combinations(
                domain, input_preprocessing_specs
            )
        return (
            bounds,
            local_bounds,
            ic_generator,
            ic_gen_kwargs,
            nonlinear_constraints,
            fixed_features,
            fixed_features_list,
        )

    def get_fixed_features(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
    ) -> Dict[int, float]:
        fixed_features = super().get_fixed_features(domain, input_preprocessing_specs)

        features2idx = self._features2idx(domain, input_preprocessing_specs)

        # in case the optimization method is free and not allowed categories are present
        # one has to fix also them, this is abit of double work as it should be also reflected
        # in the bounds but helps to make it safer

        # this could be removed if we drop support for FREE
        if self.categorical_method is not None:
            if (
                self.categorical_method == CategoricalMethodEnum.FREE
                and CategoricalEncodingEnum.ONE_HOT
                in list(input_preprocessing_specs.values())
            ):
                # for feat in self.get_true_categorical_features():
                for feat in [
                    domain.inputs.get_by_key(featkey)
                    for featkey in domain.inputs.get_keys(CategoricalInput)
                    if input_preprocessing_specs[featkey]
                    == CategoricalEncodingEnum.ONE_HOT
                ]:
                    assert isinstance(feat, CategoricalInput)
                    if feat.is_fixed() is False:
                        for cat in feat.get_forbidden_categories():
                            transformed = feat.to_onehot_encoding(pd.Series([cat]))
                            # we fix those indices to zero where one has a 1 as response from the transformer
                            for j, idx in enumerate(features2idx[feat.key]):
                                if transformed.values[0, j] == 1.0:
                                    fixed_features[idx] = 0

        # for the descriptor ones
        if self.descriptor_method is not None:
            if (
                self.descriptor_method == CategoricalMethodEnum.FREE
                and CategoricalEncodingEnum.DESCRIPTOR
                in list(input_preprocessing_specs.values())
            ):
                # for feat in self.get_true_categorical_features():
                for feat in [
                    domain.inputs.get_by_key(featkey)
                    for featkey in domain.inputs.get_keys(CategoricalDescriptorInput)
                    if input_preprocessing_specs[featkey]
                    == CategoricalEncodingEnum.DESCRIPTOR
                ]:
                    assert isinstance(feat, CategoricalDescriptorInput)
                    if feat.is_fixed() is False:
                        lower, upper = feat.get_bounds(
                            CategoricalEncodingEnum.DESCRIPTOR
                        )
                        for j, idx in enumerate(features2idx[feat.key]):
                            if lower[j] == upper[j]:
                                fixed_features[idx] = lower[j]
        return fixed_features

    def get_categorical_combinations(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
    ) -> List[Dict[int, float]]:
        methods = [
            self.descriptor_method,
            self.discrete_method,
            self.categorical_method,
        ]

        if all(m == CategoricalMethodEnum.FREE for m in methods):
            return [{}]

        return super().get_categorical_combinations(domain, input_preprocessing_specs)

    def _include_exclude_categorical_combinations(
        self, domain: Domain
    ) -> Tuple[Union[List[Type[Input]], None], Union[List[Type[Input]], None]]:
        include = []
        exclude = None

        if self.discrete_method == CategoricalMethodEnum.EXHAUSTIVE:
            include.append(DiscreteInput)

        if self.categorical_method == CategoricalMethodEnum.EXHAUSTIVE:
            include.append(CategoricalInput)
            exclude = CategoricalDescriptorInput

        if self.descriptor_method == CategoricalMethodEnum.EXHAUSTIVE:
            include.append(CategoricalDescriptorInput)
            exclude = None

        if not include:
            include = None

        return include, exclude

class GeneticAlgorithm(AcquisitionOptimizer):
    def __init__(self, data_model: GeneticAlgorithmDataModel):
        super().__init__(data_model)
        self.population_size = data_model.population_size
        self.xtol = data_model.xtol
        self.cvtol = data_model.cvtol
        self.ftol = data_model.ftol
        self.n_max_gen = data_model.n_max_gen
        self.n_max_evals = data_model.n_max_evals


    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,  # this is the preprocessing specs for the inputs
        experiments: Optional[pd.DataFrame] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """main function for optimizing the acquisition function using pymoo's genetic algorithm.

        Note: If sequential mode is needed, could be added here, and use the single_shot_optimization function in a loop
        """

        return self._single_shot_optimization(domain, input_preprocessing_specs, acqfs, candidate_count)

    def _single_shot_optimization(self, domain: Domain, input_preprocessing_specs: InputTransformSpecs,
                                        acqfs: List[AcquisitionFunction], q: int,
                                  ) -> Tuple[Tensor, Tensor]:
        """
        single optimizer call. Either for sequential, or simultaneous optimization of q-experiment proposals

        Returns
            x_opt: (d,) Tensor
            f_opt: (n_y,) Tensor
        """

        # ===== Problem ====
        bounds = self.get_bounds(domain, input_preprocessing_specs)

        class AcqfOptimizationProblem(PymooProblem):
            def __init__(self, acqfs, domain: Domain, q: int):
                self.constraints = domain.constraints.get(includes=[
                    NonlinearEqualityConstraint,
                    NonlinearInequalityConstraint,
                ])  # linear constraints handled in repair function
                n_var = bounds.shape[1] * q
                xl = bounds[0, :].detach().numpy().reshape((1, -1)).repeat(q, axis=0).reshape(-1)
                xu = bounds[1, :].detach().numpy().reshape((1, -1)).repeat(q, axis=0).reshape(-1)
                self.d = bounds.shape[1]
                self.q = q

                #assert len(self.nonlinear_constraints) == 0, "To-Do: Nonlinear Constr."

                super().__init__(
                    n_var=n_var,
                    n_obj=len(acqfs),
                    n_ieq_constr=0,  # len(constraints),  # todo: implement constraints
                    n_eq_constr=0,  # len(constraints),
                    xl=xl,
                    xu=xu,
                    elementwise_evaluation=False,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                n_pop = x.shape[0]
                x = torch.from_numpy(x).to(**tkwargs)
                x = x.reshape((n_pop, self.q, self.d))

                out["F"] = [-acqf(x).detach().numpy().reshape(-1) for acqf in self.bofire_acqfs]


        problem = AcqfOptimizationProblem(
            acqfs, domain, q)


        class LinearProjection(PymooRepair):
            """handles linear equality constraints by mapping to closest legal point in the design space
            using quadratic programming, by projecting to x':

                min(1/2 * ||(x'-x)||_2)

                s.t.
                A*x' = 0
                G*x' <= 0

            we will transform the Problem to the type:

                min( 1/2 * x'^T * P * x' + q*x')

                s.t.
                    ...


            in order to solve this with the performant cvxopt.qp solver
            (https://cvxopt.org/userguide/coneprog.html#quadratic-programming)

            For performance, the problem is solved for the complete generation X = [x1 ; x2; ...]
            where x1, x2, ... are the vectors of each individual in the population

             """

            def __init__(self, domain: Domain, d: int, n_pop: int):
                self.d = d
                self.n_pop = n_pop

                def _eq_constr_to_list(eq_constr_) -> Tuple[List[int], List[float], float]:
                    """decode "get_linear_constraints" output: x-index, coefficients, and b"""
                    index: List[int] = [int(x) for x in (eq_constr_[0].detach().numpy())]
                    coeffs: List[float] = list(eq_constr_[1].detach().numpy())
                    b: float = eq_constr_[2]
                    return index, coeffs, b
                eq_constr = [_eq_constr_to_list(eq_constr_) for eq_constr_ in\
                             get_linear_constraints(domain, LinearEqualityConstraint)]
                ineq_constr = [_eq_constr_to_list(eq_constr_) for eq_constr_ in \
                             get_linear_constraints(domain, LinearInequalityConstraint)]

                def repeated_blkdiag(m: cvxopt.matrix, N: int) -> cvxopt.spmatrix:
                    """ construct large matrix with block-diagolal matrix in the center of arbitrary size"""
                    m_zeros = cvxopt.spmatrix([], [], [], m.size)
                    return cvxopt.sparse([[m_zeros] * i + [m] + [m_zeros] * (N - i - 1) for i in range(N)])

                def vstack(m: List[cvxopt.matrix]) -> cvxopt.matrix:
                    return cvxopt.matrix([[mi] for mi in m])

                def _build_A_b_matrices_for_single_constr(index, coeffs, b)\
                        -> Tuple[cvxopt.spmatrix, cvxopt.matrix]:
                    """ a single-line constraint matrix of the form A*x = b or A*x <= b"""
                    A = cvxopt.spmatrix(coeffs, [0] * len(index), index, (1, self.d))
                    b = cvxopt.matrix(b)
                    return A, b

                def _build_A_b_matrices_for_n_points(constr: List[tuple]) -> Tuple[cvxopt.spmatrix, cvxopt.matrix]:
                    """build big sparse matrix for a constraint A*x = b, or A*x <= b"""

                    # vertically combine all linear equality constr.
                    Ab_single_eq = [_build_A_b_matrices_for_single_constr(*constr_) for constr_ in constr]
                    A = vstack([Ab[0] for Ab in Ab_single_eq])
                    b = vstack([Ab[1] for Ab in Ab_single_eq])
                    # repeat for each x in the population
                    A = repeated_blkdiag(A, self.n_pop)
                    b = cvxopt.matrix([b] * self.n_pop)
                    return A, b

                def _build_G_h_for_box_bounds() -> Tuple[cvxopt.spmatrix, cvxopt.matrix]:
                    """ build linear inequality matrices, such that lb<=x<=ub -> G*x<=h """
                    G_bounds_ = cvxopt.sparse([
                        cvxopt.spmatrix(1, range(self.d), range(self.d)),     # unity matrix
                        cvxopt.spmatrix(-1, range(self.d), range(self.d)),    # negative unity matrix
                    ])
                    lb, ub = (bounds[i, :].detach().numpy() for i in range(2))
                    h_bounds_ = cvxopt.matrix(np.concatenate((ub.reshape(-1), lb.reshape(-1))))
                    G = repeated_blkdiag(G_bounds_, self.n_pop)
                    h = cvxopt.matrix([h_bounds_] * self.n_pop)
                    return G, h

                # Prepare Matrices for solving the estimation problem
                self.P = cvxopt.spmatrix(1.0, range(self.d * self.n_pop), range(self.d * self.n_pop))  # the unit-matrix

                self.A, self.b = _build_A_b_matrices_for_n_points(eq_constr)
                G_bounds, h_bounds = _build_G_h_for_box_bounds()
                G, h = _build_A_b_matrices_for_n_points(ineq_constr)
                self.G, self.h = cvxopt.sparse([G_bounds, G]), cvxopt.sparse([h_bounds, h])

            def _do(self, problem, X, **kwargs):
                X[:, 0] = 1 / 3 * X[:, 1]
                return X

        algorithm_class = PymooGA if len(acqfs) == 1 else PymooNSGA2
        algorithm = algorithm_class(
            pop_size=self.population_size,
            repair=LinearProjection(
                domain=domain,
                d=bounds.shape[1],
                n_pop=self.population_size,
            ),
            # todo: other algorithm options, like n_offspring, crossover-functions etc.
        )

        termination_class = pymoo_default_termination.DefaultSingleObjectiveTermination if len(acqfs) == 1 \
            else pymoo_default_termination.DefaultMultiObjectiveTermination

        termination = termination_class(
            xtol=self.xtol,
            cvtol=self.cvtol,
            ftol=self.ftol,
            n_max_gen=self.n_max_gen,
            n_max_evals=self.n_max_evals,
        )

        res = pymoo_minimize(problem,
                             algorithm,
                             termination,
                             save_history=True,
                             verbose=True)

        x_opt = torch.from_numpy(res.X).to(**tkwargs).reshape(q, -1)
        f_opt = torch.from_numpy(res.F).to(**tkwargs)

        return x_opt, f_opt

OPTIMIZER_MAP: Dict[Type[AcquisitionOptimizerDataModel], Type[AcquisitionOptimizer]] = {
    BotorchOptimizerDataModel: BotorchOptimizer,
    GeneticAlgorithmDataModel: GeneticAlgorithm,
}


def get_optimizer(data_model: AcquisitionOptimizerDataModel) -> AcquisitionOptimizer:
    return OPTIMIZER_MAP[type(data_model)](data_model)
