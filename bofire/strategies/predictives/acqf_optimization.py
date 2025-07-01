import copy
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_list,
    optimize_acqf_mixed,
)
from torch import Tensor

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductConstraint,
)
from bofire.data_models.domain.api import Domain
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
)
from bofire.data_models.strategies.api import (
    GeneticAlgorithmOptimizer as GeneticAlgorithmDataModel,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import (
    ShortestPathStrategy as ShortestPathStrategyDataModel,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies import utils
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
    ) -> pd.DataFrame:
        """Optimizes the acquisition function(s) for the given domain and input preprocessing specs.

        Args:
            candidate_count: Number of candidates that should be returned.
            acqfs: List of acquisition functions that should be optimized.
            domain: The domain of the optimization problem.
            input_preprocessing_specs: The input preprocessing specs of the surrogates.
            experiments: The experiments that have been conducted so far.

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
                    experiments=experiments,  # type: ignore
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
    ) -> pd.DataFrame:
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

    @staticmethod
    def _features2idx(
        domain: Domain, input_preprocessing_specs: InputTransformSpecs
    ) -> Dict[str, Tuple[int]]:
        features2idx, _ = domain.inputs._get_transform_info(
            input_preprocessing_specs,
        )
        return features2idx

    @staticmethod
    def _features2names(
        domain, input_preprocessing_specs: InputTransformSpecs
    ) -> Dict[str, Tuple[str]]:
        _, features2names = domain.inputs._get_transform_info(
            input_preprocessing_specs,
        )
        return features2names

    @staticmethod
    def _candidates_tensor_to_dataframe(
        candidates: Tensor,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
    ) -> pd.DataFrame:
        """Converts a tensor of candidates to a pandas Dataframe.

        Args:
            candidates (Tensor): Tensor of candidates returned from `optimize_acqf`.

        Returns:
            pd.DataFrame: Dataframe of candidates.
        """
        # This method is needed here as we use a botorch method to optimize over
        # purely categorical spaces

        features2names = AcquisitionOptimizer._features2names(
            domain, input_preprocessing_specs
        )

        input_feature_keys = [
            item for key in domain.inputs.get_keys() for item in features2names[key]
        ]

        df_candidates = pd.DataFrame(
            data=candidates.detach().numpy(),
            columns=input_feature_keys,
        )

        df_candidates = domain.inputs.inverse_transform(
            df_candidates,
            input_preprocessing_specs,
        )
        return df_candidates

    @staticmethod
    def get_fixed_features(
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
    ) -> Dict[int, float]:
        """Provides the values of all fixed features

        Raises:
            NotImplementedError: [description]

        Returns:
            fixed_features (dict): Dictionary of fixed features, keys are the feature indices, values the transformed feature values

        """

        fixed_features = {}
        features2idx = AcquisitionOptimizer._features2idx(
            domain, input_preprocessing_specs
        )

        for _, feat in enumerate(domain.inputs.get(Input)):
            assert isinstance(feat, Input)
            if feat.fixed_value() is not None:
                fixed_values = feat.fixed_value(
                    transform_type=input_preprocessing_specs.get(feat.key),  # type: ignore
                )
                for j, idx in enumerate(features2idx[feat.key]):
                    fixed_features[idx] = fixed_values[j]  # type: ignore

        return fixed_features

    @staticmethod
    def _optimize_acqf_discrete(
        candidate_count: int,
        acqf: AcquisitionFunction,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        experiments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Optimizes the acquisition function for a discrete search space.

        Args:
            candidate_count: Number of candidates that should be returned.
            acqf: Acquisition function that should be optimized.

        Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
        """
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

        # remove here everything that falls under a CategoricalExcludeConstraint
        filtered_choices = filtered_choices[
            domain.constraints.is_fulfilled(filtered_choices)
        ].copy()

        # translate the filtered choice to torch
        t_choices = torch.from_numpy(
            domain.inputs.transform(
                filtered_choices,
                specs=input_preprocessing_specs,
            ).values,
        ).to(**tkwargs)
        candidates, _ = optimize_acqf_discrete(
            acq_function=acqf,
            q=candidate_count,
            unique=True,
            choices=t_choices,
        )
        return AcquisitionOptimizer._candidates_tensor_to_dataframe(
            candidates=candidates,
            domain=domain,
            input_preprocessing_specs=input_preprocessing_specs,
        )


class BotorchOptimizer(AcquisitionOptimizer):
    def __init__(self, data_model: BotorchOptimizerDataModel):
        self.n_restarts = data_model.n_restarts
        self.n_raw_samples = data_model.n_raw_samples
        self.maxiter = data_model.maxiter
        self.batch_limit = data_model.batch_limit
        self.sequential = data_model.sequential

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
    ) -> pd.DataFrame:
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
            candidate_count=candidate_count,
            acqfs=acqfs,
            bounds=bounds,
            ic_generator=ic_generator,  # type: ignore
            ic_gen_kwargs=ic_gen_kwargs,
            nonlinear_constraints=nonlinears,  # type: ignore
            fixed_features=fixed_features,
            fixed_features_list=fixed_features_list,
            sequential=self.sequential,
        )

        candidates = self._candidates_tensor_to_dataframe(
            candidates, domain, input_preprocessing_specs
        )

        if (
            self.local_search_config is not None
            and has_local_search_region(domain)
            and candidate_count == 1
        ):
            local_candidates, local_acqf_val = self._optimize_acqf_continuous(
                domain=domain,
                candidate_count=candidate_count,
                acqfs=acqfs,
                bounds=local_bounds,
                ic_generator=ic_generator,  # type: ignore
                ic_gen_kwargs=ic_gen_kwargs,
                nonlinear_constraints=nonlinears,  # type: ignore
                fixed_features=fixed_features,
                fixed_features_list=fixed_features_list,
                sequential=self.sequential,
            )
            if self.local_search_config.is_local_step(
                local_acqf_val.item(),
                global_acqf_val.item(),
            ):
                return self._candidates_tensor_to_dataframe(
                    local_candidates, domain, input_preprocessing_specs
                )

            assert experiments is not None
            sp = ShortestPathStrategy(
                data_model=ShortestPathStrategyDataModel(
                    domain=domain,
                    start=experiments.iloc[-1].to_dict(),
                    end=candidates.iloc[-1].to_dict(),
                ),
            )

            step = pd.DataFrame(sp.step(sp.start)).T
            return step

        return candidates

    def _optimize_acqf_continuous(
        self,
        domain: Domain,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        bounds: Tensor,
        ic_generator: Callable,
        ic_gen_kwargs: Dict,
        nonlinear_constraints: List[Callable[[Tensor], float]],
        fixed_features: Optional[Dict[int, float]],
        fixed_features_list: Optional[List[Dict[int, float]]],
        sequential: bool,
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
                sequential=sequential,
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
        bounds = utils.get_torch_bounds_from_domain(domain, input_preprocessing_specs)

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
        """Provides all possible combinations of fixed values

        Returns:
            list_of_fixed_features List[dict]: Each dict contains a combination of fixed values
        """
        methods = [
            self.descriptor_method,
            self.discrete_method,
            self.categorical_method,
        ]

        if all(m == CategoricalMethodEnum.FREE for m in methods):
            return [{}]

        fixed_basis = self.get_fixed_features(
            domain,
            input_preprocessing_specs,
        )

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

        combos = domain.inputs.get_categorical_combinations(
            include=include if include else Input,
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


class GeneticAlgorithmOptimizer(AcquisitionOptimizer):
    """
    Genetic Algorithm for acquisition function optimization, using the Pymoo mixed-type algorithm.

    This optimizer uses a population-based approach to optimize acquisition functions. Currently, only
    single-objective optimization is supported. The algorithm evolves a population of
    candidate solutions over multiple generations using genetic operators such as mutation, crossover,
    and selection.

    - `CategoricalInput` variables, which are treated as one-hot-encoded columns by the model and the acquisition functions, are turned into categorical variables for the GA optimization. In the objective function, these categorical variables are transformed to one-hot-encoded tensors. The object `BofireDomainMixedVars` handles this conversion.
    - `CategoricalDescriptorInput` is also transformed in to a categorical pymoo variable, but transformed into the descriptor space
    - `DiscreteInput` will be converted to an pymoo Integer.

    All transformations are handled in the helper class `BofireDomainMixedVars`

    **Constraints**
    The GA cannot handle equality constraints well. Constraints are therefor handled differently:

    - Constraints of the type `LinearEqualityConstraint`, `LinearInequalityConstraint`, and `NChooseKConstraint` are handled in a "repair-function". This repair function is used by the GA to map all individuals from the population $x$ to the feasible space $x'$. In this case, I implemented a repair-function for an arbitrary mixture of linear equality and inequality constraints with a quadratic programming approach:

    $$
    \\min_{x'} \\left( ||x-x' ||_2^2 \right)
    $$

    s.t.

    $$
    A \\cdot x' = b
    $$

    $$
    G \\cdot x' <= h
    $$

    $$
    lb <= x' <= ub
    $$

    The `NChooseKConstraint` is also handled in the reapir function: For each experiment in the population, the smallest factors are set to 0, if the *max_features* constraint is violated, and the upper bound of the largest feactors is set to an offset (defaults to $1e-3$), if the *min_features* constraint is violated.

    The repair functions are handled in the class `LinearProjection`.

    - Other supported constraints are: `ProductInequalityConstraint` and `NonlinearInequalityConstraint`. `ProductInequalityConstraint` are evaluated by the torch-callable, provided by the `get_nonlinear_constraints` function. `NonlinearInequalityConstraint` are evaluated from the experiments data-frame, by the constraints `__call__` method.


    These are handled by the optimizer.

    `NonlinearEqualityConstraints` are not supported.



    """

    def __init__(self, data_model: GeneticAlgorithmDataModel):
        super().__init__(data_model)
        self.data_model = data_model

    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,  # this is the preprocessing specs for the inputs
        experiments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Main function for optimizing the acquisition function using the genetic algorithm.

        Args:
            candidate_count (int): Number of candidates to generate.
            acqfs (List[AcquisitionFunction]): List of acquisition functions to optimize.
            domain (Domain): The domain of the optimization problem.
            input_preprocessing_specs (InputTransformSpecs): Preprocessing specifications for the inputs.
            experiments (Optional[pd.DataFrame]): Existing experiments, if any.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Optimized candidates and their corresponding objective values.
        """

        # Note: If sequential mode is needed, could be added here, and use the single_shot_optimization function in a loop
        candidates, _ = self._single_shot_optimization(
            domain, input_preprocessing_specs, acqfs, candidate_count
        )

        return self._candidates_tensor_to_dataframe(
            candidates,
            domain,
            input_preprocessing_specs,
        )

    def _single_shot_optimization(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        acqfs: List[AcquisitionFunction],
        q: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Single optimizer call. Either for sequential, or simultaneous optimization of q-experiment proposals

        Args:
            domain (Domain)
            input_preprocessing_specs (InputTransformSpecs): transformation specs, as they are needed for the models in
                the acquisition functions

        Returns
            Tensor: x_opt as (d,) Tensor
            Tensor: f_opt as (n_y,) Tensor
        """
        x_opt, f_opt = utils.run_ga(
            self.data_model,
            domain,
            acqfs,
            q,
            callable_format="torch",
            input_preprocessing_specs=input_preprocessing_specs,
            verbose=self.data_model.verbose,
        )

        return x_opt, f_opt  # type: ignore


OPTIMIZER_MAP: Dict[Type[AcquisitionOptimizerDataModel], Type[AcquisitionOptimizer]] = {
    BotorchOptimizerDataModel: BotorchOptimizer,
    GeneticAlgorithmDataModel: GeneticAlgorithmOptimizer,
}


def get_optimizer(data_model: AcquisitionOptimizerDataModel) -> AcquisitionOptimizer:
    return OPTIMIZER_MAP[type(data_model)](data_model)
