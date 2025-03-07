from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_list,
    optimize_acqf_mixed,
)

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
    ShortestPathStrategy as ShortestPathStrategyDataModel,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.utils.torch_tools import tkwargs


class AcquisitionOptimizer(ABC):
    def __init__(self, data_model: AcquisitionOptimizerDataModel):
        pass

    @abstractmethod
    def optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],  # this is a botorch object
        domain: Domain,  # we generate out of the domain all constraints in the format that is needed by the optimizer
        input_preprocessing_specs: InputTransformSpecs,  # this is the preprocessing specs for the inputs
        # bounds: Tuple[
        #    List[float], List[float]
        # ],  # the bounds are provided by the calling strategy itself and are not
        # generated from the optimizer, this gives the calling strategy the possibility for more control logic
        # as needed for LSRBO or trust region methods
        # wouldn`t the global bounds be defined by the domain, and local optimizations done in the successive calls,
        # but in the BotorchOptimizer?
        # if necessary, the "get_bounds" method can be overridden
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        categorical_method: Optional[CategoricalMethodEnum] = None,
        descriptor_method: Optional[CategoricalMethodEnum] = None,
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
                    transform_type=self.input_preprocessing_specs.get(feat.key),  # type: ignore
                )
                for j, idx in enumerate(features2idx[feat.key]):
                    fixed_features[idx] = fixed_values[j]  # type: ignore

        # in case the optimization method is free and not allowed categories are present
        # one has to fix also them, this is abit of double work as it should be also reflected
        # in the bounds but helps to make it safer

        # this could be removed if we drop support for FREE
        if categorical_method is not None:
            if (
                categorical_method == CategoricalMethodEnum.FREE
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
        if descriptor_method is not None:
            if (
                descriptor_method == CategoricalMethodEnum.FREE
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

    def optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        domain: Domain,
        input_preprocessing_specs: Dict[str, Type],
        bounds: Tuple[List[float], List[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # this is the implementation of the optimizer, here goes _optimize_acqf_continuous

        # we check here if we have a fully combinatorial search space
        # and use _optimize_acqf_discrete in this case
        if len(
            self.domain.inputs.get(includes=[DiscreteInput, CategoricalInput]),
        ) == len(self.domain.inputs):
            if len(acqfs) > 1:
                raise NotImplementedError(
                    "Multiple Acqfs are currently not supported for purely combinatorial search spaces.",
                )
            return self._optimize_acqf_discrete(
                candidate_count=candidate_count,
                acqf=acqfs[0],
            )

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
        ) = self._setup_ask()

        # do the global opt
        candidates, global_acqf_val = self._optimize_acqf_continuous(
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
            and has_local_search_region(self.domain)
            and candidate_count == 1
        ):
            local_candidates, local_acqf_val = self._optimize_acqf_continuous(
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
                return self._postprocess_candidates(candidates=local_candidates)
            sp = ShortestPathStrategy(
                data_model=ShortestPathStrategyDataModel(
                    domain=self.domain,
                    start=self.experiments.iloc[-1].to_dict(),
                    end=self._postprocess_candidates(candidates).iloc[-1].to_dict(),
                ),
            )
            step = pd.DataFrame(sp.step(sp.start)).T
            return pd.concat((step, self.predict(step)), axis=1)

    def _optimize_acqf_continuous(
        self,
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
                n_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=self.domain,
                    constraint=LinearEqualityConstraint,
                ),
                inequality_constraints=get_linear_constraints(
                    domain=self.domain,
                    constraint=LinearInequalityConstraint,
                ),
                nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                fixed_features=fixed_features,
                fixed_features_list=fixed_features_list,
                ic_gen_kwargs=ic_gen_kwargs,
                ic_generator=ic_generator,
                options=self._get_optimizer_options(),  # type: ignore
            )
        elif fixed_features_list:
            candidates, acqf_vals = optimize_acqf_mixed(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                n_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=self.domain,
                    constraint=LinearEqualityConstraint,
                ),
                inequality_constraints=get_linear_constraints(
                    domain=self.domain,
                    constraint=LinearInequalityConstraint,
                ),
                nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                fixed_features_list=fixed_features_list,
                ic_generator=ic_generator,
                ic_gen_kwargs=ic_gen_kwargs,
                options=self._get_optimizer_options(),  # type: ignore
            )
        else:
            interpoints = get_interpoint_constraints(
                domain=self.domain,
                n_candidates=candidate_count,
            )
            candidates, acqf_vals = optimize_acqf(
                acq_function=acqfs[0],
                bounds=bounds,
                q=candidate_count,
                n_restarts=self.n_restarts,
                raw_samples=self.n_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=self.domain,
                    constraint=LinearEqualityConstraint,
                )
                + interpoints,
                inequality_constraints=get_linear_constraints(
                    domain=self.domain,
                    constraint=LinearInequalityConstraint,
                ),
                fixed_features=fixed_features,
                nonlinear_inequality_constraints=nonlinear_constraints,  # type: ignore
                return_best_only=True,
                options=self._get_optimizer_options(),  # type: ignore
                ic_generator=ic_generator,
                **ic_gen_kwargs,
            )
        return candidates, acqf_vals

    def _optimize_acqf_discrete(
        self, candidate_count: int, acqf: AcquisitionFunction
    ) -> pd.DataFrame:
        """Optimizes the acquisition function for a discrete search space.

        Args:
            candidate_count: Number of candidates that should be returned.
            acqf: Acquisition function that should be optimized.

        Returns:
            Generated candidates
        """
        assert self.experiments is not None
        choices = pd.DataFrame.from_dict(
            [  # type: ignore
                {e[0]: e[1] for e in combi}
                for combi in self.domain.inputs.get_categorical_combinations()
            ],
        )
        # adding categorical features that are fixed
        for feat in self.domain.inputs.get_fixed():
            choices[feat.key] = feat.fixed_value()[0]  # type: ignore
        # compare the choices with the training data and remove all that are also part
        # of the training data
        merged = choices.merge(
            self.experiments[self.domain.inputs.get_keys()],
            on=list(choices.columns),
            how="left",
            indicator=True,
        )
        filtered_choices = merged[merged["_merge"] == "left_only"].copy()
        filtered_choices.drop(columns=["_merge"], inplace=True)

        # translate the filtered choice to torch
        t_choices = torch.from_numpy(
            self.domain.inputs.transform(
                filtered_choices,
                specs=self.input_preprocessing_specs,
            ).values,
        ).to(**tkwargs)
        candidates, _ = optimize_acqf_discrete(
            acq_function=acqf,
            q=candidate_count,
            unique=True,
            choices=t_choices,
        )
        return self._postprocess_candidates(candidates=candidates)

    def _get_optimizer_options(self) -> Dict[str, int]:
        """Returns a dictionary of settings passed to `optimize_acqf` controlling
        the behavior of the optimizer.

        Returns:
            Dict[str, int]: The dictionary with the settings.

        """
        return {
            "batch_limit": (  # type: ignore
                self.batch_limit
                if len(
                    self.domain.constraints.get(
                        [NChooseKConstraint, ProductConstraint]
                    ),
                )
                == 0
                else 1
            ),
            "maxiter": self.maxiter,
        }

    def _setup_ask(self):
        """Generates argument that can by passed to one of botorch's `optimize_acqf` method."""
        # this is botorch optimizer dependent code and should be moved to the optimizer
        # the bounds should be removed and we get in _ask

        num_categorical_features = len(
            self.domain.inputs.get([CategoricalInput, DiscreteInput]),
        )
        num_categorical_combinations = len(
            self.domain.inputs.get_categorical_combinations(),
        )
        bounds = self.get_bounds(self.domain)

        # setup local bounds
        assert self.experiments is not None
        local_lower, local_upper = self.domain.inputs.get_bounds(
            specs=self.input_preprocessing_specs,
            reference_experiment=self.experiments.iloc[-1],
        )
        local_bounds = torch.tensor([local_lower, local_upper]).to(**tkwargs)

        # setup nonlinears
        if (
            len(self.domain.constraints.get([NChooseKConstraint, ProductConstraint]))
            == 0
        ):
            ic_generator = None
            ic_gen_kwargs = {}
            nonlinear_constraints = None
        else:
            # TODO: implement LSR-BO also for constraints --> use local bounds
            ic_generator = gen_batch_initial_conditions
            ic_gen_kwargs = {
                "generator": get_initial_conditions_generator(
                    strategy=RandomStrategy(
                        data_model=RandomStrategyDataModel(domain=self.domain),
                    ),
                    transform_specs=self.input_preprocessing_specs,
                ),
            }
            nonlinear_constraints = get_nonlinear_constraints(self.domain)
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
                self.categorical_method,
                self.descriptor_method,
            )
            fixed_features_list = None
        else:
            fixed_features = None
            fixed_features_list = self.get_categorical_combinations()
        return (
            bounds,
            local_bounds,
            ic_generator,
            ic_gen_kwargs,
            nonlinear_constraints,
            fixed_features,
            fixed_features_list,
        )

    def get_categorical_combinations(self) -> List[Dict[int, float]]:
        """Provides all possible combinations of fixed values

        Returns:
            list_of_fixed_features List[dict]: Each dict contains a combination of fixed values

        """
        # this is botorch specific, it should go to the new class

        fixed_basis = self.get_fixed_features()

        methods = [
            self.descriptor_method,
            self.discrete_method,
            self.categorical_method,
        ]

        if all(m == CategoricalMethodEnum.FREE for m in methods):
            return [{}]
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

        combos = self.domain.inputs.get_categorical_combinations(
            include=(include if include else Input),
            exclude=exclude,  # type: ignore
        )
        # now build up the fixed feature list
        if len(combos) == 1:
            return [fixed_basis]
        features2idx = self._features2idx
        list_of_fixed_features = []

        for combo in combos:
            fixed_features = copy.deepcopy(fixed_basis)

            for pair in combo:
                feat, val = pair
                feature = self.domain.inputs.get_by_key(feat)
                if (
                    isinstance(feature, CategoricalDescriptorInput)
                    and self.input_preprocessing_specs[feat]
                    == CategoricalEncodingEnum.DESCRIPTOR
                ):
                    index = feature.categories.index(val)

                    for j, idx in enumerate(features2idx[feat]):
                        fixed_features[idx] = feature.values[index][j]

                elif isinstance(feature, CategoricalMolecularInput):
                    preproc = self.input_preprocessing_specs[feat]
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


OPTIMIZER_MAP: Dict[Type[AcquisitionOptimizerDataModel], Type[AcquisitionOptimizer]] = {
    BotorchOptimizerDataModel: BotorchOptimizer,
}


def get_optimizer(data_model: AcquisitionOptimizerDataModel) -> AcquisitionOptimizer:
    return OPTIMIZER_MAP[type(data_model)](data_model)
