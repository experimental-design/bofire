import copy
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, get_args

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models.gpytorch import GPyTorchModel
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
from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    DiscreteInput,
    Input,
)
from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.strategies.api import BotorchStrategy as DataModel
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.api import (
    ShortestPathStrategy as ShortestPathStrategyDataModel,
)
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.surrogates.api import AnyTrainableSurrogate
from bofire.data_models.types import InputTransformSpecs
from bofire.outlier_detection.outlier_detections import OutlierDetections
from bofire.strategies.predictives.predictive import PredictiveStrategy
from bofire.strategies.random import RandomStrategy
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.utils.torch_tools import (
    get_initial_conditions_generator,
    get_interpoint_constraints,
    get_linear_constraints,
    get_nonlinear_constraints,
    tkwargs,
)


class BotorchStrategy(PredictiveStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.num_restarts = data_model.num_restarts
        self.num_raw_samples = data_model.num_raw_samples
        self.descriptor_method = data_model.descriptor_method
        self.categorical_method = data_model.categorical_method
        self.discrete_method = data_model.discrete_method
        self.surrogate_specs = data_model.surrogate_specs
        if data_model.outlier_detection_specs is not None:
            self.outlier_detection_specs = OutlierDetections(
                data_model=data_model.outlier_detection_specs,
            )
        else:
            self.outlier_detection_specs = None
        self.min_experiments_before_outlier_check = (
            data_model.min_experiments_before_outlier_check
        )
        self.frequency_check = data_model.frequency_check
        self.frequency_hyperopt = data_model.frequency_hyperopt
        self.folds = data_model.folds
        self.surrogates = None
        self.local_search_config = data_model.local_search_config
        self.maxiter = data_model.maxiter
        self.batch_limit = data_model.batch_limit
        torch.manual_seed(self.seed)

    model: Optional[GPyTorchModel] = None

    @property
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        return self.surrogate_specs.input_preprocessing_specs

    @property
    def _features2idx(self) -> Dict[str, Tuple[int]]:
        features2idx, _ = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs,
        )
        return features2idx

    @property
    def _features2names(self) -> Dict[str, Tuple[str]]:
        _, features2names = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs,
        )
        return features2names

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

    def _fit(self, experiments: pd.DataFrame):
        """[summary]

        Args:
            transformed (pd.DataFrame): [description]

        """
        # perform outlier detection
        if self.outlier_detection_specs is not None:
            if (
                self.num_experiments >= self.min_experiments_before_outlier_check
                and self.num_experiments % self.frequency_check == 0
            ):
                experiments = self.outlier_detection_specs.detect(experiments)
        # perform hyperopt
        if (self.frequency_hyperopt > 0) and (
            self.num_experiments % self.frequency_hyperopt == 0
        ):
            # we have to import here to avoid circular imports
            from bofire.runners.hyperoptimize import hyperoptimize

            self.surrogate_specs.surrogates = [  # type: ignore
                (
                    hyperoptimize(
                        surrogate_data=surrogate_data,  # type: ignore
                        training_data=experiments,
                        folds=self.folds,
                    )[0]
                    if isinstance(surrogate_data, get_args(AnyTrainableSurrogate))
                    else surrogate_data
                )
                for surrogate_data in self.surrogate_specs.surrogates
            ]

        # map the surrogate spec, we keep it here as attribute to be able to save/dump
        # the surrogate
        self.surrogates = BotorchSurrogates(data_model=self.surrogate_specs)

        self.surrogates.fit(experiments)
        self.model = self.surrogates.compatibilize(  # type: ignore
            inputs=self.domain.inputs,
            outputs=self.domain.outputs,
        )

    def _predict(self, transformed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        # we are using self.model here for this purpose we have to take the transformed
        # input and further transform it to a torch tensor
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        with torch.no_grad():
            try:
                posterior = self.model.posterior(X=X, observation_noise=True)  # type: ignore
            except (
                NotImplementedError
            ):  # NotImplementedEerror is thrown for MultiTaskGPSurrogate
                posterior = self.model.posterior(X=X, observation_noise=False)  # type: ignore

            if len(posterior.mean.shape) == 2:
                preds = posterior.mean.cpu().detach().numpy()
                stds = np.sqrt(posterior.variance.cpu().detach().numpy())
            elif len(posterior.mean.shape) == 3:
                preds = posterior.mean.mean(dim=0).cpu().detach().numpy()
                stds = np.sqrt(posterior.variance.mean(dim=0).cpu().detach().numpy())
            else:
                raise ValueError("Wrong dimension of posterior mean. Expecting 2 or 3.")
        return preds, stds

    def calc_acquisition(
        self,
        candidates: pd.DataFrame,
        combined: bool = False,
    ) -> np.ndarray:
        """Calculate the acquisition value for a set of experiments.

        Args:
            candidates (pd.DataFrame): Dataframe with experimentes for which the acqf value should be calculated.
            combined (bool, optional): If combined an acquisition value for the whole batch is calculated, else individual ones.
                Defaults to False.

        Returns:
            np.ndarray: Dataframe with the acquisition values.

        """
        acqf = self._get_acqfs(1)[0]

        transformed = self.domain.inputs.transform(
            candidates,
            self.input_preprocessing_specs,
        )
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        if combined is False:
            X = X.unsqueeze(-2)

        with torch.no_grad():
            vals = acqf.forward(X).cpu().detach().numpy()

        return vals

    def _setup_ask(self):
        """Generates argument that can by passed to one of botorch's `optimize_acqf` method."""
        num_categorical_features = len(
            self.domain.inputs.get([CategoricalInput, DiscreteInput]),
        )
        num_categorical_combinations = len(
            self.domain.inputs.get_categorical_combinations(),
        )
        lower, upper = self.domain.inputs.get_bounds(
            specs=self.input_preprocessing_specs,
        )
        bounds = torch.tensor([lower, upper]).to(**tkwargs)
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
            fixed_features = self.get_fixed_features()
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

    def _postprocess_candidates(self, candidates: Tensor) -> pd.DataFrame:
        """Converts a tensor of candidates to a pandas Dataframe.

        Args:
            candidates (Tensor): Tensor of candidates returned from `optimize_acqf`.

        Returns:
            pd.DataFrame: Dataframe with candidates.

        """
        input_feature_keys = [
            item
            for key in self.domain.inputs.get_keys()
            for item in self._features2names[key]
        ]

        df_candidates = pd.DataFrame(
            data=candidates.detach().numpy(),
            columns=input_feature_keys,
        )

        df_candidates = self.domain.inputs.inverse_transform(
            df_candidates,
            self.input_preprocessing_specs,
        )

        preds = self.predict(df_candidates)
        return pd.concat((df_candidates, preds), axis=1)

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
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
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
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
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
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
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

    def _ask(self, candidate_count: Optional[int] = None) -> pd.DataFrame:
        """[summary]

        Args:
            candidate_count (int, optional): [description]. Defaults to 1.

        Returns:
            pd.DataFrame: [description]

        """
        candidate_count = candidate_count or 1
        assert candidate_count > 0, "candidate_count has to be larger than zero."
        if self.experiments is None:
            raise ValueError("No experiments have been provided yet.")

        acqfs = self._get_acqfs(candidate_count)

        # we check here if we have a fully combinatorial search space
        if len(
            self.domain.inputs.get(includes=[DiscreteInput, CategoricalInput]),
        ) == len(self.domain.inputs):
            if len(acqfs) > 1:
                raise NotImplementedError(
                    "Multiple Acqfs are currently not supported for purely combinatorial search spaces.",
                )
            # generate the choices as pandas dataframe
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
                acq_function=acqfs[0],
                q=candidate_count,
                unique=True,
                choices=t_choices,
            )
            return self._postprocess_candidates(candidates=candidates)

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

        return self._postprocess_candidates(candidates=candidates)

    def _tell(self) -> None:
        pass

    @abstractmethod
    def _get_acqfs(self, n: int) -> List[AcquisitionFunction]:
        pass

    def get_fixed_features(self) -> Dict[int, float]:
        """Provides the values of all fixed features

        Raises:
            NotImplementedError: [description]

        Returns:
            fixed_features (dict): Dictionary of fixed features, keys are the feature indices, values the transformed feature values

        """
        fixed_features = {}
        features2idx = self._features2idx

        for _, feat in enumerate(self.domain.inputs.get(Input)):
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

        if (
            self.categorical_method == CategoricalMethodEnum.FREE
            and CategoricalEncodingEnum.ONE_HOT
            in list(self.input_preprocessing_specs.values())
        ):
            # for feat in self.get_true_categorical_features():
            for feat in [
                self.domain.inputs.get_by_key(featkey)
                for featkey in self.domain.inputs.get_keys(CategoricalInput)
                if self.input_preprocessing_specs[featkey]
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
        if (
            self.descriptor_method == CategoricalMethodEnum.FREE
            and CategoricalEncodingEnum.DESCRIPTOR
            in list(self.input_preprocessing_specs.values())
        ):
            # for feat in self.get_true_categorical_features():
            for feat in [
                self.domain.inputs.get_by_key(featkey)
                for featkey in self.domain.inputs.get_keys(CategoricalDescriptorInput)
                if self.input_preprocessing_specs[featkey]
                == CategoricalEncodingEnum.DESCRIPTOR
            ]:
                assert isinstance(feat, CategoricalDescriptorInput)
                if feat.is_fixed() is False:
                    lower, upper = feat.get_bounds(CategoricalEncodingEnum.DESCRIPTOR)
                    for j, idx in enumerate(features2idx[feat.key]):
                        if lower[j] == upper[j]:
                            fixed_features[idx] = lower[j]
        return fixed_features

    def get_categorical_combinations(self) -> List[Dict[int, float]]:
        """Provides all possible combinations of fixed values

        Returns:
            list_of_fixed_features List[dict]: Each dict contains a combination of fixed values

        """
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

    def has_sufficient_experiments(
        self,
    ) -> bool:
        if self.experiments is None:
            return False
        if (
            len(
                self.domain.outputs.preprocess_experiments_all_valid_outputs(
                    experiments=self.experiments,
                ),
            )
            > 1
        ):
            return True
        return False

    def get_acqf_input_tensors(self):
        assert self.experiments is not None
        experiments = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments,
        )

        # TODO: should this be selectable?
        clean_experiments = experiments.drop_duplicates(
            subset=[var.key for var in self.domain.inputs.get(Input)],
            keep="first",
            inplace=False,
        )

        transformed = self.domain.inputs.transform(
            clean_experiments,
            self.input_preprocessing_specs,
        )
        X_train = torch.from_numpy(transformed.values).to(**tkwargs)

        if self.candidates is not None:
            transformed_candidates = self.domain.inputs.transform(
                self.candidates,
                self.input_preprocessing_specs,
            )
            X_pending = torch.from_numpy(transformed_candidates.values).to(**tkwargs)
        else:
            X_pending = None

        return X_train, X_pending

    def get_infeasible_cost(
        self,
        objective: Callable[[Tensor, Tensor], Tensor],
        n_samples=128,
    ) -> Tensor:
        X_train, X_pending = self.get_acqf_input_tensors()
        sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=self.domain))
        samples = sampler.ask(candidate_count=n_samples)
        # we need to transform the samples
        transformed_samples = torch.from_numpy(
            self.domain.inputs.transform(
                samples, self.input_preprocessing_specs
            ).values,
        ).to(**tkwargs)
        X = (
            torch.cat((X_train, X_pending, transformed_samples))
            if X_pending is not None
            else torch.cat((X_train, transformed_samples))
        )
        assert self.model is not None
        return get_infeasible_cost(
            X=X,
            model=self.model,
            objective=objective,  # type: ignore
        )
