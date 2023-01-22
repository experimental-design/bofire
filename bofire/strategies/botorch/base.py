import copy
from abc import abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from pydantic import PositiveInt
from pydantic.class_validators import root_validator, validator
from pydantic.types import NonNegativeInt, conlist

from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.domain import Domain
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    InputFeature,
    OutputFeatures,
    TInputTransformSpecs,
)
from bofire.models.torch_models import (
    BotorchModels,
    MixedSingleTaskGPModel,
    SingleTaskGPModel,
)
from bofire.strategies.strategy import PredictiveStrategy
from bofire.strategies.utils import is_power_of_two
from bofire.utils.enum import (  # DescriptorMethodEnum,
    CategoricalEncodingEnum,
    CategoricalMethodEnum,
)
from bofire.utils.torch_tools import get_linear_constraints, tkwargs

Tkeys = conlist(item_type=str, min_items=1)


class BotorchBasicBoStrategy(PredictiveStrategy):
    num_sobol_samples: PositiveInt = 512
    num_restarts: PositiveInt = 8
    num_raw_samples: PositiveInt = 1024
    descriptor_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    categorical_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    model_specs: Optional[BotorchModels] = None

    # private ones
    objective: Optional[MCAcquisitionObjective] = None
    acqf: Optional[AcquisitionFunction] = None
    model: Optional[GPyTorchModel] = None
    # use_combined_bounds: bool = True  # parameter to switch to legacy behavior

    @validator("num_sobol_samples")
    def validate_num_sobol_samples(cls, v):
        if is_power_of_two(v) is False:
            raise ValueError(
                "number sobol samples have to be of the power of 2 to increase performance"
            )
        return v

    @validator("num_raw_samples")
    def validate_num_raw_samples(cls, v):
        if is_power_of_two(v) is False:
            raise ValueError(
                "number raw samples have to be of the power of 2 to increase performance"
            )
        return v

    @root_validator(pre=False, skip_on_failure=True)
    def update_model_specs_for_domain(cls, values):
        """Ensures that a prediction model is specified for each output feature"""
        values["model_specs"] = BotorchBasicBoStrategy._generate_model_specs(
            values["domain"],
            values["model_specs"],
        )
        # we also have to checke here that the categorical method is compatible with the chosen models
        if values["categorical_method"] == CategoricalMethodEnum.FREE:
            for m in values["model_specs"].models:
                if isinstance(m, MixedSingleTaskGPModel):
                    raise ValueError(
                        "Categorical method FREE not compatible with a a MixedSingleTaskGPModel."
                    )
        #  we also check that if a categorical with descriptor method is used as one hot encoded the same method is
        # used for the descriptor as for the categoricals
        for m in values["model_specs"].models:
            keys = m.input_features.get_keys(CategoricalDescriptorInput)
            for k in keys:
                if m.input_preprocessing_specs[k] == CategoricalEncodingEnum.ONE_HOT:
                    if values["categorical_method"] != values["descriptor_method"]:
                        print(values["categorical_method"], values["descriptor_method"])
                        raise ValueError(
                            "One-hot encoded CategoricalDescriptorInput features has to be treated with the same method as categoricals."
                        )
        return values

    @staticmethod
    def _generate_model_specs(
        domain: Domain,
        model_specs: Optional[BotorchModels] = None,
    ) -> BotorchModels:
        """Method to generate model specifications when no model specs are passed
        As default specification, a 5/2 matern kernel with automated relevance detection and normalization of the input features is used.
        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            model_specs (List[ModelSpec], optional): List of model specification classes specifying the models to be used in the strategy. Defaults to None.
        Raises:
            KeyError: if there is a model spec for an unknown output feature
            KeyError: if a model spec has an unknown input feature
        Returns:
            List[ModelSpec]: List of model specification classes
        """
        existing_keys = (
            model_specs.output_features.get_keys() if model_specs is not None else []
        )
        non_exisiting_keys = list(set(domain.outputs.get_keys()) - set(existing_keys))
        _model_specs = model_specs.models if model_specs is not None else []
        for output_feature in non_exisiting_keys:
            if len(domain.inputs.get(CategoricalInput, exact=True)):
                _model_specs.append(
                    MixedSingleTaskGPModel(
                        input_features=domain.inputs,
                        output_features=OutputFeatures(features=[domain.outputs.get_by_key(output_feature)]),  # type: ignore
                    )
                )
            else:
                _model_specs.append(
                    SingleTaskGPModel(
                        input_features=domain.inputs,
                        output_features=OutputFeatures(features=[domain.outputs.get_by_key(output_feature)]),  # type: ignore
                    )
                )
        model_specs = BotorchModels(models=_model_specs)
        model_specs._check_compability(
            input_features=domain.inputs, output_features=domain.outputs
        )
        return model_specs

    def _init_domain(self):
        """set up the transformer and the objective"""
        torch.manual_seed(self.seed)
        self.init_objective()

    @property
    def input_preprocessing_specs(self) -> TInputTransformSpecs:
        return self.model_specs.input_preprocessing_specs  # type: ignore

    @property
    def _features2idx(self) -> Dict[str, Tuple[int]]:
        features2idx, _ = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs
        )
        return features2idx

    @property
    def _features2names(self) -> Dict[str, Tuple[str]]:
        _, features2names = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs
        )
        return features2names

    def _fit(self, experiments: pd.DataFrame):
        """[summary]

        Args:
            transformed (pd.DataFrame): [description]
        """
        self.model_specs.fit(experiments)  # type: ignore
        self.model = self.model_specs.compatibilize(  # type: ignore
            input_features=self.domain.input_features,  # type: ignore
            output_features=self.domain.output_features,  # type: ignore
        )

    def _predict(self, transformed: pd.DataFrame):
        # we are using self.model here for this purpose we have to take the transformed
        # input and further transform it to a torch tensor
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        with torch.no_grad():
            preds = self.model.posterior(X=X).mean.cpu().detach().numpy()  # type: ignore
            # TODO: add a option to return the real uncertainty including the data uncertainty
            stds = np.sqrt(self.model.posterior(X=X).variance.cpu().detach().numpy())  # type: ignore
        return preds, stds

    # TODO: test this
    def calc_acquisition(
        self, candidates: pd.DataFrame, combined: bool = False
    ) -> np.ndarray:
        """Calculate the acqusition value for a set of experiments.

        Args:
            candidates (pd.DataFrame): Dataframe with experimentes for which the acqf value should be calculated.
            combined (bool, optional): If combined an acquisition value for the whole batch is calculated, else individual ones.
                Defaults to False.

        Returns:
            np.ndarray: Dataframe with the acquisition values.
        """
        transformed = self.domain.inputs.transform(
            candidates, self.input_preprocessing_specs
        )
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        if combined is False:
            X = X.unsqueeze(-2)
        return self.acqf.forward(X).cpu().detach().numpy()  # type: ignore

    # TODO: test this
    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        """Method to choose a set of candidates from a candidate pool.

        Args:
            candidate_pool (pd.DataFrame): The pool of candidates from which the candidates should be chosen.
            candidate_count (Optional[NonNegativeInt], optional): Number of candidates to choose. Defaults to None.

        Returns:
            pd.DataFrame: The chosen set of candidates.
        """

        acqf_values = self.calc_acquisition(candidate_pool)

        return candidate_pool.iloc[
            np.argpartition(acqf_values, -1 * candidate_count)[-candidate_count:]  # type: ignore
        ]

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        """[summary]

        Args:
            candidate_count (int, optional): [description]. Defaults to 1.

        Returns:
            pd.DataFrame: [description]
        """

        assert candidate_count > 0, "candidate_count has to be larger than zero."

        # optimize
        # we have to distuinguish the following scenarios
        # - no categoricals - check
        # - categoricals with one hot and free variables
        # - categoricals with one hot and exhaustive screening, could be in combination with garrido merchan - check
        # - categoricals with one hot and OEN, could be in combination with garrido merchan - OEN not implemented
        # - descriptized categoricals not yet implemented
        num_categorical_features = len(self.domain.get_features(CategoricalInput))
        num_categorical_combinations = len(
            self.domain.inputs.get_categorical_combinations()
        )
        assert self.acqf is not None

        lower, upper = self.domain.inputs.get_bounds(
            specs=self.input_preprocessing_specs
        )
        bounds = torch.tensor([lower, upper]).to(**tkwargs)

        if (
            (num_categorical_features == 0)
            or (num_categorical_combinations == 1)
            or (
                (self.categorical_method == CategoricalMethodEnum.FREE)
                and (self.descriptor_method == CategoricalMethodEnum.FREE)
            )
        ) and len(self.domain.cnstrs.get(NChooseKConstraint)) == 0:
            candidates = optimize_acqf(
                acq_function=self.acqf,
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=self.domain, constraint=LinearEqualityConstraint  # type: ignore
                ),
                inequality_constraints=get_linear_constraints(
                    domain=self.domain, constraint=LinearInequalityConstraint  # type: ignore
                ),
                fixed_features=self.get_fixed_features(),
                return_best_only=True,
            )
            # options={"seed":self.seed})

        elif (
            (self.categorical_method == CategoricalMethodEnum.EXHAUSTIVE)
            or (self.descriptor_method == CategoricalMethodEnum.EXHAUSTIVE)
        ) and len(self.domain.cnstrs.get(NChooseKConstraint)) == 0:
            # TODO: marry this withe groups of XY
            candidates = optimize_acqf_mixed(
                acq_function=self.acqf,
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=self.domain, constraint=LinearEqualityConstraint  # type: ignore
                ),
                inequality_constraints=get_linear_constraints(
                    domain=self.domain, constraint=LinearInequalityConstraint  # type: ignore
                ),
                fixed_features_list=self.get_categorical_combinations(),
            )
            # options={"seed":self.seed})

        elif len(self.domain.cnstrs.get(NChooseKConstraint)) > 0:
            candidates = optimize_acqf_mixed(
                acq_function=self.acqf,
                bounds=bounds,
                q=candidate_count,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                equality_constraints=get_linear_constraints(
                    domain=self.domain, constraint=LinearEqualityConstraint  # type: ignore
                ),
                inequality_constraints=get_linear_constraints(
                    domain=self.domain, constraint=LinearInequalityConstraint  # type: ignore
                ),
                fixed_features_list=self.get_fixed_values_list(),
            )

        else:
            raise IOError()

        # postprocess the results
        # TODO: in case of free we have to transform back the candidates first and then compute the metrics
        # otherwise the prediction holds only for the infeasible solution, this solution should then also be
        # applicable for >1d descriptors
        preds = self.model.posterior(X=candidates[0]).mean.detach().numpy()  # type: ignore
        stds = np.sqrt(self.model.posterior(X=candidates[0]).variance.detach().numpy())  # type: ignore

        input_feature_keys = [
            item
            for key in self.domain.inputs.get_keys()
            for item in self._features2names[key]
        ]

        df_candidates = pd.DataFrame(
            data=candidates[0].detach().numpy(), columns=input_feature_keys
        )

        df_candidates = self.domain.inputs.inverse_transform(
            df_candidates, self.input_preprocessing_specs
        )

        for i, feat in enumerate(self.domain.outputs.get_by_objective(excludes=None)):
            df_candidates[feat.key + "_pred"] = preds[:, i]
            df_candidates[feat.key + "_sd"] = stds[:, i]
            df_candidates[feat.key + "_des"] = feat.objective(preds[:, i])  # type: ignore

        return df_candidates

    def _tell(self) -> None:
        if self.has_sufficient_experiments():
            # todo move this up to predictive strategy
            self.fit()
            self.init_acqf()
        return

    def init_acqf(self) -> None:
        self._init_acqf()
        return

    @abstractmethod
    def _init_acqf(
        self,
    ) -> None:
        pass

    def init_objective(self) -> None:
        self._init_objective()
        return

    @abstractmethod
    def _init_objective(
        self,
    ) -> None:
        pass

    # TODO: maybe replace the one below witht this one.
    # def get_fixed_features(self):
    #     lower, upper = self.domain.inputs.get_bounds(
    #         specs=self.input_preprocessing_specs
    #     )
    #     fixed_features = {}
    #     for i in range(len(lower)):
    #         if lower[i] == upper[i]:
    #             fixed_features[i] = lower[i]
    #     return fixed_features

    def get_fixed_features(self):
        """provides the values of all fixed features

        Raises:
            NotImplementedError: [description]

        Returns:
            fixed_features (dict): Dictionary of fixed features, keys are the feature indices, values the transformed feature values
        """
        fixed_features = {}
        features2idx = self._features2idx

        for _, feat in enumerate(self.domain.get_features(InputFeature)):
            if feat.fixed_value() is not None:  # type: ignore
                fixed_values = feat.fixed_value(transform_type=self.input_preprocessing_specs.get(feat.key))  # type: ignore
                for j, idx in enumerate(features2idx[feat.key]):
                    fixed_features[idx] = fixed_values[j]  # type: ignore

        # in case the optimization method is free and not allowed categories are present
        # one has to fix also them, this is abit of double work as it should be also reflected
        # in the bounds but helps to make it safer
        # TODO: this has to be done also for the descriptors
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

    def get_categorical_combinations(self):
        """provides all possible combinations of fixed values

        Returns:
            list_of_fixed_features List[dict]: Each dict contains a combination of fixed values
        """
        fixed_basis = self.get_fixed_features()
        include = CategoricalInput
        exclude = None

        if (self.descriptor_method == CategoricalMethodEnum.FREE) and (
            self.categorical_method == CategoricalMethodEnum.FREE
        ):
            return [{}]
        elif self.descriptor_method == CategoricalMethodEnum.FREE:
            exclude = CategoricalDescriptorInput
        elif self.categorical_method == CategoricalMethodEnum.FREE:
            include = CategoricalDescriptorInput

        combos = self.domain.inputs.get_categorical_combinations(
            include=include, exclude=exclude
        )
        # now build up the fixed feature list
        if len(combos) == 1:
            return [fixed_basis]
        else:
            features2idx = self._features2idx
            list_of_fixed_features = []

            for combo in combos:
                fixed_features = copy.deepcopy(fixed_basis)

                for pair in combo:
                    feat, val = pair
                    feature = self.domain.get_feature(feat)
                    if (
                        isinstance(feature, CategoricalDescriptorInput)
                        and self.input_preprocessing_specs[feat]
                        == CategoricalEncodingEnum.DESCRIPTOR
                    ):
                        index = feature.categories.index(val)

                        for j, idx in enumerate(features2idx[feat]):
                            fixed_features[idx] = feature.values[index][j]

                    elif isinstance(feature, CategoricalInput):
                        # it has to be onehot in this case
                        transformed = feature.to_onehot_encoding(pd.Series([val]))
                        for j, idx in enumerate(features2idx[feat]):
                            fixed_features[idx] = transformed.values[0, j]

                list_of_fixed_features.append(fixed_features)
        return list_of_fixed_features

    def get_nchoosek_combinations(self):

        """
        generate a list of fixed values dictionaries from n-choose-k constraints
        """

        # generate botorch-friendly fixed values
        features2idx = self._features2idx
        used_features, unused_features = self.domain.get_nchoosek_combinations()
        fixed_values_list_cc = []
        for used, unused in zip(used_features, unused_features):
            fixed_values = {}

            # sets unused features to zero
            for f_key in unused:
                fixed_values[features2idx[f_key][0]] = 0.0

            fixed_values_list_cc.append(fixed_values)

        if len(fixed_values_list_cc) == 0:
            fixed_values_list_cc.append({})  # any better alternative here?

        return fixed_values_list_cc

    def get_fixed_values_list(self):

        # CARTESIAN PRODUCTS: fixed values from categorical combinations X fixed values from nchoosek constraints
        fixed_values_full = []

        for ff1 in self.get_categorical_combinations():
            for ff2 in self.get_nchoosek_combinations():
                ff = ff1.copy()
                ff.update(ff2)
                fixed_values_full.append(ff)

        return fixed_values_full

    def has_sufficient_experiments(
        self,
    ) -> bool:
        if self.experiments is None:
            return False
        degrees_of_freedom = len(self.domain.get_features(InputFeature)) - len(
            self.get_fixed_features()
        )
        # degrees_of_freedom = len(self.domain.get_features(InputFeature)) + 1
        if self.experiments.shape[0] > degrees_of_freedom + 1:
            return True
        return False

    def get_acqf_input_tensors(self):

        experiments = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments
        )

        # TODO: should this be selectable?
        clean_experiments = experiments.drop_duplicates(
            subset=[var.key for var in self.domain.get_features(InputFeature)],
            keep="first",
            inplace=False,
        )

        transformed = self.domain.inputs.transform(
            clean_experiments, self.input_preprocessing_specs
        )
        X_train = torch.from_numpy(transformed.values).to(**tkwargs)

        if self.domain.candidates is not None:
            transformed_candidates = self.domain.inputs.transform(
                self.domain.candidates, self.input_preprocessing_specs
            )
            X_pending = torch.from_numpy(transformed_candidates.values).to(**tkwargs)
        else:
            X_pending = None

        return X_train, X_pending
