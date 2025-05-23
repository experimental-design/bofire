import base64
import warnings
from typing import Callable, List, Tuple, Union, cast

from pydantic import PositiveInt
from typing_extensions import Self

from bofire.data_models.api import Domain
from bofire.data_models.outlier_detection.outlier_detections import OutlierDetections
from bofire.data_models.strategies.predictives.acqf_optimization import AnyAcqfOptimizer
from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.strategies.strategy import make_strategy


try:
    import cloudpickle
except ModuleNotFoundError:
    warnings.warn(
        "Cloudpickle is not available. CustomSoboStrategy's `f` cannot be dumped or loaded.",
    )

import torch
from botorch.acquisition import get_acquisition_function
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    IdentityMCObjective,
)
from botorch.models.gpytorch import GPyTorchModel

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
    qLogNEI,
    qLogPF,
    qNEI,
    qPI,
    qSR,
    qUCB,
)
from bofire.data_models.objectives.api import ConstrainedObjective, Objective
from bofire.data_models.strategies.api import AdditiveSoboStrategy as AdditiveDataModel
from bofire.data_models.strategies.api import CustomSoboStrategy as CustomDataModel
from bofire.data_models.strategies.api import (
    MultiplicativeAdditiveSoboStrategy as MultiplicativeAdditiveDataModel,
)
from bofire.data_models.strategies.api import (
    MultiplicativeSoboStrategy as MultiplicativeDataModel,
)
from bofire.data_models.strategies.predictives.sobo import (
    SoboBaseStrategy as SoboBaseDataModel,
)
from bofire.data_models.strategies.predictives.sobo import SoboStrategy as SoboDataModel
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.torch_tools import (
    get_additive_botorch_objective,
    get_custom_botorch_objective,
    get_multiplicative_additive_objective,
    get_multiplicative_botorch_objective,
    get_objective_callable,
    get_output_constraints,
    tkwargs,
)


class SoboStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: SoboBaseDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.acquisition_function = data_model.acquisition_function

    def _get_acqfs(self, n) -> List[AcquisitionFunction]:
        assert self.is_fitted is True, "Model not trained."

        X_train, X_pending = self.get_acqf_input_tensors()

        (
            objective_callable,
            constraint_callables,
            etas,
        ) = self._get_objective_and_constraints()

        assert self.model is not None

        acqf = get_acquisition_function(
            self.acquisition_function.__class__.__name__,
            self.model,
            objective_callable,
            X_observed=X_train,
            X_pending=X_pending,
            constraints=constraint_callables,
            mc_samples=self.acquisition_function.n_mc_samples,
            beta=(
                self.acquisition_function.beta
                if isinstance(self.acquisition_function, qUCB)
                else 0.2
            ),
            tau=(
                self.acquisition_function.tau
                if isinstance(self.acquisition_function, qPI)
                else 1e-3
            ),
            eta=torch.tensor(etas).to(**tkwargs),
            cache_root=True if isinstance(self.model, GPyTorchModel) else False,
            prune_baseline=(
                self.acquisition_function.prune_baseline
                if isinstance(self.acquisition_function, (qNEI, qLogNEI))
                else True
            ),
        )
        return [acqf]

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective, IdentityMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        assert self.experiments is not None, "No experiments available."
        try:
            target_feature = self.domain.outputs.get_by_objective(
                excludes=ConstrainedObjective,
            )[0]
        except IndexError:
            target_feature = self.domain.outputs.get_by_objective(includes=Objective)[0]
        target_index = self.domain.outputs.get_keys().index(target_feature.key)
        x_adapt = torch.from_numpy(
            self.domain.outputs.preprocess_experiments_one_valid_output(
                target_feature.key,
                self.experiments,
            )[target_feature.key].values,
        ).to(**tkwargs)
        objective_callable = get_objective_callable(
            idx=target_index,
            objective=target_feature.objective,
            x_adapt=x_adapt,
        )

        # get the constraints
        if (len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0) and (
            len(self.domain.outputs.get_by_objective(Objective)) > 1
        ):
            constraint_callables, etas = get_output_constraints(
                outputs=self.domain.outputs,
                experiments=self.experiments,
            )
        else:
            constraint_callables, etas = None, 1e-3

        # special cases of qUCB and qSR do not work with separate constraints
        if (isinstance(self.acquisition_function, (qSR, qUCB))) and (
            constraint_callables is not None
        ):
            return (
                ConstrainedMCObjective(
                    objective=objective_callable,
                    constraints=constraint_callables,
                    eta=torch.tensor(etas).to(**tkwargs),
                    infeasible_cost=self.get_infeasible_cost(
                        objective=objective_callable,
                    ),
                ),
                None,
                1e-3,
            )

        # return regular objective
        return (
            GenericMCObjective(objective=objective_callable)
            if not isinstance(self.acquisition_function, qLogPF)
            else IdentityMCObjective(),
            constraint_callables,
            etas,
        )

    @classmethod
    def make(
        cls,
        domain: Domain,
        acquisition_function: AnySingleObjectiveAcquisitionFunction
        | qLogPF
        | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
    ) -> Self:
        """
        Creates a single objective Bayesian optimization strategy.
        Args:
            domain: The optimization domain of the strategy.
            acquisition_function: The acquisition function to use.
            acquisition_optimizer: The optimizer to use for the acquisition function.
            surrogate_specs: The specifications for the surrogate model.
            outlier_detection_specs: The specifications for the outlier detection.
            min_experiments_before_outlier_check: The minimum number of experiments before checking for outliers.
            frequency_check: The frequency of checking for outliers.
            frequency_hyperopt: The frequency of hyperparameter optimization.
            folds: The number of folds for cross-validation.
            seed: The random seed to use.
        """
        return cast(Self, make_strategy(cls, SoboDataModel, locals()))


class AdditiveSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: AdditiveDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.use_output_constraints = data_model.use_output_constraints

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        assert self.experiments is not None, "No experiments available."
        # get the constraints
        if (
            (len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0)
            and (len(self.domain.outputs.get_by_objective(Objective)) > 1)
            and self.use_output_constraints
        ):
            constraint_callables, etas = get_output_constraints(
                outputs=self.domain.outputs,
                experiments=self.experiments,
            )
        else:
            constraint_callables, etas = None, 1e-3
        # TODO: test this
        if self.use_output_constraints:
            objective_callable = get_additive_botorch_objective(
                outputs=self.domain.outputs,
                exclude_constraints=True,
                experiments=self.experiments,
            )

            # special cases of qUCB and qSR do not work with separate constraints
            if isinstance(self.acquisition_function, (qSR, qUCB)):
                return (
                    ConstrainedMCObjective(
                        objective=objective_callable,  # type: ignore
                        constraints=constraint_callables,  # type: ignore
                        eta=torch.tensor(etas).to(**tkwargs),
                        infeasible_cost=self.get_infeasible_cost(
                            objective=objective_callable,
                        ),
                    ),
                    None,
                    1e-3,
                )
            return (
                GenericMCObjective(objective=objective_callable),  # type: ignore
                constraint_callables,
                etas,
            )

        # we absorb all constraints into the objective
        return (
            GenericMCObjective(
                objective=get_additive_botorch_objective(
                    outputs=self.domain.outputs,
                    exclude_constraints=False,  # type: ignore
                ),
            ),
            constraint_callables,
            etas,
        )

    @classmethod
    def make(  # type: ignore
        cls,
        domain: Domain,
        use_output_constraints: bool | None = None,
        acquisition_function: AnySingleObjectiveAcquisitionFunction | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
    ):
        """
        Creates a Bayesian optimization strategy that adds multiple objectives.
        The weights of the objectives are defines in the outputs of the domain.
        Args:
            domain: The optimization domain of the strategy.
            use_output_constraints: Whether to use output constraints.
            acquisition_function: The acquisition function to use.
            acquisition_optimizer: The optimizer to use for the acquisition function.
            surrogate_specs: The specifications for the surrogate model.
            outlier_detection_specs: The specifications for the outlier detection.
            min_experiments_before_outlier_check: The minimum number of experiments before checking for outliers.
            frequency_check: The frequency of checking for outliers.
            frequency_hyperopt: The frequency of hyperparameter optimization.
            folds: The number of folds for cross-validation for hyperparameter optimization.
            seed: The random seed to use.
        """
        return make_strategy(cls, AdditiveDataModel, locals())


class MultiplicativeSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: MultiplicativeDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        GenericMCObjective,
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        # we absorb all constraints into the objective
        assert self.experiments is not None, "No experiments available."
        return (
            GenericMCObjective(
                objective=get_multiplicative_botorch_objective(  # type: ignore
                    outputs=self.domain.outputs,
                    experiments=self.experiments,
                    adapt_weights_to_1_inf=True,
                ),
            ),
            None,
            1e-3,
        )

    @classmethod
    def make(
        cls,
        domain: Domain,
        acquisition_function: AnySingleObjectiveAcquisitionFunction | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
    ) -> Self:
        """
        Creates Bayesian optimization strategy that multiplies multiple objectives. The weights of
        the objectives are defines in the outputs of the domain.
        Args:
            domain: The optimization domain of the strategy.
            acquisition_function: The acquisition function to use.
            acquisition_optimizer: The optimizer to use for the acquisition function.
            surrogate_specs: The specifications for the surrogate model.
            outlier_detection_specs: The specifications for the outlier detection.
            min_experiments_before_outlier_check: The minimum number of experiments before checking for outliers.
            frequency_check: The frequency of checking for outliers.
            frequency_hyperopt: The frequency of hyperparameter optimization.
            folds: The number of folds for cross-validation for hyperparameter optimization.
            seed: The random seed to use.
        """
        return cast(Self, make_strategy(cls, MultiplicativeDataModel, locals()))


class MultiplicativeAdditiveSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: MultiplicativeAdditiveDataModel,
        **kwargs,
    ):
        self.additive_features = data_model.additive_features
        super().__init__(data_model=data_model, **kwargs)

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        GenericMCObjective,
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        # we absorb all constraints into the objective
        assert self.experiments is not None, "No experiments available."
        return (
            GenericMCObjective(
                objective=get_multiplicative_additive_objective(  # type: ignore
                    outputs=self.domain.outputs,
                    experiments=self.experiments,
                    additive_features=self.additive_features,
                    adapt_weights_to_1_inf=True,
                )
            ),
            None,
            1e-3,
        )

    @classmethod
    def make(  # type: ignore
        cls,
        domain: Domain,
        use_output_constraints: bool | None = None,
        additive_features: List[str] | None = None,
        acquisition_function: AnySingleObjectiveAcquisitionFunction | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
    ) -> Self:
        """
        Creates a Bayesian optimization strategy that mixes additions and multiplions of multiple objectives.
        The weights of the objectives are defines in the outputs of the domain.
        By default, all objectives are multiplicative. Additive features
        (inputs or outputs) can be specified in the `additive_features` list.
        Args:
            domain: The optimization domain of the strategy.
            use_output_constraints: Whether to use output constraints.
            additive_features: The features to use for the addition.
            acquisition_function: The acquisition function to use.
            acquisition_optimizer: The optimizer to use for the acquisition function.
            surrogate_specs: The specifications for the surrogate model.
            outlier_detection_specs: The specifications for the outlier detection.
            min_experiments_before_outlier_check: The minimum number of experiments before checking for outliers.
            frequency_check: The frequency of checking for outliers.
            frequency_hyperopt: The frequency of hyperparameter optimization.
            folds: The number of folds for cross-validation for hyperparameter optimization.
            seed: The random seed to use.
        """
        return cast(Self, make_strategy(cls, MultiplicativeAdditiveDataModel, locals()))


class CustomSoboStrategy(SoboStrategy):
    def __init__(
        self,
        data_model: CustomDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.use_output_constraints = data_model.use_output_constraints
        if data_model.dump is not None:
            self.loads(data_model.dump)
        else:
            self.f = None

    def _get_objective_and_constraints(
        self,
    ) -> Tuple[
        Union[GenericMCObjective, ConstrainedMCObjective],
        Union[List[Callable[[torch.Tensor], torch.Tensor]], None],
        Union[List, float],
    ]:
        assert self.experiments is not None, "No experiments available."
        if self.f is None:
            raise ValueError("No function has been provided for the strategy")
        # get the constraints
        if (
            (len(self.domain.outputs.get_by_objective(ConstrainedObjective)) > 0)
            and (len(self.domain.outputs.get_by_objective(Objective)) > 1)
            and self.use_output_constraints
        ):
            constraint_callables, etas = get_output_constraints(
                outputs=self.domain.outputs,
                experiments=self.experiments,
            )
        else:
            constraint_callables, etas = None, 1e-3

        if self.use_output_constraints:
            objective_callable = get_custom_botorch_objective(
                outputs=self.domain.outputs,
                f=self.f,
                exclude_constraints=True,
                experiments=self.experiments,
            )
            # special cases of qUCB and qSR do not work with separate constraints
            if isinstance(self.acquisition_function, (qSR, qUCB)):
                return (
                    ConstrainedMCObjective(
                        objective=objective_callable,  # type: ignore
                        constraints=constraint_callables,  # type: ignore
                        eta=torch.tensor(etas).to(**tkwargs),
                        infeasible_cost=self.get_infeasible_cost(
                            objective=objective_callable,
                        ),
                    ),
                    None,
                    1e-3,
                )
            return (
                GenericMCObjective(objective=objective_callable),  # type: ignore
                constraint_callables,
                etas,
            )

        # we absorb all constraints into the objective
        return (
            GenericMCObjective(
                objective=get_custom_botorch_objective(  # type: ignore
                    outputs=self.domain.outputs,
                    f=self.f,
                    exclude_constraints=False,
                    experiments=self.experiments,
                ),
            ),
            constraint_callables,
            etas,
        )

    def dumps(self) -> str:
        """Dumps the function to a string via pickle as this is not directly json serializable."""
        if self.f is None:
            raise ValueError("No function has been provided for the strategy")
        f_bytes_dump = cloudpickle.dumps(self.f)  # type: ignore
        return base64.b64encode(f_bytes_dump).decode()

    def loads(self, data: str):
        """Loads the function from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        f_bytes_load = base64.b64decode(data.encode())
        self.f = cloudpickle.loads(f_bytes_load)  # type: ignore

    @classmethod
    def make(  # type: ignore
        cls,
        domain: Domain,
        use_output_constraints: bool | None = None,
        dump: str | None = None,
        acquisition_function: AnySingleObjectiveAcquisitionFunction | None = None,
        acquisition_optimizer: AnyAcqfOptimizer | None = None,
        surrogate_specs: BotorchSurrogates | None = None,
        outlier_detection_specs: OutlierDetections | None = None,
        min_experiments_before_outlier_check: PositiveInt | None = None,
        frequency_check: PositiveInt | None = None,
        frequency_hyperopt: int | None = None,
        folds: int | None = None,
        seed: int | None = None,
    ):
        """
        The `CustomSoboStrategy` can be used to design custom objectives or objective combinations for optimizations.
        In this tutorial notebook, it is shown how to use it to optimize a quantity that depends on a combination of
        an inferred quantity and one of the inputs. See tutorials/advanced_examples/custom_sobo.ipynb.

        Args:
            domain: The optimization domain of the strategy.
            use_output_constraints: Whether to use output constraints.
            dump: The function to use for the optimization.
            acquisition_function: The acquisition function to use.
            acquisition_optimizer: The optimizer to use for the acquisition function.
            surrogate_specs: The specifications for the surrogate model.
            outlier_detection_specs: The specifications for the outlier detection.
            min_experiments_before_outlier_check: The minimum number of experiments before checking for outliers.
            frequency_check: The frequency of checking for outliers.
            frequency_hyperopt: The frequency of hyperparameter optimization.
            folds: The number of folds for cross-validation.
            seed: The random seed to use.
        """
        return make_strategy(cls, CustomDataModel, locals())
