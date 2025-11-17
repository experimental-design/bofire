from typing import Literal, Type

from pydantic import PositiveInt

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class TestSurrogate:
    pass


class AdditiveMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Additive MAP SAAS single-task GP

    Maximum-a-posteriori (MAP) version of the sparse axis-aligned subspace
    `FullyBayesianSingleTaskGPSurrogate` with `model_type` equals to "saas".

    Attributes:
        n_taus (PositiveInt): Number of sub-kernels to use in the SAAS model.
    """

    type: Literal["AdditiveMapSaasSingleTaskGPSurrogate"] = (
        "AdditiveMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = 4

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

class EnsembleMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Instantiates an ``EnsembleMapSaasSingleTaskGP``, which is a batched
    ensemble of ``SingleTaskGP``s with the Matern-5/2 kernel and a SAAS prior.
    The model is intended to be trained with ``ExactMarginalLogLikelihood`` and
    ``fit_gpytorch_mll``. Under the hood, the model is equivalent to a
    multi-output ``BatchedMultiOutputGPyTorchModel``, but it produces a
    ``MixtureGaussiaPosterior``, which leads to ensembling of the model outputs.

    Args:
        train_X: An `n x d` tensor of training features.
        train_Y: An `n x 1` tensor of training observations.
        train_Yvar: An optional `n x 1` tensor of observed measurement noise.
        num_taus: The number of taus to use (4 if omitted). Each tau is
            a sparsity parameter for the corresponding kernel in the ensemble.
        taus: An optional tensor of shape `num_taus` containing the taus to use.
            If omitted, the taus are sampled from a HalfCauchy(0.1) distribution.
        outcome_transform: An outcome transform that is applied to the
            training data during instantiation and to the posterior during
            inference (that is, the `Posterior` obtained by calling
            `.posterior` on the model will be on the original scale). We use a
            `Standardize` transform if no `outcome_transform` is specified.
            Pass down `None` to use no outcome transform. Note that `.train()` will
            be called on the outcome transform during instantiation of the model.
        input_transform: An input transform that is applied in the model's
            forward pass.
    """

    type: Literal["EnsembleMapSaasSingleTaskGPSurrogate"] = (
        "EnsembleMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = 4

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
