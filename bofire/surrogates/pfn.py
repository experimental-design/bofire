import base64
import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import PFNSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.utils.torch_tools import tkwargs


# Import PFN models from botorch_community
try:
    from botorch_community.models.prior_fitted_network import (
        MultivariatePFNModel,
        PFNModel,
    )
    from botorch_community.models.utils.prior_fitted_network import ModelPaths
except ImportError as e:
    raise ImportError("Issue in importing PFN models from botorch_community.") from e


class PFNSurrogate(TrainableBotorchSurrogate):
    """Prior-data Fitted Network (PFN) surrogate for Bayesian optimization.

    PFN is a pre-trained transformer-based model that can make predictions
    in a zero-shot manner by conditioning on training data. Unlike traditional
    surrogates, PFN doesn't require gradient-based training on the specific task.

    The model is loaded from a checkpoint and makes predictions by processing
    the training data as context along with test points.

    Attributes:
        checkpoint_url: URL or path to the pre-trained PFN checkpoint.
        batch_first: Whether batch dimension comes first in tensors.
        multivariate: If True, uses MultivariatePFNModel for joint posteriors.
        constant_model_kwargs: Additional kwargs passed to model during inference.
        load_training_checkpoint: Whether to load a training checkpoint format.
        cache_dir: Directory for caching downloaded models.
        model: The underlying PFN model (PFNModel or MultivariatePFNModel).
    """

    def __init__(self, data_model: DataModel, **kwargs):
        """Initialize the PFN surrogate.

        Args:
            data_model: The PFNSurrogate data model with configuration.
            **kwargs: Additional arguments passed to parent class.
        """
        self.checkpoint_url = data_model.checkpoint_url
        self.batch_first = data_model.batch_first
        self.multivariate = data_model.multivariate
        self.constant_model_kwargs = data_model.constant_model_kwargs
        self.load_training_checkpoint = data_model.load_training_checkpoint
        self.cache_dir = data_model.cache_dir
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        # self.num_samples = data_model.num_samples
        super().__init__(data_model, **kwargs)

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    model: Optional[PFNModel] = None

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ) -> None:
        """Fit the PFN model to training data.

        For PFN, "fitting" means loading the pre-trained model and storing
        the training data as context. The model itself is not trained.

        Args:
            tX: Training features of shape (n, d).
            tY: Training targets of shape (n, 1).
            input_transform: Optional input transform to apply.
            outcome_transform: Optional outcome transform to apply.
            **kwargs: Additional keyword arguments (unused).
        """
        # Convert checkpoint_url to ModelPaths enum if it's a known path
        checkpoint_path = self.checkpoint_url
        if checkpoint_path == "pfns4bo_hebo":
            checkpoint_path = ModelPaths.pfns4bo_hebo
        elif checkpoint_path == "pfns4bo_bnn":
            checkpoint_path = ModelPaths.pfns4bo_bnn

        # Apply outcome transform if provided, but NOT input transform
        # Input transform will be handled by the PFN model itself
        if outcome_transform is not None:
            tY = outcome_transform(tY)[0]

        # Ensure data is on the correct device and has correct dtype
        tX = tX.to(**tkwargs)
        tY = tY.to(**tkwargs)

        # Ensure tY is 2-dimensional (n, 1)
        if tY.dim() == 1:
            tY = tY.unsqueeze(-1)

        # Select model class based on multivariate flag
        model_class = MultivariatePFNModel if self.multivariate else PFNModel

        # Initialize the PFN model
        # Note: We pass model=None to download from checkpoint_url
        # The model will automatically download and cache the checkpoint
        # We pass input_transform to the model so it can handle transformations
        # internally during both initialization and posterior calls
        self.model = model_class(
            train_X=tX,
            train_Y=tY,
            model=None,  # Will be downloaded from checkpoint_url
            checkpoint_url=checkpoint_path,
            train_Yvar=None,  # PFN doesn't use noise variance
            batch_first=self.batch_first,
            constant_model_kwargs=self.constant_model_kwargs,
            input_transform=input_transform,  # Let PFN handle the transform
            load_training_checkpoint=self.load_training_checkpoint,
        )

        # Store outcome transform if provided
        # PFN doesn't directly support outcome transforms, but we can apply them
        # in the prediction step if needed
        if outcome_transform is not None:
            self._outcome_transform = outcome_transform

    def _predict(self, transformed_X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the PFN model.

        The PFN model handles input transformation internally via its
        input_transform attribute, so we just convert to tensor and call posterior.
        Predictions are made by drawing samples from the posterior distribution,
        applying outcome transformation if configured, and computing statistics.

        Args:
            transformed_X: Input features as a pandas DataFrame.

        Returns:
            Tuple of predictions and standard deviations as numpy arrays.
        """
        # Convert to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)

        # The model will apply input_transform internally in posterior() if needed
        with torch.no_grad():
            posterior = self.model.posterior(X=X, observation_noise=True)

            # Get mean and variance from the posterior
            preds, var = (
                self._outcome_transform.untransform(
                    posterior.mean, posterior.variance, None
                )
                if hasattr(self, "_outcome_transform")
                and self._outcome_transform is not None
                else (posterior.mean, posterior.variance)
            )
            stds = np.sqrt(var.cpu().detach().numpy())
            preds = preds.cpu().detach().numpy()
        return preds, stds

    # Override the _dumps and _loads methods to handle the fact that PFN models are not
    # standard PyTorch models and may have additional state (outcome transform) that needs
    # to be saved and loaded.
    # TODO: PFN implementation in botorch_community uses posterior_transform.
    # However, this is not currently implemented in the botorch impementation.
    # Add this functionality once the botorch implementation is updated to match the
    # botorch_community implementation.
    def _dumps(self) -> str:
        """Dumps the model and outcome transform to a string.

        Overrides the parent method to also save the outcome transform,
        which is stored separately from the model.
        """
        # Serialize the model
        self.model.prediction_strategy = None
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        model_bytes = base64.b64encode(buffer.getvalue()).decode()

        # Serialize the outcome transform if it exists
        outcome_transform_bytes = None
        if hasattr(self, "_outcome_transform") and self._outcome_transform is not None:
            buffer = io.BytesIO()
            torch.save(self._outcome_transform, buffer)
            outcome_transform_bytes = base64.b64encode(buffer.getvalue()).decode()

        # Combine both into a single string with a delimiter
        if outcome_transform_bytes:
            return f"{model_bytes}|||OUTCOME_TRANSFORM|||{outcome_transform_bytes}"
        return model_bytes

    def loads(self, data: str) -> None:
        """Loads the model and outcome transform from a string.

        Overrides the parent method to also restore the outcome transform.
        """
        # Check if outcome transform is included
        if "|||OUTCOME_TRANSFORM|||" in data:
            model_bytes, outcome_transform_bytes = data.split("|||OUTCOME_TRANSFORM|||")

            # Load the model
            buffer = io.BytesIO(base64.b64decode(model_bytes.encode()))
            self.model = torch.load(buffer, weights_only=False)

            # Load the outcome transform
            buffer = io.BytesIO(base64.b64decode(outcome_transform_bytes.encode()))
            self._outcome_transform = torch.load(buffer, weights_only=False)
        else:
            # No outcome transform, just load the model
            buffer = io.BytesIO(base64.b64decode(data.encode()))
            self.model = torch.load(buffer, weights_only=False)
