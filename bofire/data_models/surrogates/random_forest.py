from typing import Annotated, Literal, Optional, Type, Union

from pydantic import Field

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class RandomForestSurrogate(TrainableBotorchSurrogate):
    """Random forest regressor: an average over many decision trees, each fitted to a
    random resample of the data.

    The spread of the individual trees' predictions serves as the uncertainty. The
    model makes no smoothness assumption, so it copes with abrupt changes in the
    response, but it can't extrapolate beyond the data and its uncertainty is less
    calibrated than a Gaussian process's. The hyperparameters are those of
    scikit-learn's `RandomForestRegressor`, with its defaults.

    Examples:
        >>> RandomForestSurrogate(inputs=inputs, outputs=outputs, n_estimators=200)
    """

    type: Literal["RandomForestSurrogate"] = "RandomForestSurrogate"

    n_estimators: int = Field(
        default=100,
        description="Number of trees. More trees give smoother predictions and a "
        "more stable uncertainty at a higher cost.",
    )
    criterion: Literal[
        "squared_error",
        "absolute_error",
        "friedman_mse",
        "poisson",
    ] = Field(
        default="squared_error",
        description="How the quality of a split is measured: squared error, absolute "
        "error (less sensitive to outliers), Friedman's improvement of the squared "
        "error, or Poisson deviance (for non-negative counts).",
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximal depth of a tree. If not provided, trees grow until their "
        "leaves are pure or reach `min_samples_split`.",
    )
    min_samples_split: Union[int, float] = Field(
        default=2,
        description="Minimal number of experiments a node needs to be split; a float "
        "is a fraction of all experiments.",
    )
    min_samples_leaf: Union[int, float] = Field(
        default=1,
        description="Minimal number of experiments in a leaf; a float is a fraction of "
        "all experiments. Larger values smooth the predictions.",
    )
    min_weight_fraction_leaf: float = Field(
        default=0.0,
        description="Minimal fraction of all experiments in a leaf.",
    )
    max_features: Union[int, float, Literal["sqrt", "log2"]] = Field(
        default=1.0,
        description="Number of inputs considered at each split: an integer count, a "
        "fraction of all inputs, or the square root or base-2 logarithm of their "
        "number. Fewer inputs make the trees more diverse.",
    )
    max_leaf_nodes: Optional[int] = Field(
        default=None,
        description="Maximal number of leaves per tree. If not provided, unlimited.",
    )
    min_impurity_decrease: float = Field(
        default=0.0,
        description="A node is only split if this reduces the split criterion by at "
        "least this amount, weighted by the share of experiments in the node.",
    )
    bootstrap: bool = Field(
        default=True,
        description="Whether each tree is fitted to a random resample of the "
        "experiments, drawn with replacement. If not, every tree sees all of them.",
    )
    random_state: Optional[int] = Field(
        default=None,
        description="Seed for the random resampling and the choice of inputs at each "
        "split, for reproducible fits. If not provided, every fit differs.",
    )
    ccp_alpha: Annotated[float, Field(ge=0)] = Field(
        default=0.0,
        description="Strength of the cost-complexity pruning of the trees after "
        "growing; 0 means no pruning.",
    )
    max_samples: Optional[Union[int, float]] = Field(
        default=None,
        description="Size of each tree's resample when `bootstrap` is set: a count, or "
        "a fraction of the experiments. If not provided, as many as there are "
        "experiments.",
    )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return issubclass(my_type, ContinuousOutput)
