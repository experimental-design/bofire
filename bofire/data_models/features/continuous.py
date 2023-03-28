from typing import ClassVar, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field, root_validator

from bofire.data_models.features.feature import Output
from bofire.data_models.features.numerical import NumericalInput
from bofire.data_models.objectives.api import AnyObjective, MaximizeObjective


class ContinuousInput(NumericalInput):
    """Base class for all continuous input features.

    Attributes:
        bounds (Tuple[float, float]): A tuple that stores the lower and upper bound of the feature.
    """

    type: Literal["ContinuousInput"] = "ContinuousInput"
    order_id: ClassVar[int] = 1

    bounds: Tuple[float, float]

    @property
    def lower_bound(self) -> float:
        return self.bounds[0]

    @property
    def upper_bound(self) -> float:
        return self.bounds[1]

    @root_validator(pre=False, skip_on_failure=True)
    def validate_lower_upper(cls, values):
        """Validates that the lower bound is lower than the upper bound

        Args:
            values (Dict): Dictionary with attributes key, lower and upper bound

        Raises:
            ValueError: when the lower bound is higher than the upper bound

        Returns:
            Dict: The attributes as dictionary
        """
        if values["bounds"][0] > values["bounds"][1]:
            raise ValueError(
                f'lower bound must be <= upper bound, got {values["lower_bound"]} > {values["upper_bound"]}'
            )
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Raises:
            ValueError: when non numerical values are passed
            ValueError: when values are larger than the upper bound of the feature
            ValueError: when values are lower than the lower bound of the feature

        Returns:
            pd.Series: The passed dataFrame with candidates
        """

        noise = 10e-6
        super().validate_candidental(values)
        if (values < self.lower_bound - noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}`are larger than lower bound `{self.lower_bound}` "
            )
        if (values > self.upper_bound + noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}`are smaller than upper bound `{self.upper_bound}` "
            )
        return values

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(
            name=self.key,
            data=np.random.uniform(self.lower_bound, self.upper_bound, n),
        )

    def __str__(self) -> str:
        """Method to return a string of lower and upper bound

        Returns:
            str: String of a list with lower and upper bound
        """
        return f"[{self.lower_bound},{self.upper_bound}]"


class ContinuousOutput(Output):
    """The base class for a continuous output feature

    Attributes:
        objective (objective, optional): objective of the feature indicating in which direction it should be optimzed. Defaults to `MaximizeObjective`.
    """

    type: Literal["ContinuousOutput"] = "ContinuousOutput"
    order_id: ClassVar[int] = 6
    unit: Optional[str] = None

    objective: Optional[AnyObjective] = Field(
        default_factory=lambda: MaximizeObjective(w=1.0)
    )

    def plot(
        self,
        lower: float,
        upper: float,
        experiments: Optional[pd.DataFrame] = None,
        plot_details: bool = True,
        line_options: Optional[Dict] = None,
        scatter_options: Optional[Dict] = None,
        label_options: Optional[Dict] = None,
        title_options: Optional[Dict] = None,
    ):
        """Plot the assigned objective.

        Args:
            lower (float): lower bound for the plot
            upper (float): upper bound for the plot
            experiments (Optional[pd.DataFrame], optional): If provided, scatter also the historical data in the plot. Defaults to None.
        """
        if self.objective is None:
            raise ValueError(
                f"No objective assigned for ContinuousOutputFeauture with key {self.key}."
            )

        line_options = line_options or {}
        scatter_options = scatter_options or {}
        label_options = label_options or {}
        title_options = title_options or {}

        line_options["color"] = line_options.get("color", "black")
        scatter_options["color"] = scatter_options.get("color", "red")

        x = pd.Series(np.linspace(lower, upper, 5000))
        reward = self.objective.__call__(x)
        fig, ax = plt.subplots()
        ax.plot(x, reward, **line_options)
        # TODO: validate dataframe
        if experiments is not None:
            x_data = experiments.loc[experiments[self.key].notna(), self.key].values
            ax.scatter(
                x_data,  # type: ignore
                self.objective.__call__(x_data),  # type: ignore
                **scatter_options,
            )
        ax.set_title("Objective %s" % self.key, **title_options)
        ax.set_ylabel("Objective", **label_options)
        ax.set_xlabel(self.key, **label_options)
        if plot_details:
            ax = self.objective.plot_details(ax=ax)
        return fig, ax

    def __str__(self) -> str:
        return "ContinuousOutputFeature"
