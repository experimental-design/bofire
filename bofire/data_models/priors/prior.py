from abc import abstractmethod

import gpytorch

from bofire.data_models.base import BaseModel


class Prior(BaseModel):
    """Abstract Prior class."""

    type: str

    @abstractmethod
    def to_gpytorch(self) -> gpytorch.priors.Prior:
        """Transform prior to a gpytorch prior.

        Returns:
            gpytorch.priors.Prior: Equivalent gpytorch prior object.
        """
        pass
