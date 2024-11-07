from typing import Dict, Optional

import pandas as pd

from bofire.data_models.base import BaseModel
from bofire.strategies.data_models.values import InputValue, OutputValue


class Candidate(BaseModel):
    """A Bofire candidate.

    Attributes:
        inputValues (Dict[str, InputValue]): Dictionary of input values where the key is the
            corresponding input feature key.
        outputValues (Dict[str, OutputValue], optional): Dictionary of output values where
            the key is the corresponding output feature key.

    """

    inputValues: Dict[str, InputValue]
    outputValues: Optional[Dict[str, OutputValue]] = None

    def to_series(self) -> pd.Series:
        """Transform to pandas series.

        Returns:
            pd.Series: pandas series which corresponds to one row in the original candidates dataframe

        """
        data = []
        index = []
        for key, value in self.inputValues.items():
            data.append(value.value)
            index.append(key)
        if self.outputValues is not None:
            for key, value in self.outputValues.items():
                data += [value.predictedValue, value.standardDeviation, value.objective]
                index += [f"{key}_{p}" for p in ["pred", "sd", "des"]]
        return pd.Series(data=data, index=index)
