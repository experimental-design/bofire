import numpy as np
import pandas as pd

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.domain.api import Domain


class LookUpTableBenchmark(Benchmark):
    """This class is a benchmark for discrete data where you can lookup a table for new inputs.
    It takes in a file with inputs and outputs and returns output for matching inputs from that file.

    Args:
        Benchmark: Subclass of the Benchmark function class.
    """

    def __init__(
        self,
        domain: Domain,
        LookUpTable: pd.DataFrame,
        **kwargs,
    ) -> None:
        """Initializes Molecule_benchmark. A class that connects to categorical molecular input.

        Args:
            domain (Domain): Domain of the inputs and outputs
            LookUpTable (pd.DataFrame): DataFrame containing the LookUp table.
        """
        super().__init__(**kwargs)

        self._domain = domain
        self.LookUpTable = LookUpTable
        self.domain.validate_experiments(self.LookUpTable)

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """return output values for matching SMILE candidates.

        Args:
            X (pd.DataFrame): Input values with input columns only.

        Returns:
            pd.DataFrame: output values from the LookUpTable. Columns are ouput keys and valid_output keys.
        """
        location = []
        for i in X.index:
            condition = np.ones(len(self.LookUpTable), dtype=bool)
            for k in self.domain.inputs.get_keys():
                condition = condition & (self.LookUpTable[k] == X[k][i]).to_numpy()
            if np.where(condition)[0].size != 0:
                location.append(np.where(condition)[0][0])
            else:
                raise ValueError(f"Input combination {i} not found in Look up table")
        X_temp = self.LookUpTable.loc[location]
        X_temp.index = pd.RangeIndex(len(X_temp))
        Y = pd.DataFrame()
        for k in self.domain.outputs.get_keys():
            Y[k] = X_temp[k]
            Y[f"valid_{k}"] = 1
        return Y
