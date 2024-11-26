import pandas as pd

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.domain.api import Domain


class LookupTableBenchmark(Benchmark):
    """This class is a benchmark for discrete data where you can lookup a table for new inputs.
    It takes in a file with inputs and outputs and returns output for matching inputs from that file.

    Args:
        Benchmark: Subclass of the Benchmark function class.

    """

    def __init__(
        self,
        domain: Domain,
        lookup_table: pd.DataFrame,
        **kwargs,
    ) -> None:
        """Initializes Molecule_benchmark. A class that connects to categorical molecular input.

        Args:
            domain (Domain): Domain of the inputs and outputs
            lookup_table (pd.DataFrame): DataFrame containing the LookUp table.
            **kwargs: Additional arguments for the Benchmark class.

        """
        super().__init__(**kwargs)

        self._domain = domain
        self.lookup_table = lookup_table
        self.domain.validate_experiments(self.lookup_table)

    def _f(self, sampled: pd.DataFrame, **kwargs) -> pd.DataFrame:  # type: ignore
        """Return output values for matching SMILE candidates.

        Args:
            sampled (pd.DataFrame): Input values with input columns only.
            **kwargs: Allow additional unused arguments to prevent errors.

        Returns:
            pd.DataFrame: output values from the LookUpTable. Columns are output keys and valid_output keys.

        """
        X = sampled.copy()
        X["proxy_index"] = X.index
        X_temp = pd.merge(
            X,
            self.lookup_table,
            on=self.domain.inputs.get_keys(),
            how="left",
        ).dropna()
        df = pd.merge(X, X_temp, how="left", indicator=True)
        if (
            df.loc[df._merge == "left_only", df.columns != "_merge"].proxy_index.size
            != 0
        ):
            indices = df.loc[
                df._merge == "left_only",
                df.columns != "_merge",
            ].proxy_index.to_list()
            raise ValueError(f"Input combination {indices} not found in Look up table")
        Y = X_temp[
            self.domain.outputs.get_keys()
            + [f"valid_{k}" for k in self.domain.outputs.get_keys()]
        ]
        Y.index = pd.RangeIndex(len(Y))
        return Y
