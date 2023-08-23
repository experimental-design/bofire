import os
from typing import Dict

import pandas as pd

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import CategoricalMolecularInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective


class Molecule_benchmark(Benchmark):
    """This class connects to categorical molecular input benchmark in https://github.com/leojklarner/gauche.
    It takes in a file with SMILES and data and returns output for matching SMILES from that file.

    Args:
        Benchmark: Subclass of the Benchmark function class.
    """

    def __init__(
        self,
        filename: str,
        benchmark: Dict,
        **kwargs,
    ) -> None:
        """Initializes Molecule_benchmark. A class that connects to categorical molecular input.

        Args:
            filename (str): Filepath of the input data. should be in .csv format.
            benchmark (Dict): dictionary containing input feature ("input") name and output label name ("output").
        Raises:
            ValueError: In case the filename does not exist or not in .csv format
        """
        super().__init__(**kwargs)
        if os.path.exists(filename):
            self.filename = filename
        else:
            raise ValueError("Unable to find file " + filename)
        if not self.filename.endswith(".csv"):
            raise ValueError("file not in .csv format")
        df = pd.read_csv(self.filename)
        self.benchmark = benchmark
        self.main_file = pd.DataFrame(
            columns=[self.benchmark["input"], self.benchmark["output"]]
        )
        nans = df[self.benchmark["output"]].isnull().to_list()
        nan_indices = [nan for nan, x in enumerate(nans) if x]
        self.main_file[self.benchmark["input"]] = (
            df[self.benchmark["input"]].drop(nan_indices).to_list()
        )
        self.main_file[self.benchmark["output"]] = (
            df[self.benchmark["output"]].dropna().to_numpy().reshape(-1, 1)
        )
        input_feature = CategoricalMolecularInput(
            key=self.benchmark["input"],
            categories=list(set(self.main_file[self.benchmark["input"]].to_list())),
        )
        objective = MaximizeObjective(
            w=1.0,
        )
        inputs = Inputs(features=[input_feature])
        output_feature = ContinuousOutput(
            key=self.benchmark["output"], objective=objective
        )
        outputs = Outputs(features=[output_feature])
        self._domain = Domain(inputs=inputs, outputs=outputs)

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """return output values for matching SMILE candidates.

        Args:
            X (pd.DataFrame): Input values. Column is benchmark["input"]

        Returns:
            pd.DataFrame: output values of the function. Columns are benchmark["output"] and valid_benchmark["output"].
        """
        X_temp = self.main_file.loc[
            self.main_file[self.benchmark["input"]].isin(
                X[self.benchmark["input"]].values
            )
        ]
        X_temp.index = pd.RangeIndex(len(X_temp))
        Y = pd.DataFrame(
            {
                self.benchmark["output"]: X_temp[self.benchmark["output"]],
                f"valid_{self.benchmark['output']}": 1,
            }
        )
        return Y
