import logging
import os
from typing import Callable, Dict, List, Optional

import pandas as pd

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.domain.api import Domain


# Create a folder for the log file, if not already exists.
if not os.path.exists("bofire_logs"):
    os.makedirs("bofire_logs")

# Create a Logger that safes console output.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
file_handler = logging.FileHandler("bofire_logs/aspen_benchmark.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Aspen_benchmark(Benchmark):
    """This class connects to a Aspen plus file that runs the desired process.
    It writes incoming input values into Aspen plus, runs the simulation and
    returns the results. When initializing this class, make sure not to block
    multiple Aspen plus licenses at once when is not absolutely needed.
    """

    def __init__(
        self,
        filename: str,
        domain: Domain,
        paths: Dict[str, str],
        additional_output_keys: Optional[List] = None,
        translate_into_aspen_readable: Optional[
            Callable[[Domain, pd.DataFrame], pd.DataFrame]
        ] = None,
        **kwargs,
    ) -> None:
        """Initializes Aspen_benchmark. A class that connects to Aspen plus.

        Args:
            filename (str): Filepath of the Aspen plus simulation file.
            domain (Domain): Domain of the benchmark setting inclunding bounds
                and information about input values.
            paths (dict[str, str]): A dictionary with the key value pairs
                "key_of_variable": "path_to_variable". The keys must be the
                same as provided in the domain.
            additional_output_keys: (list, optional): A list of additional output
                keys to be retrieved from Aspen. Defaults to None.
            translate_into_aspen_readable (Optional: Callable): A function that
                converts the columns of a candidate dataframe into integers or
                floats so Aspen plus is able to read their values.
            **kwargs: Additional arguments for the Benchmark class.

        Raises:
            ValueError: In case the number of provided variable names does not
                match the number of provided Aspen variable tree paths.

        """
        super().__init__(**kwargs)
        if os.path.exists(filename):
            self.filename = filename
        else:
            raise ValueError("Unable to find Aspen file " + filename)

        self.translate_into_aspen_readable = translate_into_aspen_readable
        self._domain = domain
        self.additional_output_keys = additional_output_keys or []

        for key in self.domain.inputs.get_keys() + self.domain.outputs.get_keys():
            # Check, if every input and output variable has a path to Aspen provided.
            if key not in paths:
                raise ValueError("Path for " + key + " is not provided.")

        self.paths = paths
        self.aspen_is_running = False

    # Start Aspen
    def start_aspen(self):
        """Starts Aspen plus and opens desired simulation file.

        Raises:
            ValueError: In case it is not possible to start Aspen plus.

        """
        import win32com.client as win32  # type: ignore

        logger.info("Starting Aspen plus")
        # Aspen should be accessible from every function as a global variable.
        global aspen
        try:
            aspen = win32.Dispatch("Apwn.Document")
            aspen.InitFromFile2(os.path.abspath(self.filename))
            self.aspen_is_running = True
        except OSError as e:
            logger.exception(e)
            raise ValueError(e)

    # Run simulation in Aspen
    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Function evaluation of the Aspen plus benchmark. Passes input values to Aspen, runs
        a simulation and reads the output values from Aspen. Checks for errors/warnings from Aspen.

        Args:
            candidates (pd.DataFrame): Input values that need to be passed to Aspen.
            Function can take either one input vector as a row in the input dataframe or multiple input vectors.

        Returns:
            pd.DataFrame: Output values from Aspen. The dataframe includes valid_(variable_name) columns for
            each output variable when the simulation went successful.

        """
        # Only start Aspen, when it is not already blocking.
        if self.aspen_is_running is False:
            self.start_aspen()

        # Make inputs Aspen-readable
        if self.translate_into_aspen_readable is not None:
            X = self.translate_into_aspen_readable(
                domain=self.domain,  # type: ignore
                candidates=candidates.copy(),
            )
        else:
            X = candidates

        y_outputs = {
            k: []
            for key in self.domain.outputs.get_keys()
            for k in (key, "valid_" + key)
        }
        add_outputs = {key: [] for key in self.additional_output_keys}

        # Iterate through dataframe rows to retrieve multiple input vectors. Running separate simulations for each.
        for index, row in X.iterrows():
            logger.info("Writing inputs into Aspen")
            # Write input variables corresping to columns into aspen according to predefined paths.
            for key in self.domain.inputs.get_keys():
                try:
                    aspen.Tree.FindNode(self.paths.get(key)).Value = row[key]
                except ConnectionAbortedError:
                    logger.exception("Not able to write " + key + " into Aspen.")
                    raise ValueError("Not able to write " + key + " into Aspen.")

            # Reset Aspen simulation
            aspen.Reinit()
            # Start new Aspen simulation
            logger.info("Aspen simulation run " + str(index))
            aspen.Engine.Run2()
            logger.info("Simulation done.")

            # Retrieve outputs from Aspen and write into data frame
            logger.info("Retrieving outputs from Aspen.")
            try:
                # Check for errors during simulation in Aspen that disqualify the results
                status = aspen.Tree.FindNode(
                    "\\Data\\Results Summary\\Run-Status\\Output\\UOSSTAT2",
                ).Value

                if status != 8:
                    if status == 9:
                        logger.error(
                            "Result"
                            + " does not converge. Simulation status: "
                            + str(status),
                        )
                    elif status == 10:
                        logger.warning(
                            "Result"
                            + " gives an Aspen warning. Simulation status: "
                            + str(status),
                        )
                    else:
                        logger.warning("Unknown simulation status: " + str(status))

                for key in self.domain.outputs.get_keys():
                    y_outputs[key].append(
                        aspen.Tree.FindNode(self.paths.get(key)).Value,
                    )
                    if status == 8:
                        # Result is valid and add valid_var = 1
                        # Status = 8 corresponds to a valid result that should be kept, 10 is a warning, 9 does not converge
                        y_outputs[f"valid_{key}"].append(1)
                    else:
                        y_outputs[f"valid_{key}"].append(0)

                for key in self.additional_output_keys:
                    add_outputs[key].append(
                        aspen.Tree.FindNode(self.paths.get(key)).Value,
                    )

            except ConnectionAbortedError:
                logger.exception("Not able to retrieve values from Aspen.")
                raise ValueError("Not able to retrieve values from Aspen.")

        Y = pd.DataFrame(y_outputs)
        Z = pd.DataFrame(add_outputs)
        XYZ = pd.concat([candidates, Y, Z], axis=1)
        YZ = pd.concat([Y, Z], axis=1)
        logger.info("Simluation completed. Results:")
        logger.info(XYZ)
        return YZ
