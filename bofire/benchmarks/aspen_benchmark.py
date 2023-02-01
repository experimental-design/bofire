import logging
import os

import pandas as pd
import win32com.client as win32

from bofire.benchmarks.benchmark import Benchmark
from bofire.domain import Domain

# Create a folder for the log file, if not alredy exists.
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
    It writes incoming input values into Aspen plus, runs the simulation and returns the results.
    When initializing this class, make sure not to block multiple Aspen plus licenses at once
    when is not absolutely needed.

    Args:
        Benchmark: Subclass of the Benchmark function class.
    """

    def __init__(self, filename: str, domain: Domain, paths: dict[str, str]) -> None:
        """Initializes Aspen_benchmark. A class that connects to Aspen plus.

        Args:
            filename (str): Filepath of the Aspen plus simulation file.
            domain (Domain): Domain of the benchmark setting inclunding bounds and information about input values.
            paths (dict[str, str]): A dictionary with the key value pairs "key_of_variable": "path_to_variable".
            The keys must be the same as provided in the domain.

        Raises:
            ValueError: In case the number of provided variable names does not match the number of provided Aspen variable tree paths.
        """
        if os.path.exists(filename):
            self.filename = filename
        else:
            raise ValueError("Unable to find Aspen file " + filename)

        self._domain = domain
        # Get the variable names (keys) from the domain to access them later easily.
        self.keys = [self.domain.inputs.get_keys(), self.domain.outputs.get_keys()]
        # keys[0] for x, keys[1] for y

        for key_list in self.keys:
            # Check, if every input and output variable has a path to Aspen provided.
            for key in key_list:
                if key not in paths.keys():
                    raise ValueError("Path for " + key + " is not provided.")

        # Check, if number of paths matches number of variables
        if (len(self.keys[0]) + len(self.keys[1])) != len(paths.items()):
            log_string = (
                "Number of variables ("
                + str(len(self.keys[0]) + len(self.keys[1]))
                + ") names must match number of paths ("
                + str(len(paths.items()))
                + "). \n"
                + "Variables: "
                + str(self.keys)
                + "\nPaths: "
                + str(paths.items())
            )
            raise ValueError(log_string)

        self.paths = paths
        self.aspen_is_running = False

    # Start Aspen
    def start_aspen(self):
        """Starts Aspen plus and opens desired simulation file.

        Raises:
            ValueError: In case it is not possible to start Aspen plus.
        """
        logger.info("Starting Aspen plus")
        # Aspen should be accessible from every function as a global variable.
        global aspen
        try:
            aspen = win32.Dispatch("Apwn.Document")
            aspen.InitFromFile2(os.path.abspath(self.filename))
            self.aspen_is_running = True
        except OSError:
            log_string = "Unable to start Aspen plus."
            logger.exception(log_string)
            raise ValueError(log_string)

    def translate_into_aspen_readable(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Tranlates the input data that may contain strings and other datatypes used by bofire
        that Aspen plus is not able to read natively.

        Returns:
            pd.DataFrame: Input data ready to be given to Aspen plus.
        """
        for feature in self.domain.inputs.features:
            if feature.type == "CategoricalDescriptorInput":
                key = feature.key
                candidates[key] = candidates[key].astype(int)
        return candidates

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
        candidates_aspen_readable = self.translate_into_aspen_readable(
            candidates=candidates
        )

        y_outputs = {}
        for key in self.keys[1]:
            y_outputs[key] = []
        # Iterate through dataframe rows to retrieve multiple input vectors. Running seperate simulations for each.
        for index, row in candidates_aspen_readable.iterrows():
            logger.info("Writing inputs into Aspen")
            # Write input variables corresping to columns into aspen according to predefined paths.
            for key in self.keys[0]:
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
            for key in self.keys[1]:
                try:
                    y_outputs[key].append(
                        aspen.Tree.FindNode(self.paths.get(key)).Value
                    )
                    # Check for errors during simulation in Aspen that disqualify the y_value
                    status = aspen.Tree.FindNode(
                        "\\Data\\Results Summary\\Run-Status\\Output\\UOSSTAT2"
                    ).Value
                    if status == 8:
                        # Result is valid and add valid_var = 1
                        # Status = 8 corresponds to a valid result that should be kept, 10 is a warning, 9 does not converge
                        y_outputs["valid_" + key] = 1
                    else:
                        y_outputs["valid_" + key] = 0
                        if status == 9:
                            logger.error(
                                "Result"
                                + " does not converge. Simulation status: "
                                + str(status)
                            )
                        elif status == 10:
                            logger.warning(
                                "Result"
                                + " gives an Aspen warning. Simulation status: "
                                + str(status)
                            )
                        else:
                            logger.warning("Unknown simulation status: " + str(status))

                except ConnectionAbortedError:
                    logger.exception("Not able to retrieve " + key + " from Aspen.")
                    raise ValueError("Not able to retrieve " + key + " from Aspen.")
        Y = pd.DataFrame(y_outputs)
        XY = pd.concat([candidates, Y], axis=1)
        logger.info("Simluation completed. Results:")
        logger.info(XY)
        return Y
