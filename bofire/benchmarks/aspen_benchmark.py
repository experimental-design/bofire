import logging
import os

import pandas as pd
import win32com.client as win32

from bofire.benchmarks.benchmark import Benchmark
from bofire.domain import Domain

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

    def __init__(
        self, filename: str, domain: Domain, paths: list[list[dict[str, str]]]
    ) -> None:
        """Initializes Aspen_benchmark. A class that connects to Aspen plus.

        Args:
            filename (str): Filepath of the Aspen plus simulation file.
            domain (Domain): Domain of the benchmark setting inclunding bounds and information about input values.
            paths (list[list[dict[str, str]]]): 2xn list with dictionaries containing the filepaths to the variables in the Aspen variable tree
            in Aspen plus {"name": "path"}.

        Raises:
            SystemExit: In case the number of provided variable names does not match the number of provided Aspen variable tree paths.
        """

        self.filename = filename
        self._domain = domain
        self.paths = paths

        self.aspen_is_running = False

        # Get the variable names (keys) from the domain to access them later easily.
        self.keys = [[], []]  # keys[0] for x, keys[1] for y
        for feature in self.domain.inputs.features:
            self.keys[0].append(feature.key)
        for feature in self.domain.outputs.features:
            self.keys[1].append(feature.key)

        # Check, if number of paths matches number of variables
        for key_list, path_list in zip(self.keys, self.paths):
            if len(key_list) != len(path_list):
                log_string = (
                    "Number of variables ("
                    + str(len(key_list))
                    + ") names must match number of paths ("
                    + str(len(path_list))
                    + "). \n"
                    + "Variables: "
                    + str(key_list)
                    + "\nPaths: "
                    + str(path_list)
                )
                logger.exception(log_string)
                raise SystemExit(log_string)

    # Start Aspen
    def start_aspen(self):
        """Starts Aspen plus and opens desired simulation file.

        Raises:
            SystemExit: In case it is not possible to start Aspen plus.
        """
        logger.info("Starting Aspen plus")
        # Aspen should be accessible from every function as a global variable.
        global aspen
        try:
            aspen = win32.Dispatch("Apwn.Document")
            aspen.InitFromFile2(os.path.abspath(self.filename))
            self.aspen_is_running = True
        except:
            log_string = "Unable to start Aspen plus."
            logger.exception(log_string)
            raise SystemExit(log_string)

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

        # TODO: Initiate dataframe more memory efficient without interfering with the pd.concat method.
        Y = pd.DataFrame()
        # Iterate through dataframe rows to retrieve multiple input vectors. Running seperate simulations for each.
        for index, row in candidates.iterrows():
            logger.info("Writing inputs into Aspen")
            # Write input variables corresping to columns into aspen according to predefined paths.
            for key, path in zip(self.keys[0], self.paths[0]):
                try:
                    aspen.Tree.FindNode(path.get(key)).Value = row[key]
                except:
                    logger.exception("Not able to write " + key + " into Aspen.")

            # Reset Aspen simulation
            aspen.Reinit()
            # Start new Aspen simulation
            logger.info("Aspen simulation run " + str(index))
            aspen.Engine.Run2()
            logger.info("Simulation done.")

            # Retrieve outputs from Aspen and write into data frame
            logger.info("Retrieving outputs from Aspen.")
            for key, path in zip(self.keys[1], self.paths[1]):
                try:
                    Y.at[index, key] = aspen.Tree.FindNode(path.get(key)).Value

                    # Check for errors during simulation in Aspen that disqualify the y_value
                    status = aspen.Tree.FindNode(
                        "\\Data\\Results Summary\\Run-Status\\Output\\UOSSTAT2"
                    ).Value
                    if status == 8:
                        # Result is valid and add valid_var = 1
                        # Status = 8 corresponds to a valid result that should be kept, 10 is a warning, 9 does not converge
                        Y.at[index, "valid_" + key] = 1
                    else:
                        Y.at[index, "valid_" + key] = 0
                        value = Y.at[index, key]
                        if status == 9:
                            logger.error(
                                "Result of "
                                + str(value)
                                + " does not converge. Simulation status: "
                                + str(status)
                            )
                        elif status == 10:
                            logger.warning(
                                "Result of "
                                + str(value)
                                + " gives an Aspen warning. Simulation status: "
                                + str(status)
                            )
                        else:
                            logger.warning("Unknown simulation status: " + str(status))

                except:
                    logger.exception("Not able to retrieve " + key + " from Aspen.")

        logger.info("Simluation completed. Results:")
        logger.info(Y)
        return Y

    def __del__(self):
        # Can cause trouble. The qehvi takes time to generate the next input, meanwhile the python cleaner can close the class.
        """Deinitializes class and closes Aspen plus."""
        try:
            aspen.Close()
            logger.info("Aspen closed.")
        except:
            pass
